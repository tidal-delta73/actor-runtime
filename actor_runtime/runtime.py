"""In-process, deterministic in-memory actor runtime.

The runtime is deliberately single-process and synchronous: callers register
named actors, deliver messages to their mailboxes and explicitly advance
processing with :meth:`ActorRuntime.run`. Scheduling is fully determined by
registration order, message priority and delivery id -- never by wall-clock
time, randomness or thread interleaving.

Time is just another explicitly replayed input: a logical clock starts at 0
and only moves when the caller calls :meth:`ActorRuntime.advance`. Timed
deliveries are held outside the mailboxes until their deadline; at a deadline
that is also their expiry tick expiry wins, so such a message never reaches a
handler.

The runtime state can be exported at any moment as deterministic,
persistable bytes (:meth:`ActorRuntime.export_snapshot`) and later resumed
as a new runtime from those bytes plus a handler mapping
(:meth:`ActorRuntime.restore_snapshot`). Snapshots are pure data: they
never contain handlers or any executable code, and restoring never imports
modules or executes objects carried by the bytes.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

__all__ = [
    "ActorContext",
    "ActorDataCopyError",
    "ActorExecutionError",
    "ActorRuntime",
    "AdvanceResult",
    "DedupResult",
    "FailureRecord",
    "SnapshotError",
    "SupervisionError",
    "TraceEntry",
]


class TraceEntry(NamedTuple):
    """One completed message processing, in completion order.

    ``state_before`` and ``state_after`` are independent snapshots; mutating
    them never affects the runtime or later queries.
    """

    message_id: int
    actor_name: str
    priority: int
    state_before: Any
    state_after: Any


class AdvanceResult(NamedTuple):
    """Outcome of one :meth:`ActorRuntime.advance` call.

    ``time`` is the clock value after the advance. ``released`` and
    ``expired`` are the message ids of the deliveries that respectively
    entered a mailbox or were dropped. Both lists are stably ordered by the
    tick the event happened at and then by message id.
    """

    time: int
    released: tuple[int, ...]
    expired: tuple[int, ...]


class DedupResult(NamedTuple):
    """Outcome of an idempotent :meth:`ActorRuntime.send_once` delivery.

    ``message_id`` is the id of the first delivery accepted for the
    ``(target, delivery_key)`` pair; it is returned again by every later
    call with the same pair. ``accepted`` is ``True`` only for that first
    call -- duplicate calls neither enqueue nor consume an id.
    """

    message_id: int
    accepted: bool


class FailureRecord(NamedTuple):
    """A supervised actor failure awaiting a supervision decision.

    Produced when a supervised actor's handler raises an :class:`Exception`
    or an :class:`ActorDataCopyError` while processing ``message_id``. The
    actor is paused and the failing message stays in its mailbox with its
    original id and priority until the caller resolves the record as the
    actor's current ``supervisor`` via :meth:`ActorRuntime.resolve_failure`.

    ``supervision_path`` is the chain from the failed actor to the tree
    root, including both ends; it is captured once and never shortened, so
    an escalated record keeps showing where the failure happened. Records
    are immutable, and the lists returned by :meth:`ActorRuntime.failures`
    are independent copies.
    """

    actor_name: str
    message_id: int
    supervisor: str | None
    error_type: str
    error_text: str
    supervision_path: tuple[str, ...]


class SupervisionError(Exception):
    """Raised when a supervision decision cannot be honoured.

    A root actor -- one registered without a supervisor -- cannot escalate,
    since there is no parent above it: :meth:`ActorRuntime.resolve_failure`
    raises this error for ``"escalate"`` on such a record while leaving the
    record, the pause and every mailbox exactly as they were.
    """


class ActorDataCopyError(Exception):
    """Raised when a message or state cannot be deep-copied.

    :meth:`ActorRuntime.send` raises it directly when the message cannot be
    copied. During :meth:`ActorRuntime.run`, an unsupervised actor's copy
    failure is reported as an :class:`ActorExecutionError` whose
    ``original`` is an ``ActorDataCopyError``; a supervised actor is paused
    with a :class:`FailureRecord` whose ``error_type`` is
    ``"ActorDataCopyError"`` instead. Either way nothing is committed: no
    mailbox change, no state change, no derived delivery, no trace entry and
    no consumed message id. ``what`` names the data being copied and
    ``original`` is the exception the copy raised.
    """

    def __init__(self, what: str, original: BaseException):
        self.what = what
        self.original = original
        super().__init__(
            f"failed to deep-copy {what}: "
            f"{type(original).__name__}: {original}"
        )


class ActorExecutionError(Exception):
    """Raised by :meth:`ActorRuntime.run` when an unsupervised actor fails.

    The failing message stays unacknowledged in its actor's mailbox and no
    state or derived delivery is committed for it. Messages completed earlier
    in the same :meth:`~ActorRuntime.run` call remain committed.

    An actor registered with a supervisor is handled differently: raises
    from its handler never escape :meth:`~ActorRuntime.run` -- the actor is
    paused, a :class:`FailureRecord` is produced and the run ends normally
    returning the number of earlier completions.
    """

    def __init__(self, actor_name: str, message_id: int, original: BaseException):
        self.actor_name = actor_name
        self.message_id = message_id
        self.original = original
        super().__init__(
            f"actor {actor_name!r} failed while handling message {message_id}: "
            f"{type(original).__name__}: {original}"
        )


class SnapshotError(Exception):
    """Raised when a snapshot cannot be exported or restored.

    :meth:`ActorRuntime.export_snapshot` raises it when the runtime holds
    data outside the snapshot domain: sets, custom objects, non-string
    mapping keys, NaN or infinite floats or circular references. The
    runtime is left completely unchanged -- states, queues, clock, dedup
    records, trace and numbering all stay as they were.

    :meth:`ActorRuntime.restore_snapshot` raises it when the bytes are
    truncated, tampered with, of an unsupported version, missing fields or
    internally inconsistent. No runtime is produced in that case.
    """


class ActorContext:
    """Handle passed to message handlers for sending derived messages.

    Messages sent through :meth:`send` are buffered and only enter the target
    mailboxes after the current handler returns successfully.
    """

    def __init__(self, runtime: "ActorRuntime"):
        self._runtime = runtime
        self._pending: list[tuple[str, Any]] = []

    def send(self, target: str, message: Any) -> None:
        """Buffer a message for a registered actor with default priority 0.

        Raises :class:`LookupError` if ``target`` is not registered. Raising
        inside the handler aborts the current processing exactly like any
        other handler exception.
        """
        # The same target boundary every external entry uses; the target is
        # re-checked here so an unknown actor aborts the whole processing
        # before commit. Only (target, message) is buffered: priority is a
        # property of an accepted delivery, so derived mail never inherits
        # the triggering message's priority.
        self._runtime._resolve_actor(target)
        self._pending.append((target, message))


Handler = Callable[[Any, Any, ActorContext], Any]


def _copy_data(what: str, value: Any) -> Any:
    """Deep-copy ``value``, reporting any failure as ActorDataCopyError."""
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise ActorDataCopyError(what, exc) from exc


def _validate_priority(priority: Any) -> None:
    """Check the priority argument shared by every delivery entry point."""
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise TypeError("priority must be an integer")


def _validate_delay(delay: Any) -> None:
    """Check a schedule ``delay``: non-boolean, non-negative integer."""
    if not isinstance(delay, int) or isinstance(delay, bool):
        raise TypeError("delay must be an integer")
    if delay < 0:
        raise ValueError("delay must be a non-negative integer")


def _validate_ttl(ttl: Any) -> None:
    """Check a schedule ``ttl``: ``None`` or a non-boolean positive integer."""
    if ttl is None:
        return
    if not isinstance(ttl, int) or isinstance(ttl, bool):
        raise TypeError("ttl must be None or an integer")
    if ttl <= 0:
        raise ValueError("ttl must be a positive integer or None")


def _validate_delivery_key(delivery_key: Any) -> None:
    """Check a send_once key: non-empty string."""
    if not isinstance(delivery_key, str):
        raise TypeError("delivery_key must be a string")
    if delivery_key == "":
        raise ValueError("delivery_key must be a non-empty string")


class _Actor:
    __slots__ = (
        "name", "state", "handler", "mailbox", "order", "dedup_keys",
        "supervisor_name", "paused",
    )

    def __init__(self, name: str, state: Any, handler: Handler, order: int,
                 supervisor_name: str | None = None):
        self.name = name
        self.state = state
        self.handler = handler
        self.order = order
        # Heap of (-priority, message_id, message); message_id is unique, so
        # the message object never participates in comparison.
        self.mailbox: list[tuple[int, int, Any]] = []
        # delivery_key -> first delivery's message id, for send_once. A
        # reservation is made at acceptance and never withdrawn, so the key
        # stays deduplicated while its message is pending, completed or
        # waiting in the mailbox after a handler failure.
        self.dedup_keys: dict[str, int] = {}
        # Name of the direct supervisor, or None for a tree root. The
        # supervision tree never participates in scheduling.
        self.supervisor_name = supervisor_name
        # A supervised actor is paused while a FailureRecord for it is
        # pending: new mail still arrives, but run skips the actor.
        self.paused = False


class _ScheduledDelivery:
    """A timed delivery held outside the mailboxes until its deadline."""

    __slots__ = (
        "message_id", "actor", "priority", "message",
        "scheduled_at", "release_at", "expire_at",
    )

    def __init__(self, message_id: int, actor: _Actor, priority: int,
                 message: Any, scheduled_at: int, release_at: int,
                 expire_at: int | None):
        self.message_id = message_id
        self.actor = actor
        self.priority = priority
        self.message = message
        self.scheduled_at = scheduled_at
        self.release_at = release_at
        # None means the delivery never expires.
        self.expire_at = expire_at


class ActorRuntime:
    """Deterministic in-memory actor runtime.

    Every delivery entry point -- :meth:`send`, :meth:`send_once`,
    :meth:`schedule` (its ``delay == 0`` case), timed release in
    :meth:`advance` and the derived-mail commit in :meth:`run` -- crosses the
    same internal boundary: validate, resolve the target, deep-copy the
    payload, take an id from the single runtime sequence and submit to the
    destination mailbox. The helpers below are that boundary; failures
    before :meth:`_admit`/:meth:`_push_mail` leave every structure untouched.
    """

    def __init__(self) -> None:
        self._actors: dict[str, _Actor] = {}
        self._trace: list[TraceEntry] = []
        self._next_message_id = 1
        self._clock = 0
        # Pending timed deliveries, kept in scheduling (message-id) order;
        # _release_due scans and rebuilds it.
        self._scheduled: list[_ScheduledDelivery] = []
        # Unresolved supervised failures in production order. Each paused
        # actor has exactly one record here; retry/drop remove it, escalate
        # may move it to the actor's parent.
        self._failures: list[FailureRecord] = []

    # -- shared delivery boundary -----------------------------------------

    def _resolve_actor(self, target: str) -> _Actor:
        """Resolve a target name, the single LookupError boundary.

        Used by every entry point, including the ActorContext.send check
        inside a handler and the read-only queries.
        """
        actor = self._actors.get(target)
        if actor is None:
            raise LookupError(f"unknown actor: {target!r}")
        return actor

    def _issue_message_id(self) -> int:
        """Take the next runtime-wide monotonic message id.

        All accepted deliveries -- plain, idempotent, scheduled and derived
        -- draw from this one sequence.
        """
        message_id = self._next_message_id
        self._next_message_id += 1
        return message_id

    def _push_mail(self, actor: _Actor, priority: int, message_id: int,
                   stored: Any) -> None:
        """Submit an already-copied message with a known id to a mailbox."""
        heapq.heappush(actor.mailbox, (-priority, message_id, stored))

    def _admit(self, actor: _Actor, priority: int, stored: Any) -> int:
        """Number and submit a payload that has already passed copy/validate.

        This is the common tail of a freshly accepted external delivery
        (:meth:`send`, a zero-delay :meth:`schedule`) and of a derived
        delivery committed at the end of a successful processing.
        """
        message_id = self._issue_message_id()
        self._push_mail(actor, priority, message_id, stored)
        return message_id

    # -- registration -----------------------------------------------------

    def register(self, name: str, initial_state: Any, handler: Handler,
                 supervisor: str | None = None) -> None:
        """Register a named actor, optionally under a direct ``supervisor``.

        With ``supervisor=None`` (the default) the actor is a supervision
        tree root and keeps the baseline failure behaviour: a handler failure
        aborts :meth:`run` with :class:`ActorExecutionError`. With a
        supervisor name the actor is supervised: a handler failure pauses it
        and produces a :class:`FailureRecord` instead of raising.

        Raises :class:`ValueError` when ``name`` is empty or already
        registered, when ``supervisor`` is empty or names the actor itself
        (self-supervision); :class:`TypeError` when ``supervisor`` is neither
        a string nor ``None``; :class:`LookupError` when ``supervisor`` names
        an unregistered actor. A failed registration changes nothing: no
        state is stored and registration order is unaffected.
        """
        if not isinstance(name, str) or name == "":
            raise ValueError("actor name must be a non-empty string")
        supervisor_name = self._validate_supervisor_ref(name, supervisor)
        if name in self._actors:
            raise ValueError(f"actor already registered: {name!r}")
        self._actors[name] = _Actor(
            name, copy.deepcopy(initial_state), handler,
            len(self._actors), supervisor_name,
        )

    def _validate_supervisor_ref(self, name: str,
                                 supervisor: Any) -> str | None:
        """Check the optional supervisor argument of :meth:`register`."""
        if supervisor is None:
            return None
        if not isinstance(supervisor, str):
            raise TypeError("supervisor must be a string or None")
        if supervisor == "":
            raise ValueError("supervisor must be a non-empty string")
        if supervisor == name:
            raise ValueError(f"actor cannot supervise itself: {name!r}")
        if supervisor not in self._actors:
            raise LookupError(f"unknown supervisor: {supervisor!r}")
        return supervisor

    def _supervision_path(self, actor: _Actor) -> tuple[str, ...]:
        """Walk the direct-supervisor links from ``actor`` to its root.

        Registrations never disappear, so every link resolves. The tuple
        starts with the failed actor and ends with the tree root.
        """
        path = [actor.name]
        current = actor
        while current.supervisor_name is not None:
            current = self._actors[current.supervisor_name]
            path.append(current.name)
        return tuple(path)

    # -- delivery ---------------------------------------------------------

    def send(self, target: str, message: Any, priority: int = 0) -> int:
        """Deliver a message to ``target``'s mailbox and return its id.

        Ids are monotonically increasing within the runtime. Higher
        ``priority`` is processed first; equal priority is processed in id
        order. Raises :class:`TypeError` if ``priority`` is not an integer,
        :class:`LookupError` if ``target`` is unknown and
        :class:`ActorDataCopyError` if ``message`` cannot be deep-copied;
        a failed delivery consumes no id and changes no mailbox.
        """
        _validate_priority(priority)
        actor = self._resolve_actor(target)
        # Copy before assigning an id: a failed copy leaves the mailbox,
        # the trace, the actor state and the next message id untouched.
        stored = _copy_data("message", message)
        return self._admit(actor, priority, stored)

    def send_once(self, target: str, delivery_key: str, message: Any,
                  priority: int = 0) -> DedupResult:
        """Deliver a message at most once per ``(target, delivery_key)``.

        Behaves exactly like :meth:`send` on the first call for a pair: the
        message is deep-copied, gets the next runtime-wide monotonic id and is
        enqueued by the usual priority rules; the result reports that id with
        ``accepted=True``. Every later call with the same target and key gets
        :class:`DedupResult` with the *first* id and ``accepted=False``: it
        neither enqueues nor consumes an id, leaves the original delivery's
        state, trace and priority untouched, and never copies or compares the
        passed message -- it is a mere call placeholder. The scope is the
        target/key pair, so different actors may freely reuse the same key.

        The dedup record lives for this runtime instance's whole lifetime: it
        already covers the key while the first message is pending, stays in
        place if its handler fails (the message keeps waiting in the mailbox
        under the usual retry/rollback semantics) and remains after the
        message completes. A handler failure therefore still results in at
        most one trace entry: re-submitting the same key only confirms the
        duplicate, while the original message is retried by a later
        :meth:`run`.

        Raises :class:`TypeError` if ``priority`` is not a non-boolean integer
        or ``delivery_key`` is not a string, :class:`ValueError` if
        ``delivery_key`` is empty and :class:`LookupError` if ``target`` is
        unknown; these checks all precede the duplicate lookup. On the first
        call an uncopyable message raises :class:`ActorDataCopyError`: no key
        is reserved, no id is consumed and nothing is enqueued, so after
        fixing the data the same key can be used to retry.
        """
        _validate_priority(priority)
        _validate_delivery_key(delivery_key)
        actor = self._resolve_actor(target)
        existing = actor.dedup_keys.get(delivery_key)
        if existing is not None:
            # Duplicate confirmation: the placeholder message is neither
            # copied nor compared, nothing is enqueued and no id consumed.
            return DedupResult(existing, False)
        # Copy before reserving the key or consuming an id, mirroring send:
        # a failed copy leaves the key free, so the corrected call is the
        # first accepted delivery for it.
        stored = _copy_data("message", message)
        message_id = self._issue_message_id()
        actor.dedup_keys[delivery_key] = message_id
        self._push_mail(actor, priority, message_id, stored)
        return DedupResult(message_id, True)

    def schedule(self, target: str, message: Any, delay: int,
                 ttl: int | None = None, priority: int = 0) -> int:
        """Schedule ``message`` for ``target`` at a future logical tick.

        Returns the delivery's id, taken from the same runtime-wide monotonic
        sequence as :meth:`send`. ``delay`` is the number of ticks counted
        from the current clock: the message is released when the clock reaches
        ``now + delay``. A ``delay`` of 0 releases it immediately unless it is
        already expired. ``ttl`` is the number of ticks the delivery stays
        alive after scheduling; when the clock reaches ``scheduled_at + ttl``
        the delivery expires without ever entering a mailbox or reaching a
        handler. At a tick that is both deadline and expiry, expiry wins.

        Raises :class:`TypeError` for wrong argument types, :class:`ValueError`
        for out-of-range ``delay``/``ttl``, :class:`LookupError` for an
        unknown ``target`` and :class:`ActorDataCopyError` if ``message``
        cannot be deep-copied; a failed schedule consumes no id and queues
        nothing.
        """
        _validate_priority(priority)
        _validate_delay(delay)
        _validate_ttl(ttl)
        actor = self._resolve_actor(target)
        # Copy before assigning an id: a failed copy queues nothing and
        # consumes no id, exactly like send.
        stored = _copy_data("message", message)
        message_id = self._issue_message_id()
        scheduled_at = self._clock
        release_at = scheduled_at + delay
        expire_at = None if ttl is None else scheduled_at + ttl
        if delay == 0:
            # ttl is a strictly positive integer, so a zero-delay delivery is
            # never already expired: it enters the mailbox straight away.
            self._push_mail(actor, priority, message_id, stored)
        else:
            self._scheduled.append(
                _ScheduledDelivery(
                    message_id, actor, priority, stored,
                    scheduled_at, release_at, expire_at,
                )
            )
        return message_id

    # -- logical clock ----------------------------------------------------

    def clock(self) -> int:
        """Return the current logical tick (initially 0)."""
        return self._clock

    def advance(self, ticks: int) -> AdvanceResult:
        """Move the clock forward ``ticks`` and settle due deliveries.

        ``ticks`` must be a non-boolean positive integer; otherwise
        :class:`TypeError`/:class:`ValueError` is raised and nothing changes.
        The clock moves to the new tick first; then every pending delivery is
        settled: one whose deadline has been reached is released into its
        actor's mailbox with its scheduled id and priority, while one whose
        ttl has elapsed expires and is dropped. When both happen at the same
        tick expiry wins, so the message never reaches a handler. Released and
        expired ids are returned stably ordered by event tick and then by id.
        Releasing never invokes a handler and never writes a trace entry; use
        :meth:`run` afterwards to process the mail.
        """
        if not isinstance(ticks, int) or isinstance(ticks, bool):
            raise TypeError("ticks must be an integer")
        if ticks <= 0:
            raise ValueError("ticks must be a positive integer")

        new_time = self._clock + ticks
        released: list[tuple[int, int]] = []  # (release_at, id)
        expired: list[tuple[int, int]] = []   # (expire_at, id)
        remaining: list[_ScheduledDelivery] = []
        # _scheduled stays in message-id (scheduling) order; the event lists
        # are sorted by event tick and id afterwards, so the scan order is
        # irrelevant to the stable result.
        for delivery in self._scheduled:
            if (
                delivery.expire_at is not None
                and delivery.expire_at <= new_time
                and delivery.expire_at <= delivery.release_at
            ):
                # The ttl elapsed no later than the deadline: expiry at
                # expire_at. Equality with the deadline is decided in favour
                # of expiry, so a same-tick message never reaches a handler.
                expired.append((delivery.expire_at, delivery.message_id))
                continue
            if new_time >= delivery.release_at:
                # The deadline was reached during this advance. Even if the
                # ttl elapses at a later tick inside the same jump, the
                # delivery was released at its deadline and is ordinary
                # mailbox mail from then on; the ttl only guards the wait.
                # Submit through the same mailbox boundary as a fresh send,
                # keeping the id and priority assigned at schedule time.
                self._push_mail(
                    delivery.actor, delivery.priority,
                    delivery.message_id, delivery.message,
                )
                released.append((delivery.release_at, delivery.message_id))
                continue
            remaining.append(delivery)

        self._scheduled = remaining
        self._clock = new_time
        released.sort(key=lambda event: (event[0], event[1]))
        expired.sort(key=lambda event: (event[0], event[1]))
        return AdvanceResult(
            new_time,
            tuple(message_id for _, message_id in released),
            tuple(message_id for _, message_id in expired),
        )

    # -- execution --------------------------------------------------------
    def run(self, limit: int | None = None) -> int:
        """Process pending messages until mailboxes empty or ``limit`` hit.

        Derived messages become eligible only after the handling message
        completes. Actors paused by a supervised failure are skipped while
        the other actors keep draining; their new and due mail still enters
        their mailboxes.

        For an unsupervised actor (a tree root) the first handler failure
        aborts the run: an ordinary exception is raised as
        :class:`ActorExecutionError`, while a :class:`BaseException` that is
        not an :class:`Exception` (such as :class:`KeyboardInterrupt`)
        propagates unchanged, and a deep-copy failure of the state, message,
        handler result or buffered derived message arrives as an
        :class:`ActorExecutionError` whose ``original`` is an
        :class:`ActorDataCopyError`.

        For a supervised actor the same rollback happens -- the failing
        message keeps its id and priority in the mailbox and no state,
        derived delivery, id or trace entry is committed -- but instead of
        raising, the actor is paused, a :class:`FailureRecord` is produced
        and the run ends normally, returning the number of messages already
        completed by this call. Non-:class:`Exception` ``BaseException``
        subclasses still propagate unchanged regardless of supervision.
        """
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")

        processed = 0
        while limit is None or processed < limit:
            actor = None
            for candidate in self._actors.values():  # registration order
                # A paused actor waits for a supervision decision; its mail
                # stays queued and run moves on to the other actors.
                if candidate.mailbox and not candidate.paused:
                    actor = candidate
                    break
            if actor is None:
                break

            neg_priority, message_id, message = heapq.heappop(actor.mailbox)
            priority = -neg_priority
            try:
                # Every copy the commit depends on is made up front, so a
                # copy failure (custom __deepcopy__, an exception raised
                # while copying) commits nothing: the message goes back to
                # the mailbox with its original priority and no state,
                # derived delivery, id or trace entry is produced.
                state_before = _copy_data(
                    "state before processing", actor.state
                )

                # Hand the handler private copies so a failure (including an
                # in-place mutation before raising) can never commit state
                # or alter the message kept in the mailbox.
                context = ActorContext(self)
                try:
                    new_state = actor.handler(
                        _copy_data("actor state", actor.state),
                        _copy_data("message", message),
                        context,
                    )
                except BaseException as exc:
                    # Ordinary exceptions of an unsupervised actor are
                    # wrapped; supervised actors carry the raw exception to
                    # the shared rollback boundary below, which records it.
                    # BaseException subclasses that are not Exception
                    # (KeyboardInterrupt, SystemExit, ...) propagate
                    # unchanged either way.
                    if isinstance(exc, Exception) \
                            and actor.supervisor_name is None:
                        raise ActorExecutionError(
                            actor.name, message_id, exc
                        ) from exc
                    raise

                # Stage the commit: the returned state and every buffered
                # derived message are copied before anything becomes
                # visible, so even a failure in the last derived copy hides
                # the earlier ones and consumes no id.
                committed_state = _copy_data("handler result", new_state)
                staged = [
                    (target, _copy_data("derived message", derived))
                    for target, derived in context._pending
                ]
                trace_state = _copy_data("handler result", committed_state)
            except BaseException as exc:
                # The single rollback boundary for the whole attempt: the
                # popped mail goes back through the same submission path
                # with its original id and priority, and nothing is
                # committed -- no state, derived delivery, consumed id or
                # trace entry.
                self._push_mail(actor, priority, message_id, message)
                if actor.supervisor_name is not None \
                        and isinstance(exc, Exception):
                    # Supervised failure: pause the actor, keep the message
                    # for a later retry/drop decision and end this run
                    # normally with the count so far. ActorDataCopyError at
                    # the staging boundary is recorded as-is, mirroring the
                    # unsupervised ActorExecutionError.original contract.
                    self._record_failure(actor, message_id, exc)
                    break
                if isinstance(exc, ActorDataCopyError):
                    raise ActorExecutionError(
                        actor.name, message_id, exc
                    ) from exc
                raise

            # Commit: only runtime-owned bookkeeping below, so nothing here
            # can fail because of user data. Derived sends number and enter
            # their mailboxes only after successful completion; targets were
            # resolved when buffered and registrations never disappear, and
            # they submit at the derived default priority 0 rather than
            # inheriting the triggering message's priority.
            for derived_target, stored in staged:
                self._admit(self._actors[derived_target], 0, stored)

            actor.state = committed_state
            self._trace.append(
                TraceEntry(
                    message_id=message_id,
                    actor_name=actor.name,
                    priority=priority,
                    state_before=state_before,
                    state_after=trace_state,
                )
            )
            processed += 1

        return processed

    # -- supervision ------------------------------------------------------

    def _record_failure(self, actor: _Actor, message_id: int,
                        exc: BaseException) -> FailureRecord:
        """Pause a supervised actor and append its immutable failure record.

        Called only from the rollback boundary of :meth:`run`, after the
        failing message was put back in the mailbox. Producing the record
        consumes no message id and writes no trace entry: supervision is
        not scheduling.
        """
        actor.paused = True
        record = FailureRecord(
            actor_name=actor.name,
            message_id=message_id,
            supervisor=actor.supervisor_name,
            error_type=type(exc).__name__,
            error_text=str(exc),
            supervision_path=self._supervision_path(actor),
        )
        self._failures.append(record)
        return record

    def failures(self) -> list[FailureRecord]:
        """Return pending failure records in production order.

        A fresh list of independent records is returned each time; the
        tuples are immutable, so neither the list nor its entries can affect
        runtime state.
        """
        return list(self._failures)

    def resolve_failure(self, actor_name: str, supervisor: str,
                        action: str) -> FailureRecord | None:
        """Resolve the pending failure of ``actor_name`` as ``supervisor``.

        ``action`` is one of:

        ``"retry"``
            Delete the failure record and resume the actor; the failing
            message stays in the mailbox with its original id and priority
            and is retried by the next :meth:`run`.
        ``"drop"``
            Delete the failing message from the mailbox, delete the record
            and resume the actor. The ``send_once`` reservation for the
            message is deliberately kept, so the same delivery key still
            confirms the dropped id.
        ``"escalate"``
            Hand the record to the actor's parent supervisor without
            copying the message or consuming a message id: the record keeps
            its position in the failure order and gains the parent as its
            current supervisor; the actor stays paused.

        Returns the resulting record after the action -- the escalated
        record (same identity fields and path, current supervisor moved up)
        for ``"escalate"`` -- or ``None`` when the failure was resolved away
        (``"retry"``/``"drop"``). Raises :class:`LookupError` when no
        pending record exists for ``actor_name`` or ``supervisor`` is not
        the record's current supervisor, :class:`ValueError` for an unknown
        ``action`` and :class:`SupervisionError` when a tree-root record is
        escalated. A failed call changes no state: records, pauses,
        mailboxes, order and dedup keys are all untouched.
        """
        if action not in ("retry", "drop", "escalate"):
            raise ValueError(
                "action must be one of 'retry', 'drop', 'escalate'"
            )
        index = self._find_failure_index(actor_name)
        if index is None:
            raise LookupError(
                f"no pending failure for actor: {actor_name!r}"
            )
        record = self._failures[index]
        if record.supervisor != supervisor:
            raise LookupError(
                f"actor {actor_name!r} is not supervised by "
                f"{supervisor!r} for this failure"
            )
        if action == "escalate":
            # The next step up the captured path: the current supervisor is
            # always on it, and a root record has no successor.
            path = record.supervision_path
            current_index = path.index(record.supervisor)
            if current_index == len(path) - 1:
                raise SupervisionError(
                    f"cannot escalate failure of {actor_name!r} beyond root "
                    f"supervisor {record.supervisor!r}"
                )
            parent_name = path[current_index + 1]
            escalated = record._replace(supervisor=parent_name)
            self._failures[index] = escalated
            return escalated

        actor = self._actors[actor_name]
        if action == "drop":
            self._remove_mailbox_message(actor, record.message_id)
        # retry leaves the message for the next run; both actions delete the
        # record and release the pause. The send_once reservation is kept in
        # either case.
        del self._failures[index]
        actor.paused = False
        return None

    def _find_failure_index(self, actor_name: str) -> int | None:
        """Index of the pending failure record for ``actor_name``, if any."""
        for index, record in enumerate(self._failures):
            if record.actor_name == actor_name:
                return index
        return None

    def _remove_mailbox_message(self, actor: _Actor,
                                message_id: int) -> None:
        """Drop one message by id from an actor's paused mailbox.

        The failed actor is paused, so its mailbox cannot change between
        the failed run and the supervision decision; the id is guaranteed
        to be present. The heap is rebuilt rather than surgically removed,
        keeping the canonical (-priority, id) order.
        """
        remaining = [
            entry for entry in actor.mailbox if entry[1] != message_id
        ]
        heapq.heapify(remaining)
        actor.mailbox = remaining

    # -- read-only queries ------------------------------------------------

    def get_state(self, actor_name: str) -> Any:
        """Return an independent snapshot of the actor's current state."""
        actor = self._resolve_actor(actor_name)
        return copy.deepcopy(actor.state)

    def pending_count(self, actor_name: str) -> int:
        """Return the number of unprocessed messages for the actor.

        Only messages that have already entered the mailbox are counted;
        scheduled deliveries that have not been released (and not expired) are
        reported separately by :meth:`scheduled_count`.
        """
        actor = self._resolve_actor(actor_name)
        return len(actor.mailbox)

    def scheduled_count(self, actor_name: str) -> int:
        """Return the actor's deliveries neither released nor expired.

        Raises :class:`LookupError` for an unknown actor.
        """
        actor = self._resolve_actor(actor_name)
        return sum(
            1 for delivery in self._scheduled if delivery.actor is actor
        )

    def trace(self) -> list[TraceEntry]:
        """Return completed processings in completion order.

        A fresh list of independent snapshots is returned each time; mutating
        it or the states inside entries cannot affect the runtime.
        """
        return copy.deepcopy(self._trace)

    # -- snapshots --------------------------------------------------------

    def export_snapshot(self) -> bytes:
        """Export the complete runtime state as deterministic bytes.

        The snapshot covers everything needed to resume later: actor
        registration order and current states, the supervision tree (each
        actor's direct supervisor), paused actors, every unacknowledged
        mailbox message with its priority and message id, the timed
        deliveries neither released nor expired, the ``send_once`` dedup
        records, the pending supervision failure records in their
        production order, the logical clock, the next message id and the
        completion trace. Handlers are code and are never serialised.

        Only snapshot-safe data is supported: ``None``, booleans, integers,
        finite floats, strings, bytes and lists, tuples and string-keyed
        mappings composed recursively from those. Mappings with the same
        content in different key order produce identical bytes, and two
        runtimes with identical observable state export byte-identical
        snapshots. Unsupported values -- sets, custom objects, non-string
        mapping keys, NaN or infinite floats, circular references -- raise
        :class:`SnapshotError`; a failed export changes nothing: states,
        queues, clock, dedup records, trace and numbering all stay as they
        were.
        """
        return _encode_snapshot(self)

    @classmethod
    def restore_snapshot(
        cls, data: bytes, handlers: Mapping[str, Handler]
    ) -> "ActorRuntime":
        """Create a new runtime from snapshot ``data`` and ``handlers``.

        ``data`` must be bytes previously produced by
        :meth:`export_snapshot`. ``handlers`` maps actor names to handler
        callables; a name missing for any actor in the snapshot raises
        :class:`LookupError`, extra entries are ignored. Bytes that are
        truncated, tampered with, of an unsupported version, missing fields
        or internally inconsistent raise :class:`SnapshotError`; a failed
        restore returns no runtime at all. Restoring never imports modules
        or executes objects carried by the snapshot -- the bytes are pure
        data.

        On success every read-only query immediately reflects the snapshot
        moment, old dedup keys still confirm their first message id through
        :meth:`send_once`, new deliveries continue numbering from the saved
        next id, timed deliveries keep their original deadlines and
        :meth:`run` keeps its registration-order, priority, rollback and
        derived-message commit rules. The supervision tree, paused actors
        and pending failure records are restored exactly, so the same
        supervision decisions and inputs afterwards reproduce the same
        state, trace and numbering; a message removed by a supervision
        ``drop`` stays gone while its ``send_once`` reservation survives.
        """
        return _decode_snapshot(cls, data, handlers)


# -- snapshot encoding ----------------------------------------------------
#
# A snapshot is a canonical JSON document: sorted keys, compact separators,
# ASCII-only, no NaN/Infinity. The payload (format, version and all runtime
# state) is wrapped in an envelope carrying the SHA-256 of the payload's
# canonical bytes, so any truncation or tampering is detected on restore
# before a single field is trusted. Every business value is encoded as a
# tagged node -- a JSON array whose first element is a reserved tag -- so
# user data can never be mistaken for structure and the encoding needs no
# escaping:
#   ["n"]            None
#   ["b", bool]      boolean
#   ["i", int]       integer
#   ["f", float]     finite float
#   ["s", str]       string
#   ["y", hex]       bytes (lowercase hex)
#   ["l", [items]]   list
#   ["t", [items]]   tuple
#   ["d", [[k, v]]]  mapping, pairs sorted by string key

_SNAPSHOT_FORMAT = "actor-runtime-snapshot"
_SNAPSHOT_VERSION = 1

_HEXDIGITS = frozenset("0123456789abcdef")


def _encode_value(value: Any, active: set[int]) -> list:
    """Encode a supported value as a tagged JSON tree.

    ``active`` holds the id()s of the containers on the current recursion
    path, so a circular reference is reported instead of recursing forever.
    Raises :class:`SnapshotError` for anything outside the snapshot domain.
    """
    if value is None:
        return ["n"]
    value_type = type(value)
    if value_type is bool:
        return ["b", value]
    if value_type is int:
        return ["i", value]
    if value_type is float:
        if not math.isfinite(value):
            raise SnapshotError(
                f"non-finite float cannot be snapshotted: {value!r}"
            )
        return ["f", value]
    if value_type is str:
        return ["s", value]
    if value_type is bytes:
        return ["y", value.hex()]
    if value_type is list or value_type is tuple:
        marker = id(value)
        if marker in active:
            raise SnapshotError("circular reference cannot be snapshotted")
        active.add(marker)
        try:
            items = [_encode_value(item, active) for item in value]
        finally:
            active.discard(marker)
        return ["l" if value_type is list else "t", items]
    if value_type is dict:
        marker = id(value)
        if marker in active:
            raise SnapshotError("circular reference cannot be snapshotted")
        active.add(marker)
        try:
            pairs = []
            for key, item in value.items():
                if type(key) is not str:
                    raise SnapshotError(
                        "mapping key must be a string, got "
                        f"{type(key).__name__}"
                    )
                pairs.append((key, _encode_value(item, active)))
        finally:
            active.discard(marker)
        # Canonical form: content, not insertion order, determines the bytes.
        pairs.sort(key=lambda pair: pair[0])
        return ["d", [[key, item] for key, item in pairs]]
    raise SnapshotError(
        f"value of type {type(value).__name__} cannot be snapshotted"
    )


def _canonical_json(value: Any) -> bytes:
    """Serialise to the one canonical byte form of a JSON structure."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _encode_snapshot(runtime: ActorRuntime) -> bytes:
    """Serialise the whole runtime state to canonical JSON bytes."""
    active: set[int] = set()
    actors = []
    for actor in runtime._actors.values():  # registration order
        # The heap array is serialised in sorted (-priority, id) order;
        # message ids are unique, so the message never participates in the
        # sort and the restore-side heap is rebuilt deterministically.
        mailbox = [
            [message_id, -neg_priority, _encode_value(message, active)]
            for neg_priority, message_id, message in sorted(actor.mailbox)
        ]
        dedup = [[key, actor.dedup_keys[key]] for key in sorted(actor.dedup_keys)]
        actor_node = {
            "name": actor.name,
            "state": _encode_value(actor.state, active),
            "mailbox": mailbox,
            "dedup": dedup,
        }
        # Supervision fields are only emitted when they carry non-default
        # data, so a runtime without supervised actors exports exactly the
        # bytes the baseline format did; restore treats them as optional.
        if actor.supervisor_name is not None:
            actor_node["supervisor"] = actor.supervisor_name
        if actor.paused:
            actor_node["paused"] = True
        actors.append(actor_node)
    scheduled = [
        {
            "id": delivery.message_id,
            "actor": delivery.actor.name,
            "priority": delivery.priority,
            "message": _encode_value(delivery.message, active),
            "scheduled_at": delivery.scheduled_at,
            "release_at": delivery.release_at,
            "expire_at": delivery.expire_at,
        }
        for delivery in runtime._scheduled
    ]
    trace = [
        {
            "message_id": entry.message_id,
            "actor": entry.actor_name,
            "priority": entry.priority,
            "state_before": _encode_value(entry.state_before, active),
            "state_after": _encode_value(entry.state_after, active),
        }
        for entry in runtime._trace
    ]
    payload = {
        "format": _SNAPSHOT_FORMAT,
        "version": _SNAPSHOT_VERSION,
        "clock": runtime._clock,
        "next_message_id": runtime._next_message_id,
        "actors": actors,
        "scheduled": scheduled,
        "trace": trace,
    }
    # Pending failures are only emitted when supervision is actually in use,
    # keeping the bytes of an unsupervised runtime baseline-identical. The
    # list order is the production order resolve_failure preserves.
    if runtime._failures:
        payload["failures"] = [
            {
                "actor": record.actor_name,
                "message_id": record.message_id,
                "supervisor": record.supervisor,
                "error_type": record.error_type,
                "error_text": record.error_text,
                "path": list(record.supervision_path),
            }
            for record in runtime._failures
        ]
    # The digest covers the canonical payload bytes, so a restore can
    # recompute it over the re-serialised parsed payload: the canonical
    # form is a fixed point of parse/serialise.
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return _canonical_json({"payload": payload, "digest": digest})


def _require(condition: bool, message: str) -> None:
    """The single validation boundary of snapshot restore."""
    if not condition:
        raise SnapshotError(message)


def _decode_value(node: Any) -> Any:
    """Decode a tagged node back into a value, validating every shape."""
    if not isinstance(node, list) or not node or type(node[0]) is not str:
        raise SnapshotError("malformed encoded value")
    tag = node[0]
    if tag == "n":
        _require(len(node) == 1, "malformed null node")
        return None
    if tag == "b":
        _require(len(node) == 2 and type(node[1]) is bool, "malformed bool node")
        return node[1]
    if tag == "i":
        # type check, not isinstance: a boolean is not an integer here.
        _require(len(node) == 2 and type(node[1]) is int, "malformed int node")
        return node[1]
    if tag == "f":
        _require(
            len(node) == 2 and type(node[1]) in (int, float),
            "malformed float node",
        )
        value = float(node[1])
        _require(math.isfinite(value), "non-finite float in snapshot")
        return value
    if tag == "s":
        _require(len(node) == 2 and type(node[1]) is str, "malformed string node")
        return node[1]
    if tag == "y":
        _require(len(node) == 2 and type(node[1]) is str, "malformed bytes node")
        text = node[1]
        _require(
            len(text) % 2 == 0 and all(c in _HEXDIGITS for c in text),
            "malformed bytes hex",
        )
        return bytes.fromhex(text)
    if tag == "l" or tag == "t":
        _require(len(node) == 2 and isinstance(node[1], list), "malformed sequence node")
        items = [_decode_value(item) for item in node[1]]
        return items if tag == "l" else tuple(items)
    if tag == "d":
        _require(len(node) == 2 and isinstance(node[1], list), "malformed mapping node")
        result = {}
        for pair in node[1]:
            _require(
                isinstance(pair, list) and len(pair) == 2
                and type(pair[0]) is str,
                "malformed mapping entry",
            )
            _require(pair[0] not in result, f"duplicate mapping key: {pair[0]!r}")
            result[pair[0]] = _decode_value(pair[1])
        return result
    raise SnapshotError(f"unknown value tag: {tag!r}")


def _decode_snapshot(
    cls: type, data: bytes, handlers: Mapping[str, Handler]
) -> ActorRuntime:
    """Validate snapshot bytes and build a fresh runtime from them.

    Everything is validated before anything is assembled, so a failure
    raises without producing a partially usable runtime.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("snapshot data must be bytes")
    if not isinstance(handlers, Mapping):
        raise TypeError("handlers must be a mapping of actor name to handler")
    try:
        envelope = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotError(f"invalid snapshot bytes: {exc}") from exc

    # Integrity first: until the digest matches, no field is trusted.
    _require(isinstance(envelope, dict), "snapshot root must be an object")
    payload = envelope.get("payload")
    digest = envelope.get("digest")
    _require(isinstance(payload, dict), "snapshot payload must be an object")
    _require(type(digest) is str, "snapshot digest must be a string")
    _require(
        hashlib.sha256(_canonical_json(payload)).hexdigest() == digest,
        "snapshot integrity check failed",
    )

    _require(
        payload.get("format") == _SNAPSHOT_FORMAT,
        "not an actor-runtime snapshot",
    )
    version = payload.get("version")
    _require(
        type(version) is int and version == _SNAPSHOT_VERSION,
        f"unsupported snapshot version: {version!r}",
    )
    try:
        clock = payload["clock"]
        next_message_id = payload["next_message_id"]
        actors_node = payload["actors"]
        scheduled_node = payload["scheduled"]
        trace_node = payload["trace"]
    except KeyError as exc:
        raise SnapshotError(f"missing snapshot field: {exc.args[0]!r}") from exc
    _require(type(clock) is int and clock >= 0,
             "clock must be a non-negative integer")
    _require(type(next_message_id) is int and next_message_id >= 1,
             "next_message_id must be a positive integer")
    _require(isinstance(actors_node, list), "actors must be a list")
    _require(isinstance(scheduled_node, list), "scheduled must be a list")
    _require(isinstance(trace_node, list), "trace must be a list")

    def take_id(message_id: Any) -> int:
        _require(type(message_id) is int and 1 <= message_id < next_message_id,
                 "message id out of range")
        _require(message_id not in live_ids, "duplicate message id")
        live_ids.add(message_id)
        return message_id

    live_ids: set[int] = set()     # mailbox + scheduled + trace ids
    reachable_ids: set[int] = set()  # mailbox + trace ids (dedup targets)

    decoded_actors = []
    names: set[str] = set()
    for actor_node in actors_node:
        _require(isinstance(actor_node, dict), "actor entry must be an object")
        try:
            name = actor_node["name"]
            state_node = actor_node["state"]
            mailbox_node = actor_node["mailbox"]
            dedup_node = actor_node["dedup"]
        except KeyError as exc:
            raise SnapshotError(f"missing actor field: {exc.args[0]!r}") from exc
        _require(type(name) is str and name != "",
                 "actor name must be a non-empty string")
        _require(name not in names, f"duplicate actor: {name!r}")
        names.add(name)
        state = _decode_value(state_node)
        _require(isinstance(mailbox_node, list), "mailbox must be a list")
        mailbox = []
        for entry in mailbox_node:
            _require(isinstance(entry, list) and len(entry) == 3,
                     "mailbox entry must be [id, priority, message]")
            message_id, priority, message_node = entry
            take_id(message_id)
            reachable_ids.add(message_id)
            _require(type(priority) is int, "priority must be an integer")
            mailbox.append((message_id, priority, _decode_value(message_node)))
        # Canonical heap order, independent of the file's entry order.
        mailbox.sort(key=lambda item: (-item[1], item[0]))
        _require(isinstance(dedup_node, list), "dedup must be a list")
        dedup = {}
        dedup_first_ids: set[int] = set()
        for pair in dedup_node:
            _require(isinstance(pair, list) and len(pair) == 2,
                     "dedup entry must be [key, message id]")
            key, first_id = pair
            _require(type(key) is str and key != "",
                     "dedup key must be a non-empty string")
            _require(key not in dedup, f"duplicate dedup key: {key!r}")
            _require(type(first_id) is int and 1 <= first_id < next_message_id,
                     "dedup message id out of range")
            _require(first_id not in dedup_first_ids,
                     "dedup message id shared by two keys")
            dedup_first_ids.add(first_id)
            dedup[key] = first_id
        # Optional supervision fields: absent means a baseline root actor,
        # never paused.
        supervisor_name = actor_node.get("supervisor")
        _require(
            supervisor_name is None or type(supervisor_name) is str,
            "supervisor must be a string",
        )
        paused_flag = actor_node.get("paused", False)
        _require(type(paused_flag) is bool, "paused must be a boolean")
        decoded_actors.append(
            (name, state, mailbox, dedup, supervisor_name, paused_flag)
        )

    # The supervision graph only makes sense once every name is known.
    supervisor_by_name = {
        name: supervisor_name
        for name, _state, _mailbox, _dedup, supervisor_name, _paused
        in decoded_actors
    }
    actual_paths: dict[str, tuple[str, ...]] = {}
    for name, _state, _mailbox, _dedup, supervisor_name, _paused \
            in decoded_actors:
        if supervisor_name is not None:
            _require(
                supervisor_name != name and supervisor_name in names,
                f"actor {name!r} references unknown supervisor "
                f"{supervisor_name!r}",
            )
            # Re-walk the links with a visited set so a tampered cycle can
            # never loop forever; every chain must end at a root.
            path = [name]
            current = supervisor_name
            seen = {name}
            while True:
                _require(current not in seen,
                         f"supervision cycle at {current!r}")
                seen.add(current)
                path.append(current)
                current_sup = supervisor_by_name[current]
                if current_sup is None:
                    break
                current = current_sup
            actual_paths[name] = tuple(path)

    decoded_scheduled = []
    for node in scheduled_node:
        _require(isinstance(node, dict), "scheduled entry must be an object")
        try:
            message_id = node["id"]
            actor_name = node["actor"]
            priority = node["priority"]
            message_node = node["message"]
            scheduled_at = node["scheduled_at"]
            release_at = node["release_at"]
            expire_at = node["expire_at"]
        except KeyError as exc:
            raise SnapshotError(
                f"missing scheduled field: {exc.args[0]!r}"
            ) from exc
        take_id(message_id)
        _require(actor_name in names,
                 f"scheduled delivery targets unknown actor: {actor_name!r}")
        _require(type(priority) is int, "priority must be an integer")
        _require(type(scheduled_at) is int and 0 <= scheduled_at <= clock,
                 "scheduled_at must be an integer between 0 and the clock")
        _require(type(release_at) is int and release_at >= scheduled_at,
                 "release_at must be an integer no earlier than scheduled_at")
        _require(release_at > clock, "scheduled delivery is already due")
        _require(
            expire_at is None
            or (type(expire_at) is int and expire_at > scheduled_at),
            "expire_at must be None or an integer later than scheduled_at",
        )
        if expire_at is not None:
            _require(expire_at > clock, "scheduled delivery is already expired")
        decoded_scheduled.append((
            message_id, actor_name, priority, _decode_value(message_node),
            scheduled_at, release_at, expire_at,
        ))

    decoded_trace = []
    for node in trace_node:
        _require(isinstance(node, dict), "trace entry must be an object")
        try:
            message_id = node["message_id"]
            actor_name = node["actor"]
            priority = node["priority"]
            before_node = node["state_before"]
            after_node = node["state_after"]
        except KeyError as exc:
            raise SnapshotError(
                f"missing trace field: {exc.args[0]!r}"
            ) from exc
        take_id(message_id)
        reachable_ids.add(message_id)
        _require(actor_name in names,
                 f"trace entry names unknown actor: {actor_name!r}")
        _require(type(priority) is int, "priority must be an integer")
        decoded_trace.append(TraceEntry(
            message_id=message_id,
            actor_name=actor_name,
            priority=priority,
            state_before=_decode_value(before_node),
            state_after=_decode_value(after_node),
        ))

    # Cross-structure relation: a dedup record always points at the first
    # delivery accepted for its key. That id is still waiting in a mailbox,
    # already completed and present in the trace, or was removed by a
    # supervision ``drop`` while the key stayed reserved -- in which case it
    # is simply an issued id (below next_message_id) absent from every live
    # structure. The only rejected case is an id that lives solely in the
    # scheduled set, which can never have been accepted into a mailbox.
    scheduled_only_ids = set(live_ids) - reachable_ids
    for name, _state, _mailbox, dedup, _sup, _paused in decoded_actors:
        for key, first_id in dedup.items():
            _require(first_id not in scheduled_only_ids,
                     f"dedup key {key!r} of actor {name!r} references an "
                     "unknown message")

    # Pending supervised failures. Absent on baseline snapshots; when
    # present, every record must describe a supervised, paused actor whose
    # failing message is still waiting in that actor's mailbox.
    failures_node = payload.get("failures", [])
    _require(isinstance(failures_node, list), "failures must be a list")
    mailbox_ids = {
        name: {message_id for message_id, _priority, _message in mailbox}
        for name, _state, mailbox, _dedup, _sup, _paused in decoded_actors
    }
    paused_names: set[str] = set()
    decoded_failures: list[FailureRecord] = []
    for node in failures_node:
        _require(isinstance(node, dict), "failure entry must be an object")
        try:
            actor_name = node["actor"]
            message_id = node["message_id"]
            supervisor_name = node["supervisor"]
            error_type = node["error_type"]
            error_text = node["error_text"]
            path_node = node["path"]
        except KeyError as exc:
            raise SnapshotError(
                f"missing failure field: {exc.args[0]!r}"
            ) from exc
        _require(type(actor_name) is str and actor_name in names,
                 "failure names an unknown actor")
        _require(actor_name not in paused_names,
                 f"multiple pending failures for actor: {actor_name!r}")
        paused_names.add(actor_name)
        actor_tuple = next(
            a for a in decoded_actors if a[0] == actor_name
        )
        _require(actor_tuple[4] is not None,
                 f"failure for unsupervised actor: {actor_name!r}")
        _require(type(message_id) is int
                 and message_id in mailbox_ids[actor_name],
                 "failure message must still be in the actor's mailbox")
        path = actual_paths[actor_name]
        _require(
            type(supervisor_name) is str and supervisor_name in path[1:],
            "failure supervisor must be an ancestor on the supervision path",
        )
        _require(type(error_type) is str and error_type != "",
                 "failure error_type must be a non-empty string")
        _require(type(error_text) is str,
                 "failure error_text must be a string")
        _require(isinstance(path_node, list)
                 and all(type(part) is str and part != ""
                         for part in path_node),
                 "failure path must be a list of non-empty strings")
        _require(tuple(path_node) == path,
                 "failure path does not match the supervision tree")
        decoded_failures.append(FailureRecord(
            actor_name=actor_name,
            message_id=message_id,
            supervisor=supervisor_name,
            error_type=error_type,
            error_text=error_text,
            supervision_path=path,
        ))

    # Pause markers and pending records must agree in both directions.
    for name, _state, _mailbox, _dedup, _sup, paused_flag in decoded_actors:
        _require(paused_flag == (name in paused_names),
                 f"paused flag and failure records disagree for {name!r}")

    # Handler boundary: every snapshotted actor needs a handler; extra
    # mappings are ignored. Checked before assembly so a failure produces
    # no runtime at all.
    for name, _state, _mailbox, _dedup, _sup, _paused in decoded_actors:
        if name not in handlers:
            raise LookupError(f"no handler provided for actor: {name!r}")

    runtime = cls()
    runtime._clock = clock
    runtime._next_message_id = next_message_id
    for order, (name, state, mailbox, dedup,
                supervisor_name, paused_flag) in enumerate(decoded_actors):
        actor = _Actor(name, state, handlers[name], order, supervisor_name)
        for message_id, priority, message in mailbox:
            heapq.heappush(actor.mailbox, (-priority, message_id, message))
        actor.dedup_keys = dict(dedup)
        actor.paused = paused_flag
        runtime._actors[name] = actor
    runtime._scheduled = [
        _ScheduledDelivery(
            message_id, runtime._actors[actor_name], priority, message,
            scheduled_at, release_at, expire_at,
        )
        for (message_id, actor_name, priority, message,
             scheduled_at, release_at, expire_at) in decoded_scheduled
    ]
    runtime._trace = list(decoded_trace)
    runtime._failures = list(decoded_failures)
    return runtime
