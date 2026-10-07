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
    """A supervised actor failure awaiting its supervisor's decision.

    The record is immutable and is created when a supervised actor's
    handler raises an :class:`Exception` (including an
    :class:`ActorDataCopyError` at the copy boundary) during
    :meth:`ActorRuntime.run`: the failing message keeps its original id and
    priority in the actor's mailbox, nothing is committed and the actor is
    paused until its supervisor calls
    :meth:`ActorRuntime.resolve_failure`.

    ``actor`` is the failed actor, ``message_id`` the id of the message it
    failed on, ``supervisor`` the failing actor's direct supervisor at the
    time of the failure and ``supervision_path`` the names from the root to
    that supervisor, both ends included (a one-element tuple for a
    supervisor that is itself a root). ``error_type`` is the type name of
    the original exception and ``error_text`` its ``str()`` text.
    """

    actor: str
    message_id: int
    supervisor: str
    error_type: str
    error_text: str
    supervision_path: tuple[str, ...]


class ActorDataCopyError(Exception):
    """Raised when a message or state cannot be deep-copied.

    :meth:`ActorRuntime.send` raises it directly when the message cannot be
    copied; :meth:`ActorRuntime.run` reports a copy failure during processing
    as an :class:`ActorExecutionError` whose ``original`` is an
    ``ActorDataCopyError``. Either way nothing is committed: no mailbox
    change, no state change, no derived delivery, no trace entry and no
    consumed message id. ``what`` names the data being copied and
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
    """Raised by :meth:`ActorRuntime.run` when a message handler fails.

    Raised only for actors without a supervisor. The failing message stays
    unacknowledged in its actor's mailbox and no state or derived delivery
    is committed for it. Messages completed earlier in the same
    :meth:`~ActorRuntime.run` call remain committed. Supervised actors
    never raise this: their failures pause the actor and produce a
    :class:`FailureRecord` instead.
    """

    def __init__(self, actor_name: str, message_id: int, original: BaseException):
        self.actor_name = actor_name
        self.message_id = message_id
        self.original = original
        super().__init__(
            f"actor {actor_name!r} failed while handling message {message_id}: "
            f"{type(original).__name__}: {original}"
        )


class SupervisionError(Exception):
    """Raised when a failure reaches the top of the supervision tree.

    A root actor has no supervisor above it, so escalating its pending
    :class:`FailureRecord` cannot be handled any further. The record is
    retained and the actor stays paused, exactly as it was before the
    :meth:`ActorRuntime.resolve_failure` call; nothing else changes.
    """


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
        "supervisor", "paused",
    )

    def __init__(self, name: str, state: Any, handler: Handler, order: int,
                 supervisor: str | None = None):
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
        # Name of the direct supervising actor, or None for a root. Fixed
        # for the actor's whole lifetime; supervision never participates in
        # scheduling. A paused actor still receives mail (its pending_count
        # keeps growing) but run skips it while a FailureRecord is pending.
        self.supervisor: str | None = supervisor
        self.paused: bool = False


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
        # Pending supervised failures, in production order. An actor has at
        # most one record: it pauses on failure and is skipped until the
        # record is resolved; an escalation rewrites the record in place.
        self._failures: list[FailureRecord] = []
        # Ids of messages removed by a supervision "drop". They are kept
        # only so a snapshot can reject a forged id that pretends a live
        # message was dropped; the ids themselves are consumed (they keep
        # their numbering) and never re-enter a mailbox.
        self._dropped_message_ids: set[int] = set()

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

    def _supervision_path_to(self, supervisor: _Actor) -> tuple[str, ...]:
        """Names from the root to ``supervisor``, both ends included."""
        path = [supervisor.name]
        current = supervisor
        while current.supervisor is not None:
            current = self._actors[current.supervisor]
            path.append(current.name)
        return tuple(reversed(path))

    def register(self, name: str, initial_state: Any, handler: Handler,
                 supervisor: str | None = None) -> None:
        """Register a named actor, optionally beneath a direct supervisor.

        A registered actor with ``supervisor=None`` (the default) is a root
        and keeps every unsupervised behaviour: a handler failure in
        :meth:`run` raises :class:`ActorExecutionError` exactly as before.
        Passing the name of an already registered actor makes that actor the
        new actor's *direct* supervisor; failures of the new actor then
        pause it and await the supervisor's decision through
        :meth:`resolve_failure`, and may be escalated up the chain. The
        supervision relation never participates in scheduling.

        Raises :class:`ValueError` when ``name`` is empty or already
        registered, when ``supervisor`` is the empty string or when an actor
        is registered as its own supervisor; raises :class:`TypeError` when
        ``supervisor`` is neither ``None`` nor a string and
        :class:`LookupError` when it names an actor that is not registered.
        A failed registration changes nothing -- no order slot, no
        supervision edge and no state are added.
        """
        if not isinstance(name, str) or name == "":
            raise ValueError("actor name must be a non-empty string")
        if name in self._actors:
            raise ValueError(f"actor already registered: {name!r}")
        if supervisor is not None:
            if not isinstance(supervisor, str):
                raise TypeError("supervisor must be a string or None")
            if supervisor == "":
                raise ValueError("supervisor must be a non-empty string")
            if supervisor == name:
                raise ValueError(f"actor cannot supervise itself: {name!r}")
            if supervisor not in self._actors:
                raise LookupError(f"unknown supervisor: {supervisor!r}")
        self._actors[name] = _Actor(
            name, copy.deepcopy(initial_state), handler,
            len(self._actors), supervisor,
        )

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
        completes. Paused (supervised, failed) actors are skipped; their
        mail waits while other actors keep being processed.

        When an actor *without* a supervisor fails, the first failure ends
        the run by raising: an ordinary exception is raised as
        :class:`ActorExecutionError`, while a :class:`BaseException` that is
        not an :class:`Exception` (such as :class:`KeyboardInterrupt`)
        propagates unchanged. A failure to deep-copy the state, the message,
        the handler result or a buffered derived message is raised as
        :class:`ActorExecutionError` whose ``original`` is an
        :class:`ActorDataCopyError`.

        A *supervised* actor never raises on an ordinary failure: the same
        rollback is performed (the failing message keeps its original id
        and priority in the mailbox; no state, derived delivery, id or trace
        entry is committed), the actor is paused and an immutable
        :class:`FailureRecord` is queued for its current supervisor; the
        run then ends, returning the number of messages already completed.
        The decision -- retry, drop or escalate -- belongs to
        :meth:`resolve_failure`. A non-:class:`Exception`
        :class:`BaseException` still propagates unchanged even for a
        supervised actor, with no record and no pause.
        """
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")

        processed = 0
        while limit is None or processed < limit:
            actor = None
            for candidate in self._actors.values():  # registration order
                # A paused (supervised, failed) actor keeps its mail and
                # still receives new deliveries, but is never scheduled
                # until its pending FailureRecord is resolved.
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
                    # Ordinary exceptions are wrapped; BaseException
                    # subclasses that are not Exception (KeyboardInterrupt,
                    # SystemExit, ...) propagate unchanged.
                    if isinstance(exc, Exception):
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
                if isinstance(exc, ActorDataCopyError):
                    if actor.supervisor is None:
                        raise ActorExecutionError(
                            actor.name, message_id, exc
                        ) from exc
                    original = exc
                elif isinstance(exc, ActorExecutionError):
                    if actor.supervisor is None:
                        raise
                    original = exc.original
                else:
                    # Non-Exception BaseExceptions propagate as-is, for
                    # supervised and root actors alike: no record, no pause.
                    raise
                # A supervised ordinary failure pauses the actor and waits
                # for its supervisor instead of aborting the caller. The
                # run ends here; already completed messages stay committed.
                self._failures.append(self._make_failure_record(
                    actor, message_id, original
                ))
                actor.paused = True
                break

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

    def _make_failure_record(self, actor: _Actor, message_id: int,
                             original: BaseException) -> FailureRecord:
        """Build the immutable record of a supervised actor failure."""
        supervisor = self._actors[actor.supervisor]  # type: ignore[arg-type]
        return FailureRecord(
            actor=actor.name,
            message_id=message_id,
            supervisor=supervisor.name,
            error_type=type(original).__name__,
            error_text=str(original),
            supervision_path=self._supervision_path_to(supervisor),
        )

    # -- supervision ------------------------------------------------------

    def failures(self) -> list[FailureRecord]:
        """Return pending supervised failures, in production order.

        A fresh list is returned each time; the records are immutable and
        hold only copies of the failure facts, so mutating the result
        cannot affect the runtime.
        """
        return list(self._failures)

    def resolve_failure(self, supervisor: str, actor: str, message_id: int,
                        action: str) -> FailureRecord:
        """Resolve a pending failure as its current supervisor.

        ``supervisor`` must be the failure's current supervisor and
        ``(actor, message_id)`` must identify a pending
        :class:`FailureRecord`. ``action`` is one of:

        * ``"retry"`` -- delete the record and resume the actor; the failing
          message keeps its original id and priority in the mailbox and is
          retried by the next :meth:`run`;
        * ``"drop"`` -- delete the failing message and resume the actor.
          The message is removed with no trace entry and no consumed id,
          while ``send_once`` keys are retained;
        * ``"escalate"`` -- hand the record to the failing actor's
          grandparent (the current supervisor's own supervisor). The
          record is rewritten in place, keeping its production-order
          position; nothing is copied or renumbered and the failed actor
          stays paused. Escalating a root's failure raises
          :class:`SupervisionError` and keeps the record and pause.

        Raises :class:`LookupError` when no pending record matches
        ``(actor, message_id)`` or the caller is not its current
        supervisor, and :class:`ValueError` for an unknown action. A failed
        resolution changes nothing. Returns the removed record for
        ``retry``/``drop`` or the rewritten record for ``escalate``.
        """
        index = None
        for i, record in enumerate(self._failures):
            if record.actor == actor and record.message_id == message_id:
                index = i
                break
        if index is None:
            raise LookupError(
                f"no pending failure for actor {actor!r} "
                f"message {message_id!r}"
            )
        record = self._failures[index]
        if record.supervisor != supervisor:
            raise LookupError(
                f"failure of actor {actor!r} belongs to supervisor "
                f"{record.supervisor!r}, not {supervisor!r}"
            )
        if action not in ("retry", "drop", "escalate"):
            raise ValueError(f"unknown supervision action: {action!r}")

        failed_actor = self._actors[record.actor]
        if action == "retry":
            # The message is already back in the mailbox with its original
            # id and priority; just forget the failure and resume.
            del self._failures[index]
            failed_actor.paused = False
            return record

        if action == "drop":
            # Remove exactly the failing mail entry (the rollback boundary
            # guarantees it is present) and rebuild the heap; no
            # renumbering, no trace, and the send_once key stays reserved.
            mailbox = failed_actor.mailbox
            for pos, entry in enumerate(mailbox):
                if entry[1] == message_id:
                    mailbox[pos] = mailbox[-1]
                    mailbox.pop()
                    heapq.heapify(mailbox)
                    break
            self._dropped_message_ids.add(message_id)
            del self._failures[index]
            failed_actor.paused = False
            return record

        # escalate: the current supervisor must itself be supervised.
        current = self._actors[record.supervisor]
        if current.supervisor is None:
            raise SupervisionError(
                f"cannot escalate failure of {record.actor!r}: supervisor "
                f"{current.name!r} is a root actor"
            )
        next_supervisor = self._actors[current.supervisor]
        new_record = FailureRecord(
            actor=record.actor,
            message_id=record.message_id,
            supervisor=next_supervisor.name,
            error_type=record.error_type,
            error_text=record.error_text,
            supervision_path=self._supervision_path_to(next_supervisor),
        )
        # Rewrite in place: production order is preserved, the failed actor
        # stays paused, the message is neither copied nor renumbered.
        self._failures[index] = new_record
        return new_record

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
        actor's direct supervisor) and pause marks, every unacknowledged
        mailbox message with its priority and message id, the pending
        supervision failures in production order with their current
        supervisor and supervision path, the ids removed by a supervision
        drop, the timed deliveries neither released nor expired, the
        ``send_once`` dedup records, the logical clock, the next message id
        and the completion trace. Handlers are code and are never
        serialised.

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
        next id, timed deliveries keep their original deadlines, the
        supervision tree and pause marks are restored with their pending
        :class:`FailureRecord` records in the same order and :meth:`run`
        keeps its registration-order, priority, rollback and derived-message
        commit rules.
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
        actors.append({
            "name": actor.name,
            "state": _encode_value(actor.state, active),
            "mailbox": mailbox,
            "dedup": dedup,
            "supervisor": actor.supervisor,
            "paused": actor.paused,
        })
    # Pending failures keep their production order; the supervision path is
    # derivable from the actor tree but is stored so a restore can verify
    # the bytes against the rebuilt tree instead of trusting them.
    failures = [
        {
            "actor": record.actor,
            "message_id": record.message_id,
            "supervisor": record.supervisor,
            "error_type": record.error_type,
            "error_text": record.error_text,
            "supervision_path": list(record.supervision_path),
        }
        for record in runtime._failures
    ]
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
        "failures": failures,
        "dropped_message_ids": sorted(runtime._dropped_message_ids),
        "scheduled": scheduled,
        "trace": trace,
    }
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
    # Supervision was added after version 1 shipped: bytes written by the
    # older runtime simply omit these fields, and missing means roots, no
    # pause, no pending failure and no dropped id.
    failures_node = payload.get("failures", [])
    dropped_node = payload.get("dropped_message_ids", [])
    _require(type(clock) is int and clock >= 0,
             "clock must be a non-negative integer")
    _require(type(next_message_id) is int and next_message_id >= 1,
             "next_message_id must be a positive integer")
    _require(isinstance(actors_node, list), "actors must be a list")
    _require(isinstance(scheduled_node, list), "scheduled must be a list")
    _require(isinstance(trace_node, list), "trace must be a list")
    _require(isinstance(failures_node, list), "failures must be a list")
    _require(isinstance(dropped_node, list),
             "dropped_message_ids must be a list")

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
        # Absent on pre-supervision bytes: every actor was a root.
        supervisor = actor_node.get("supervisor")
        _require(supervisor is None or type(supervisor) is str,
                 "supervisor must be null or a string")
        if supervisor is not None:
            _require(supervisor != "", "supervisor must be a non-empty string")
            _require(supervisor != name,
                     f"actor {name!r} cannot supervise itself")
        paused = actor_node.get("paused", False)
        _require(type(paused) is bool, "paused must be a boolean")
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
        decoded_actors.append(
            (name, state, mailbox, dedup, supervisor, paused)
        )

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

    # Dropped message ids: messages a supervision "drop" removed. They are
    # neither mailbox, scheduled nor trace ids, but a retained send_once key
    # may still point at one.
    dropped_ids: set[int] = set()
    for dropped_id in dropped_node:
        _require(type(dropped_id) is int and 1 <= dropped_id < next_message_id,
                 "dropped message id out of range")
        _require(dropped_id not in dropped_ids, "duplicate dropped message id")
        _require(dropped_id not in live_ids,
                 "dropped message id is still live in a queue or trace")
        dropped_ids.add(dropped_id)

    # Supervision edges: a supervisor is always registered earlier than its
    # child (registration is the only way to add one), so checking order
    # validates the tree against cycles and dangling edges in one step.
    supervisor_of: dict[str, str | None] = {}
    order_of: dict[str, int] = {}
    paused_names: set[str] = set()
    for order, (name, _state, _mailbox, _dedup, supervisor, paused) \
            in enumerate(decoded_actors):
        if supervisor is not None:
            _require(supervisor in supervisor_of,
                     f"actor {name!r} names unknown supervisor "
                     f"{supervisor!r}")
            _require(order_of[supervisor] < order,
                     f"supervisor {supervisor!r} of {name!r} must be "
                     "registered first")
        supervisor_of[name] = supervisor
        order_of[name] = order
        if paused:
            _require(supervisor is not None,
                     f"paused actor {name!r} has no supervisor")
            paused_names.add(name)

    # Pending failures, kept in their stored production order.
    mailbox_ids: dict[str, set[int]] = {
        name: {message_id for message_id, _p, _m in mailbox}
        for name, _state, mailbox, _dedup, _sup, _paused in decoded_actors
    }
    decoded_failures: list[FailureRecord] = []
    failed_actors: set[str] = set()
    failure_keys: set[tuple[str, int]] = set()
    for node in failures_node:
        _require(isinstance(node, dict), "failure entry must be an object")
        try:
            failed_name = node["actor"]
            failure_id = node["message_id"]
            failure_supervisor = node["supervisor"]
            error_type = node["error_type"]
            error_text = node["error_text"]
            path_node = node["supervision_path"]
        except KeyError as exc:
            raise SnapshotError(
                f"missing failure field: {exc.args[0]!r}"
            ) from exc
        _require(type(failed_name) is str and failed_name in supervisor_of,
                 f"failure names unknown actor: {failed_name!r}")
        _require(type(failure_id) is int and 1 <= failure_id < next_message_id,
                 "failure message id out of range")
        key = (failed_name, failure_id)
        _require(key not in failure_keys, "duplicate failure record")
        failure_keys.add(key)
        _require(failed_name not in failed_actors,
                 f"actor {failed_name!r} has more than one pending failure")
        failed_actors.add(failed_name)
        _require(supervisor_of[failed_name] is not None,
                 f"failure recorded for root actor: {failed_name!r}")
        _require(type(failure_supervisor) is str,
                 "failure supervisor must be a string")
        # After one or more escalations the record's current holder is an
        # ancestor of the failed actor rather than its direct supervisor; it
        # just has to lie on the chain from the direct supervisor to a root.
        ancestor = supervisor_of[failed_name]
        while ancestor is not None and ancestor != failure_supervisor:
            ancestor = supervisor_of[ancestor]
        _require(ancestor == failure_supervisor,
                 f"failure supervisor {failure_supervisor!r} of "
                 f"{failed_name!r} is not on its supervision chain")
        _require(type(error_type) is str and error_type != "",
                 "failure error_type must be a non-empty string")
        _require(type(error_text) is str, "failure error_text must be a string")
        _require(isinstance(path_node, list) and path_node,
                 "supervision_path must be a non-empty list")
        path = []
        for step in path_node:
            _require(type(step) is str and step in supervisor_of,
                     f"supervision_path names unknown actor: {step!r}")
            path.append(step)
        _require(len(set(path)) == len(path),
                 "supervision_path contains a cycle")
        # The path walks root -> ... -> current supervisor along the edges.
        _require(supervisor_of[path[0]] is None,
                 "supervision_path must start at a root")
        for parent, child in zip(path, path[1:]):
            _require(supervisor_of[child] == parent,
                     "supervision_path does not follow the supervision tree")
        _require(path[-1] == failure_supervisor,
                 "supervision_path must end at the failure's supervisor")
        # The paused actor's failing message is still waiting in its mailbox.
        _require(failure_id in mailbox_ids.get(failed_name, set()),
                 f"failure message {failure_id} of {failed_name!r} is not "
                 "waiting in the actor's mailbox")
        decoded_failures.append(FailureRecord(
            actor=failed_name,
            message_id=failure_id,
            supervisor=failure_supervisor,
            error_type=error_type,
            error_text=error_text,
            supervision_path=tuple(path),
        ))
    # Pause flag and pending record must agree, both ways.
    _require(paused_names == failed_actors,
             "paused actors and pending failure records disagree")

    # Cross-structure relation: a dedup record always points at the first
    # delivery accepted for its key, which is still waiting in a mailbox,
    # already completed and present in the trace, or removed by a
    # supervision drop while the key stays reserved.
    reachable_ids.update(dropped_ids)
    for name, _state, _mailbox, dedup, _sup, _paused in decoded_actors:
        for key, first_id in dedup.items():
            _require(first_id in reachable_ids,
                     f"dedup key {key!r} of actor {name!r} references an "
                     "unknown message")

    # Handler boundary: every snapshotted actor needs a handler; extra
    # mappings are ignored. Checked before assembly so a failure produces
    # no runtime at all.
    for name, *_rest in decoded_actors:
        if name not in handlers:
            raise LookupError(f"no handler provided for actor: {name!r}")

    runtime = cls()
    runtime._clock = clock
    runtime._next_message_id = next_message_id
    for order, (name, state, mailbox, dedup, supervisor, paused) \
            in enumerate(decoded_actors):
        actor = _Actor(name, state, handlers[name], order, supervisor)
        for message_id, priority, message in mailbox:
            heapq.heappush(actor.mailbox, (-priority, message_id, message))
        actor.dedup_keys = dict(dedup)
        actor.paused = paused
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
    runtime._dropped_message_ids = set(dropped_ids)
    return runtime
