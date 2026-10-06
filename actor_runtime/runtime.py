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
"""
from __future__ import annotations

import copy
import heapq
from typing import Any, Callable, NamedTuple

__all__ = [
    "ActorContext",
    "ActorDataCopyError",
    "ActorExecutionError",
    "ActorRuntime",
    "AdvanceResult",
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

    The failing message stays unacknowledged in its actor's mailbox and no
    state or derived delivery is committed for it. Messages completed earlier
    in the same :meth:`~ActorRuntime.run` call remain committed.
    """

    def __init__(self, actor_name: str, message_id: int, original: BaseException):
        self.actor_name = actor_name
        self.message_id = message_id
        self.original = original
        super().__init__(
            f"actor {actor_name!r} failed while handling message {message_id}: "
            f"{type(original).__name__}: {original}"
        )


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
        if target not in self._runtime._actors:
            raise LookupError(f"unknown actor: {target!r}")
        self._pending.append((target, message))


Handler = Callable[[Any, Any, ActorContext], Any]


def _copy_data(what: str, value: Any) -> Any:
    """Deep-copy ``value``, reporting any failure as ActorDataCopyError."""
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise ActorDataCopyError(what, exc) from exc


class _Actor:
    __slots__ = ("name", "state", "handler", "mailbox", "order")

    def __init__(self, name: str, state: Any, handler: Handler, order: int):
        self.name = name
        self.state = state
        self.handler = handler
        self.order = order
        # Heap of (-priority, message_id, message); message_id is unique, so
        # the message object never participates in comparison.
        self.mailbox: list[tuple[int, int, Any]] = []


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
    """Deterministic in-memory actor runtime."""

    def __init__(self) -> None:
        self._actors: dict[str, _Actor] = {}
        self._trace: list[TraceEntry] = []
        self._next_message_id = 1
        self._clock = 0
        # Pending timed deliveries, kept in scheduling (message-id) order;
        # _release_due scans and rebuilds it.
        self._scheduled: list[_ScheduledDelivery] = []

    # -- registration -----------------------------------------------------

    def register(self, name: str, initial_state: Any, handler: Handler) -> None:
        """Register a named actor.

        Raises :class:`ValueError` when ``name`` is empty or already
        registered; a failed registration changes nothing.
        """
        if not isinstance(name, str) or name == "":
            raise ValueError("actor name must be a non-empty string")
        if name in self._actors:
            raise ValueError(f"actor already registered: {name!r}")
        self._actors[name] = _Actor(
            name, copy.deepcopy(initial_state), handler, len(self._actors)
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
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        actor = self._actors.get(target)
        if actor is None:
            raise LookupError(f"unknown actor: {target!r}")
        # Copy before assigning an id: a failed copy leaves the mailbox,
        # the trace, the actor state and the next message id untouched.
        stored = _copy_data("message", message)
        message_id = self._next_message_id
        self._next_message_id += 1
        heapq.heappush(actor.mailbox, (-priority, message_id, stored))
        return message_id

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
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise TypeError("delay must be an integer")
        if delay < 0:
            raise ValueError("delay must be a non-negative integer")
        if ttl is not None and (
            not isinstance(ttl, int) or isinstance(ttl, bool)
        ):
            raise TypeError("ttl must be None or an integer")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be a positive integer or None")
        actor = self._actors.get(target)
        if actor is None:
            raise LookupError(f"unknown actor: {target!r}")
        # Copy before assigning an id: a failed copy queues nothing and
        # consumes no id, exactly like send.
        stored = _copy_data("message", message)
        message_id = self._next_message_id
        self._next_message_id += 1
        scheduled_at = self._clock
        release_at = scheduled_at + delay
        expire_at = None if ttl is None else scheduled_at + ttl
        if delay == 0:
            # ttl is a strictly positive integer, so a zero-delay delivery is
            # never already expired: it enters the mailbox straight away.
            heapq.heappush(actor.mailbox, (-priority, message_id, stored))
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
                heapq.heappush(
                    delivery.actor.mailbox,
                    (-delivery.priority, delivery.message_id,
                     delivery.message),
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
        completes. The first handler failure aborts the run: an ordinary
        exception is raised as :class:`ActorExecutionError`, while a
        :class:`BaseException` that is not an :class:`Exception` (such as
        :class:`KeyboardInterrupt`) propagates unchanged. A failure to
        deep-copy the state, the message, the handler result or a buffered
        derived message is raised as :class:`ActorExecutionError` whose
        ``original`` is an :class:`ActorDataCopyError`. Either way the
        failing message stays pending and nothing is committed for it.
        Returns the number of messages completed by this call.
        """
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")

        processed = 0
        while limit is None or processed < limit:
            actor = None
            for candidate in self._actors.values():  # registration order
                if candidate.mailbox:
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
            except ActorDataCopyError as exc:
                heapq.heappush(actor.mailbox, (neg_priority, message_id, message))
                raise ActorExecutionError(
                    actor.name, message_id, exc
                ) from exc
            except BaseException:
                # Keep the message unacknowledged; commit nothing. The
                # message stays pending either way.
                heapq.heappush(actor.mailbox, (neg_priority, message_id, message))
                raise

            # Commit: only runtime-owned bookkeeping below, so nothing here
            # can fail because of user data. Derived sends enqueue only
            # after successful completion.
            for derived_target, stored in staged:
                derived_id = self._next_message_id
                self._next_message_id += 1
                derived_actor = self._actors[derived_target]
                heapq.heappush(derived_actor.mailbox, (0, derived_id, stored))

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

    # -- read-only queries ------------------------------------------------

    def get_state(self, actor_name: str) -> Any:
        """Return an independent snapshot of the actor's current state."""
        actor = self._require_actor(actor_name)
        return copy.deepcopy(actor.state)

    def pending_count(self, actor_name: str) -> int:
        """Return the number of unprocessed messages for the actor.

        Only messages that have already entered the mailbox are counted;
        scheduled deliveries that have not been released (and not expired) are
        reported separately by :meth:`scheduled_count`.
        """
        actor = self._require_actor(actor_name)
        return len(actor.mailbox)

    def scheduled_count(self, actor_name: str) -> int:
        """Return the actor's deliveries neither released nor expired.

        Raises :class:`LookupError` for an unknown actor.
        """
        actor = self._require_actor(actor_name)
        return sum(
            1 for delivery in self._scheduled if delivery.actor is actor
        )

    def trace(self) -> list[TraceEntry]:
        """Return completed processings in completion order.

        A fresh list of independent snapshots is returned each time; mutating
        it or the states inside entries cannot affect the runtime.
        """
        return copy.deepcopy(self._trace)

    def _require_actor(self, name: str) -> _Actor:
        actor = self._actors.get(name)
        if actor is None:
            raise LookupError(f"unknown actor: {name!r}")
        return actor
