"""In-process, deterministic in-memory actor runtime.

The runtime is deliberately single-process and synchronous: callers register
named actors, deliver messages to their mailboxes and explicitly advance
processing with :meth:`ActorRuntime.run`. Scheduling is fully determined by
registration order, message priority and delivery id -- never by wall-clock
time, randomness or thread interleaving.

Timed delivery is equally explicit: :meth:`ActorRuntime.schedule` parks a
message against a caller-driven logical clock and :meth:`ActorRuntime.advance`
moves that clock forward, releasing due messages into mailboxes and expiring
the ones whose time-to-live ran out. The runtime never reads wall-clock time;
the clock only moves when the caller advances it, so time is just another
replayable input.
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

    ``now`` is the clock value after the advance. ``released`` holds the ids
    of scheduled messages that became due and entered their actor's mailbox;
    ``expired`` holds the ids of scheduled messages whose time-to-live ran
    out before release. Both are ordered by event tick, then by message id,
    and both are immutable tuples.
    """

    now: int
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


class _Scheduled(NamedTuple):
    """One parked timed delivery.

    ``fire_at`` is the tick at which the message becomes due (schedule tick
    plus delay); ``expire_at`` is the tick at which it dies (schedule tick
    plus ttl) or ``None`` when it never expires.
    """

    message_id: int
    fire_at: int
    expire_at: int | None
    priority: int
    message: Any


class _Actor:
    __slots__ = ("name", "state", "handler", "mailbox", "order", "scheduled")

    def __init__(self, name: str, state: Any, handler: Handler, order: int):
        self.name = name
        self.state = state
        self.handler = handler
        self.order = order
        # Heap of (-priority, message_id, message); message_id is unique, so
        # the message object never participates in comparison.
        self.mailbox: list[tuple[int, int, Any]] = []
        # Parked timed deliveries, in schedule call order.
        self.scheduled: list[_Scheduled] = []


class ActorRuntime:
    """Deterministic in-memory actor runtime."""

    def __init__(self) -> None:
        self._actors: dict[str, _Actor] = {}
        self._trace: list[TraceEntry] = []
        self._next_message_id = 1
        self._clock = 0

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

    # -- logical clock ------------------------------------------------------

    def now(self) -> int:
        """Return the current logical clock tick (0 before any advance)."""
        return self._clock

    def schedule(
        self,
        target: str,
        message: Any,
        delay: int,
        ttl: int | None = None,
        priority: int = 0,
    ) -> int:
        """Park a timed delivery for ``target`` and return its message id.

        Ids come from the same runtime-wide monotonic sequence as
        :meth:`send`. ``delay`` is the number of ticks from the current tick
        after which the message becomes due; ``ttl`` is the number of ticks
        from now the message stays alive, or ``None`` for no expiry. A
        message with ``delay == 0`` has not yet expired (``ttl`` is at least
        1) and therefore enters the mailbox immediately; any other message
        is parked until :meth:`advance` releases or expires it.

        Raises :class:`TypeError` if ``priority``, ``delay`` or ``ttl`` has
        a non-integer (or boolean) type, :class:`ValueError` if ``delay``
        is negative or ``ttl`` is not positive, :class:`LookupError` if
        ``target`` is unknown and :class:`ActorDataCopyError` if ``message``
        cannot be deep-copied. A failed schedule consumes no id and changes
        no clock, mailbox or scheduled queue.
        """
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        if not isinstance(delay, int) or isinstance(delay, bool):
            raise TypeError("delay must be an integer")
        if delay < 0:
            raise ValueError("delay must be a non-negative integer")
        if ttl is not None:
            if not isinstance(ttl, int) or isinstance(ttl, bool):
                raise TypeError("ttl must be None or an integer")
            if ttl <= 0:
                raise ValueError("ttl must be a positive integer")
        actor = self._actors.get(target)
        if actor is None:
            raise LookupError(f"unknown actor: {target!r}")
        # Copy before assigning an id, exactly like send: a failed copy
        # leaves the clock, the queues and the next message id untouched.
        stored = _copy_data("message", message)
        message_id = self._next_message_id
        self._next_message_id += 1
        if delay == 0:
            heapq.heappush(actor.mailbox, (-priority, message_id, stored))
        else:
            actor.scheduled.append(
                _Scheduled(
                    message_id=message_id,
                    fire_at=self._clock + delay,
                    expire_at=None if ttl is None else self._clock + ttl,
                    priority=priority,
                    message=stored,
                )
            )
        return message_id

    def advance(self, ticks: int) -> AdvanceResult:
        """Advance the logical clock by ``ticks`` and settle timed deliveries.

        Every parked message whose expiry tick has been reached expires --
        it is dropped without touching a handler, the state or the trace --
        and every remaining parked message whose due tick has been reached
        is released into its actor's mailbox, keeping the id and priority it
        was scheduled with. Expiry is judged before release, so a message
        whose ttl ran out at or before its due tick never arrives. Released
        messages are processed by later :meth:`run` calls under the usual
        mailbox ordering; :meth:`run` itself never moves the clock.

        Returns an immutable :class:`AdvanceResult` with the new clock value
        and the released/expired message ids, each ordered by event tick
        then message id. Raises :class:`TypeError` if ``ticks`` is not an
        integer (booleans included) and :class:`ValueError` if it is not
        positive; a failed call changes no clock, id or queue.
        """
        if not isinstance(ticks, int) or isinstance(ticks, bool):
            raise TypeError("ticks must be an integer")
        if ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        self._clock += ticks
        now = self._clock

        released: list[tuple[int, int, _Actor, _Scheduled]] = []
        expired: list[tuple[int, int]] = []
        for actor in self._actors.values():  # registration order
            kept = []
            for entry in actor.scheduled:
                if entry.expire_at is not None and now >= entry.expire_at:
                    expired.append((entry.expire_at, entry.message_id))
                elif now >= entry.fire_at:
                    released.append(
                        (entry.fire_at, entry.message_id, actor, entry)
                    )
                else:
                    kept.append(entry)
            actor.scheduled = kept

        # Stable event order: by the tick the event fell due, then by id.
        released.sort(key=lambda item: (item[0], item[1]))
        expired.sort(key=lambda item: (item[0], item[1]))
        for _, _, actor, entry in released:
            heapq.heappush(
                actor.mailbox, (-entry.priority, entry.message_id, entry.message)
            )
        return AdvanceResult(
            now=now,
            released=tuple(item[1] for item in released),
            expired=tuple(item[1] for item in expired),
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
        Returns the number of messages completed by this call. ``run``
        never advances the logical clock and never touches parked timed
        deliveries; only :meth:`advance` releases or expires them.
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

        Only messages already in the mailbox count; parked timed deliveries
        are reported by :meth:`scheduled_count` until they are released.
        """
        actor = self._require_actor(actor_name)
        return len(actor.mailbox)

    def scheduled_count(self, actor_name: str) -> int:
        """Return the number of parked timed deliveries for the actor.

        A scheduled message counts until :meth:`advance` releases it into
        the mailbox or expires it. Unknown actors raise :class:`LookupError`.
        """
        actor = self._require_actor(actor_name)
        return len(actor.scheduled)

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
