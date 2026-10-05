"""In-process, deterministic in-memory actor runtime.

The runtime is deliberately single-process and synchronous: callers register
named actors, deliver messages to their mailboxes and explicitly advance
processing with :meth:`ActorRuntime.run`. Scheduling is fully determined by
registration order, message priority and delivery id -- never by wall-clock
time, randomness or thread interleaving.
"""
from __future__ import annotations

import copy
import heapq
from typing import Any, Callable, NamedTuple

__all__ = [
    "ActorContext",
    "ActorExecutionError",
    "ActorRuntime",
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


class ActorRuntime:
    """Deterministic in-memory actor runtime."""

    def __init__(self) -> None:
        self._actors: dict[str, _Actor] = {}
        self._trace: list[TraceEntry] = []
        self._next_message_id = 1

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
        :class:`LookupError` if ``target`` is unknown; a failed delivery
        consumes no id and changes no mailbox.
        """
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        actor = self._actors.get(target)
        if actor is None:
            raise LookupError(f"unknown actor: {target!r}")
        message_id = self._next_message_id
        self._next_message_id += 1
        heapq.heappush(actor.mailbox, (-priority, message_id, copy.deepcopy(message)))
        return message_id

    # -- execution --------------------------------------------------------

    def run(self, limit: int | None = None) -> int:
        """Process pending messages until mailboxes empty or ``limit`` hit.

        Derived messages become eligible only after the handling message
        completes. The first handler failure aborts the run: an ordinary
        exception is raised as :class:`ActorExecutionError`, while a
        :class:`BaseException` that is not an :class:`Exception` (such as
        :class:`KeyboardInterrupt`) propagates unchanged. Either way the
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
            state_before = copy.deepcopy(actor.state)

            # Hand the handler private copies so a failure (including an
            # in-place mutation before raising) can never commit state or
            # alter the message kept in the mailbox.
            context = ActorContext(self)
            try:
                new_state = actor.handler(
                    copy.deepcopy(actor.state),
                    copy.deepcopy(message),
                    context,
                )
            except BaseException as exc:
                # Keep the message unacknowledged; commit nothing. Ordinary
                # exceptions are wrapped; BaseException subclasses that are
                # not Exception (KeyboardInterrupt, SystemExit, ...) propagate
                # unchanged, but the message still stays pending either way.
                heapq.heappush(actor.mailbox, (neg_priority, message_id, message))
                if isinstance(exc, Exception):
                    raise ActorExecutionError(
                        actor.name, message_id, exc
                    ) from exc
                raise

            # Commit: derived sends enqueue only after successful completion.
            for derived_target, derived_message in context._pending:
                derived_id = self._next_message_id
                self._next_message_id += 1
                derived_actor = self._actors[derived_target]
                heapq.heappush(
                    derived_actor.mailbox,
                    (0, derived_id, copy.deepcopy(derived_message)),
                )

            actor.state = new_state
            self._trace.append(
                TraceEntry(
                    message_id=message_id,
                    actor_name=actor.name,
                    priority=priority,
                    state_before=state_before,
                    state_after=copy.deepcopy(new_state),
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
        """Return the number of unprocessed messages for the actor."""
        actor = self._require_actor(actor_name)
        return len(actor.mailbox)

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
