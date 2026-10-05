"""In-memory actor runtime: named actors, prioritized mailboxes, deterministic runs.

Single-process semantics only: no persistence, supervision, timers, or threads.
Given the same registration order, initial states, and delivery sequence, a run
produces identical states, pending counts, and traces every time.
"""
from __future__ import annotations

import copy
import heapq
import inspect
from typing import Any, Callable, NamedTuple

__all__ = ["ActorRuntime", "ActorExecutionError", "TraceEntry"]


class ActorExecutionError(Exception):
    """Raised by ``ActorRuntime.run`` when an actor's handler throws.

    Carries the actor name, the id of the unacknowledged message, and the
    original exception (also chained as ``__cause__``).
    """

    def __init__(self, actor_name: str, message_id: int, original: BaseException):
        self.actor_name = actor_name
        self.message_id = message_id
        self.original = original
        super().__init__(
            f"actor {actor_name!r} failed while handling message "
            f"{message_id}: {original!r}"
        )


class TraceEntry(NamedTuple):
    """One completed message, in completion order. Immutable."""

    message_id: int
    actor: str
    priority: int
    state_before: Any
    state_after: Any


class _Actor:
    __slots__ = ("handler", "takes_send", "state", "mailbox")

    def __init__(self, handler: Callable, takes_send: bool, state: Any):
        self.handler = handler
        self.takes_send = takes_send
        self.state = state
        # Heap of (-priority, message_id, message); message_id is unique so the
        # message payload itself is never compared.
        self.mailbox: list[tuple[int, int, Any]] = []


def _takes_send(handler: Callable) -> bool:
    """True if the handler accepts a third positional argument (the send hook)."""
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional >= 3


class ActorRuntime:
    """A deterministic, explicitly advanced in-memory actor runtime."""

    def __init__(self) -> None:
        self._actors: dict[str, _Actor] = {}
        self._next_message_id = 1
        self._trace: list[TraceEntry] = []
        # While a handler runs, derived deliveries are buffered here and only
        # enqueued after the handler returns successfully.
        self._derived: list[tuple[str, tuple[int, int, Any]]] | None = None

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, name: str, initial_state: Any, handler: Callable) -> None:
        """Register a named actor. Raises ValueError on empty/duplicate names."""
        if not isinstance(name, str) or not name:
            raise ValueError("actor name must be a non-empty string")
        if name in self._actors:
            raise ValueError(f"actor already registered: {name!r}")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._actors[name] = _Actor(handler, _takes_send(handler), initial_state)

    # ------------------------------------------------------------------
    # delivery
    # ------------------------------------------------------------------
    def send(self, name: str, message: Any, priority: int = 0) -> int:
        """Deliver a message to an actor's mailbox; returns its message id.

        Ids are monotonically increasing per runtime. Higher priority is
        processed first; ties break by ascending message id. Calls made from
        inside a handler are buffered and enqueued only after that handler
        completes successfully.
        """
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        actor = self._actors.get(name)
        if actor is None:
            raise LookupError(f"unknown actor: {name!r}")
        message_id = self._next_message_id
        self._next_message_id += 1
        item = (-priority, message_id, message)
        if self._derived is not None:
            self._derived.append((name, item))
        else:
            heapq.heappush(actor.mailbox, item)
        return message_id

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def run(self, max_messages: int | None = None) -> int:
        """Process messages until mailboxes are empty or the limit is hit.

        Returns the number of messages processed. Raises ValueError if
        ``max_messages`` is not a positive integer (consuming nothing), and
        ActorExecutionError if a handler throws.
        """
        if max_messages is not None and (
            not isinstance(max_messages, int)
            or isinstance(max_messages, bool)
            or max_messages <= 0
        ):
            raise ValueError("max_messages must be a positive integer")
        processed = 0
        while max_messages is None or processed < max_messages:
            ready = self._next_ready()
            if ready is None:
                break
            name, actor = ready
            self._step(name, actor)
            processed += 1
        return processed

    def _next_ready(self) -> tuple[str, _Actor] | None:
        # Registration order decides among actors that all have pending mail.
        for name, actor in self._actors.items():
            if actor.mailbox:
                return name, actor
        return None

    def _step(self, name: str, actor: _Actor) -> None:
        neg_priority, message_id, message = heapq.heappop(actor.mailbox)
        state_snapshot = copy.deepcopy(actor.state)
        derived: list[tuple[str, tuple[int, int, Any]]] = []
        self._derived = derived
        try:
            if actor.takes_send:
                new_state = actor.handler(actor.state, message, self.send)
            else:
                new_state = actor.handler(actor.state, message)
        except Exception as exc:
            self._derived = None
            # Unacknowledged: the message stays pending, no state or derived
            # deliveries are committed; completed work so far is kept.
            heapq.heappush(actor.mailbox, (neg_priority, message_id, message))
            raise ActorExecutionError(name, message_id, exc) from exc
        self._derived = None
        # Commit only after the handler returned normally.
        actor.state = new_state
        for target, item in derived:
            heapq.heappush(self._actors[target].mailbox, item)
        self._trace.append(
            TraceEntry(
                message_id=message_id,
                actor=name,
                priority=-neg_priority,
                state_before=state_snapshot,
                state_after=copy.deepcopy(new_state),
            )
        )

    # ------------------------------------------------------------------
    # read-only queries
    # ------------------------------------------------------------------
    def state(self, name: str) -> Any:
        """Current state of an actor. Raises LookupError for unknown names."""
        actor = self._actors.get(name)
        if actor is None:
            raise LookupError(f"unknown actor: {name!r}")
        return copy.deepcopy(actor.state)

    def pending(self, name: str) -> int:
        """Number of unprocessed messages for an actor. LookupError if unknown."""
        actor = self._actors.get(name)
        if actor is None:
            raise LookupError(f"unknown actor: {name!r}")
        return len(actor.mailbox)

    def trace(self) -> list[TraceEntry]:
        """Execution trace in completion order (a detached copy)."""
        return copy.deepcopy(self._trace)
