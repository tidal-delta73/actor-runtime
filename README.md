# actor-runtime

Deterministic actor runtime.

Pure-Python, no runtime dependencies. Requires Python 3.10+.

## Usage

```bash
python3 -m actor_runtime version
python3 -m actor_runtime help
```

## In-memory actor runtime

Register named actors, deliver messages to their mailboxes and explicitly
advance the runtime. Processing order is fully deterministic and depends only
on registration order, message priority and delivery id — never on wall-clock
time, randomness or threads.

```python
from actor_runtime import ActorRuntime, ActorContext

def counter(state, message, ctx: ActorContext):
    state = dict(state)
    state[message] = state.get(message, 0) + 1
    if state[message] == 1:
        ctx.send("logger", f"first sighting of {message}")
    return state

rt = ActorRuntime()
rt.register("counter", {}, counter)
rt.register("logger", [], lambda s, m, ctx: s + [m])

msg_id = rt.send("counter", "hello", priority=10)  # -> 1, monotonic
rt.send("counter", "hello")                        # default priority 0
rt.run()                  # process until all mailboxes are empty
rt.run(limit=5)           # or process at most 5 messages

rt.get_state("counter")   # independent snapshot of current state
rt.pending_count("counter")
rt.trace()                # list of TraceEntry, in completion order
```

### Semantics

- An actor is a unique non-empty string name, an initial state and a handler
  `handler(state, message, ctx) -> new_state`. Registering an empty or
  duplicate name raises `ValueError`.
- `send(target, message, priority=0)` returns a runtime-wide monotonic
  message id and enqueues a private copy in the actor's mailbox. Unknown
  targets raise `LookupError`; non-integer priority raises `TypeError`;
  a message that cannot be deep-copied raises `ActorDataCopyError`
  (exposing the copy's own exception as `original`). Failed deliveries
  consume no id and change no mailbox.
- Scheduling: the actor with the earliest registration order and a non-empty
  mailbox is selected; within a mailbox, higher priority wins, then lower
  message id. Messages a handler sends through `ctx.send` are enqueued only
  after the handler returns successfully.
- Atomic commit: state and derived deliveries are committed together only
  after the handler returns normally and every boundary copy (pre-processing
  state, handler message, returned state, buffered derived messages) has
  succeeded. If the handler raises, or any of those copies fails, the
  message stays unacknowledged, nothing is committed -- no state change, no
  derived delivery, no consumed id, no trace entry -- and `run` raises
  `ActorExecutionError` (exposing `actor_name`, `message_id` and
  `original`; for copy failures `original` is an `ActorDataCopyError`);
  earlier completed messages remain committed.
- `run(limit=None)` processes until idle; a non-positive or non-integer
  `limit` raises `ValueError` without consuming any message. `BaseException`
  subclasses (e.g. `KeyboardInterrupt`) propagate unwrapped.
- `get_state`, `pending_count` and `trace` are read-only and return
  independent copies; unknown actors raise `LookupError`. Each
  `TraceEntry` has `message_id`, `actor_name`, `priority`, `state_before`
  and `state_after`.

Only single-process in-memory semantics are provided: no persistence,
supervision, timed messages or parallel scheduling.

## Tests

```bash
python3 -m unittest discover -s tests
```
