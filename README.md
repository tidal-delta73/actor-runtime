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
time, randomness or threads. Timed delivery uses a caller-driven logical
clock: `schedule` parks a message and `advance` moves the clock, so time is
just another replayable input.

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
  targets raise `LookupError`; non-integer priority raises `TypeError`; a
  message that cannot be deep-copied raises `ActorDataCopyError`.
  Failed deliveries consume no id.
- Scheduling: the actor with the earliest registration order and a non-empty
  mailbox is selected; within a mailbox, higher priority wins, then lower
  message id. Messages a handler sends through `ctx.send` are enqueued only
  after the handler returns successfully.
- Atomic commit: state and derived deliveries are committed together only
  after the handler returns normally and every result has been copied. If
  the handler raises, or copying the state, the message, the handler result
  or a buffered derived message fails, the message stays unacknowledged,
  nothing is committed (no state, no derived delivery, no id, no trace
  entry) and `run` raises `ActorExecutionError` (exposing `actor_name`,
  `message_id` and `original` — an `ActorDataCopyError` for copy
  failures); earlier completed messages remain committed.
- `run(limit=None)` processes until idle; a non-positive or non-integer
  `limit` raises `ValueError` without consuming any message. `BaseException`
  subclasses (e.g. `KeyboardInterrupt`) propagate unwrapped. `run` never
  advances the logical clock.
- `now()` returns the current logical clock tick (0 initially). The clock
  only moves through `advance`; the runtime never reads wall-clock time.
- `schedule(target, message, delay, ttl=None, priority=0)` parks a timed
  delivery and returns a message id from the same monotonic sequence as
  `send`. `delay` (non-negative integer) is the number of ticks from now
  until the message is due; `ttl` (`None` or positive integer) is how many
  ticks from now it stays alive. `delay == 0` enters the mailbox
  immediately. Validation mirrors `send`: non-integer (or boolean)
  `priority`/`delay`/`ttl` raises `TypeError`, negative `delay` or
  non-positive `ttl` raises `ValueError`, unknown targets raise
  `LookupError`, uncopyable messages raise `ActorDataCopyError` — all
  without consuming an id or changing any queue.
- `advance(ticks)` moves the clock forward by a positive integer number of
  ticks and returns an immutable `AdvanceResult(now, released, expired)`.
  A parked message due at or before the new tick is released into its
  actor's mailbox with its scheduled id and priority; one whose ttl ran
  out expires instead — expiry is judged first, so it never reaches a
  handler, produces no `TraceEntry` and changes no state. `released` and
  `expired` are tuples ordered by event tick, then message id. Invalid
  `ticks` (`TypeError`/`ValueError`) changes nothing.
- `scheduled_count(actor)` counts parked timed deliveries not yet released
  or expired; `pending_count(actor)` counts only messages already in the
  mailbox. Unknown actors raise `LookupError`.
- `get_state`, `pending_count`, `scheduled_count` and `trace` are read-only
  and return independent copies; unknown actors raise `LookupError`. Each
  `TraceEntry` has `message_id`, `actor_name`, `priority`, `state_before`
  and `state_after`.

Only single-process in-memory semantics are provided: no persistence,
supervision or parallel scheduling.

## Tests

```bash
python3 -m unittest discover -s tests
```
