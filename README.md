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

### Logical clock and timed delivery

Time is an explicitly replayed input, never wall-clock time: a logical clock
starts at 0 and only moves when `advance` is called.

```python
rt.clock()                          # -> 0, the current logical tick
mid = rt.schedule("counter", "later", delay=5, ttl=10, priority=0)
# mid comes from the same runtime-wide monotonic sequence as send.
# delay: ticks from the current tick until the message is released.
# ttl:   optional ticks (measured from scheduling) the delivery stays alive;
#        None (the default) means it never expires.

result = rt.advance(5)              # AdvanceResult(time=5, released=(mid,), expired=())
# result.released / result.expired are immutable tuples of message ids,
# stably ordered by the tick the event happened at and then by message id.
rt.pending_count("counter")         # released mail is ordinary mailbox mail
rt.scheduled_count("counter")       # deliveries neither released nor expired
rt.run()                            # advance never invokes a handler
```

- `delay=0` releases immediately into the mailbox; `schedule` itself never
  moves the clock, and neither does `run`.
- A delivery is released when `now >= scheduled_at + delay`; it expires when
  `now >= scheduled_at + ttl`. When both happen on the same tick, expiry is
  decided first, so the message never enters a mailbox or reaches a handler.
  A delivery already released is ordinary mail — a later ttl tick does not
  remove it.
- Released messages keep the id returned by `schedule` and their scheduled
  priority, so they participate in the usual registration-order /
  priority / id scheduling. They only enter the trace after a successful
  `run`, exactly like `send` messages; expired messages produce no handler
  call, no trace entry and no state change.
- Validation: `priority` must be a non-boolean integer (`TypeError`
  otherwise); `delay` must be a non-boolean non-negative integer
  (`TypeError`/`ValueError`); `ttl` must be `None` or a non-boolean positive
  integer; `advance` takes a non-boolean positive integer. Unknown targets
  raise `LookupError`; an uncopyable scheduled message raises
  `ActorDataCopyError` and consumes no id. Any failed `schedule`/`advance`
  leaves the clock, the id counter and every queue untouched.

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
  subclasses (e.g. `KeyboardInterrupt`) propagate unwrapped.
- `get_state`, `pending_count` and `trace` are read-only and return
  independent copies; unknown actors raise `LookupError`. Each
  `TraceEntry` has `message_id`, `actor_name`, `priority`, `state_before`
  and `state_after`.

Only single-process in-memory semantics are provided: no persistence,
supervision or parallel scheduling. Time is logical and only advances via
explicit `advance` calls — there is no wall-clock access and no threads.

## Tests

```bash
python3 -m unittest discover -s tests
```
