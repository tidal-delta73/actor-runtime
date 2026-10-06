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

### Idempotent delivery (at-least-once protection)

`send_once(target, delivery_key, message, priority=0)` makes a delivery
idempotent on the `(target, delivery_key)` pair. The key scope is the target
actor, so different actors may reuse the same key freely. Plain `send`
deliveries never participate in deduplication.

```python
first = rt.send_once("counter", "order-42", {"n": 1})
# DedupResult(message_id=1, accepted=True) — same copy/id/priority rules as send
again = rt.send_once("counter", "order-42", {"n": 999})
# DedupResult(message_id=1, accepted=False) — placeholder message is never
# copied or compared; nothing is re-enqueued and no id is consumed.
rt.send_once("other", "order-42", {"n": 1})  # accepted: other actor, same key
```

- The first call for a pair deep-copies the message, takes the next
  runtime-wide monotonic id, enqueues it by the existing priority rules and
  returns `DedupResult(message_id, accepted=True)`.
- Every later call with the same target and key returns the **first** id with
  `accepted=False`, whether the first message is still pending, has already
  completed or is waiting in the mailbox for retry after a handler failure.
  It changes no state, trace entry or the original delivery's priority, and
  its `message` argument is never copied or compared.
- The dedup record is reserved at acceptance and lives for the whole lifetime
  of the runtime instance. If the first handler fails, the existing
  `ActorExecutionError` / message-stays-in-mailbox / transactional rollback
  contract applies unchanged and the key reservation is **not** withdrawn:
  re-submitting the same key only confirms the duplicate, while a later `run`
  retries the original message. A successful processing therefore produces at
  most one `TraceEntry`.
- Validation: `priority` that is not a non-boolean integer raises
  `TypeError`; a non-string `delivery_key` raises `TypeError` and an empty one
  raises `ValueError`; an unknown target raises `LookupError`. All of these
  checks run before the duplicate record is consulted. If the first message
  cannot be deep-copied, `ActorDataCopyError` is raised with no key reserved
  and no id consumed — after fixing the data, the same key can be retried as
  the first delivery.
- Determinism: for the same registrations, idempotent calls and `run` order,
  independent runtimes produce identical `DedupResult` sequences, final
  states and completion traces.

### Semantics

- An actor is a unique non-empty string name, an initial state and a handler
  `handler(state, message, ctx) -> new_state`. Registering an empty or
  duplicate name raises `ValueError`.
- `send(target, message, priority=0)` returns a runtime-wide monotonic
  message id and enqueues a private copy in the actor's mailbox. Unknown
  targets raise `LookupError`; non-integer priority raises `TypeError`; a
  message that cannot be deep-copied raises `ActorDataCopyError`.
  Failed deliveries consume no id.
- `send_once(target, delivery_key, message, priority=0)` is the idempotent
  variant: dedup is scoped to the `(target, delivery_key)` pair for the whole
  runtime lifetime, and it returns an immutable
  `DedupResult(message_id, accepted)` distinguishing the first acceptance
  from a duplicate confirmation. Plain `send`, `schedule`, `ctx.send`,
  mailbox selection, timed release/expiry, read-only queries and failure
  atomicity keep their current behaviour; ordinary deliveries do not
  deduplicate.
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

### Deterministic snapshots and recovery

`export_snapshot()` serialises the whole observable runtime into
deterministic `bytes`, and `ActorRuntime.restore_snapshot(data, handlers)`
builds a fresh runtime from those bytes plus a mapping of actor name to
handler. Handlers (and any executable code) are never serialised; recovery
imports nothing and executes no object carried by the snapshot.

```python
blob = rt.export_snapshot()                       # persist these bytes
rt2 = ActorRuntime.restore_snapshot(
    blob, {"counter": counter, "logger": logger}, # extras are ignored
)
```

The snapshot captures: actor registration order and current state; every
mailbox's unacknowledged messages with priority and message id; timed
deliveries neither released nor expired, with their original release/expiry
ticks; each `send_once` key's first message id; the logical clock; the next
message id; and the complete completion trace. Restored read-only queries
return independent copies immediately, old idempotent keys keep returning
their original id with `accepted=False`, new deliveries continue from the
saved next id, restored timers still release/expire at their original
deadlines (same-tick events stay id-sorted), and `run` keeps its
registration-order, priority, failure-rollback and derived-mail rules.

Snapshots are canonical JSON: equal observable states produce byte-for-byte
identical bytes regardless of mapping insertion order (signed zero is
canonicalised to `0.0`). Business data is limited to `None`, booleans,
integers, finite floats, strings, `bytes`, and recursively nested `list`,
`tuple` and string-keyed `dict`. Sets, custom instances, non-string mapping
keys, NaN/infinite floats and cyclic containers make `export_snapshot` raise
`SnapshotError`; export is a pure read, so the source runtime — mailboxes,
timers, clock, dedup records, trace and id counter — is unchanged. On
restore, truncated, tampered, unsupported-version, field-missing or
internally inconsistent bytes raise `SnapshotError`; a missing handler for
any snapshot actor raises `LookupError` (extras are ignored). Either failure
occurs before a runtime exists, so a partially restored runtime can never be
observed. Existing entry points and exception types are unchanged when
snapshots are not used.

Only single-process in-memory semantics are provided: no persistence,
supervision or parallel scheduling. Time is logical and only advances via
explicit `advance` calls — there is no wall-clock access and no threads.

## Tests

```bash
python3 -m unittest discover -s tests
```
