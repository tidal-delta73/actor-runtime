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

### Supervision trees

Actors registered without a `supervisor` argument are roots and keep the
unsupervised failure contract unchanged: a handler failure aborts `run`
with `ActorExecutionError`. Passing an already registered actor's name as
`register(name, state, handler, supervisor=...)` makes it the new actor's
**direct supervisor**; supervision edges form a forest and never
participate in scheduling.

```python
rt.register("root", [], handler)
rt.register("worker-a", [], handler, supervisor="root")
rt.register("worker-b", [], handler, supervisor="root")
```

- Validation: a non-string `supervisor` raises `TypeError`; `""` or
  self-supervision raises `ValueError`; an unknown supervisor raises
  `LookupError`; name checks stay as before. A failed registration changes
  nothing — no state, no order slot, no supervision edge.
- When a **supervised** actor's handler raises an `Exception` (including an
  `ActorDataCopyError` at the copy boundary), the existing rollback
  semantics apply exactly as for roots: the failing message keeps its
  original id and priority in the mailbox and no state, derived delivery,
  consumed id or trace entry is committed. Instead of raising, the actor is
  **paused**, `run` ends and returns the number of messages already
  completed, and an immutable `FailureRecord(actor, message_id,
  supervisor, error_type, error_text, supervision_path)` is produced.
  `BaseException` subclasses (e.g. `KeyboardInterrupt`) still propagate
  unwrapped, for every actor.
- While paused, new deliveries and released timed messages still enter the
  actor's mailbox (counted by `pending_count`); `run` skips the paused
  actor and keeps processing the others.
- Pending records are queried in production order with `rt.failures()`
  (independent copies). The current supervisor resolves one with
  `resolve_failure(supervisor, actor, message_id, action)`:
  - `"retry"` — delete the record and resume; the original message is
    retried by the next `run` with its id and priority intact;
  - `"drop"` — delete the failing message and resume, with no trace entry
    and no consumed id; `send_once` keys pointing at it are retained;
  - `"escalate"` — hand the record to the supervisor's own supervisor;
    the record is rewritten in place (same production-order slot, same
    failure message — nothing is copied or renumbered) and the failed
    actor stays paused. Escalating a root's failure raises
    `SupervisionError` and keeps the record and pause.
- An unknown record, or a caller that is not the record's current
  supervisor, raises `LookupError`; an unknown action raises
  `ValueError`. A failed resolution changes nothing.
- Supervision edges, pause marks, pending records and their order, and the
  ids removed by `drop` are all part of snapshots: after
  `restore_snapshot` the same actions and input sequence reproduce the
  same states, traces and message numbering.

### Deterministic snapshots

`rt.export_snapshot()` returns the complete runtime state as persistable
`bytes`; `ActorRuntime.restore_snapshot(data, handlers)` resumes a new
runtime from those bytes plus a mapping of actor name to handler. Handlers
are code and are never serialised — restoring never imports modules or
executes objects carried by the snapshot.

```python
data = rt.export_snapshot()          # bytes: write them anywhere
restored = ActorRuntime.restore_snapshot(data, {"counter": counter, "logger": logger})
restored.clock()                    # the snapshot moment
restored.send_once("counter", "order-42", {"n": 1})
# DedupResult(message_id=1, accepted=False) — dedup records survive
```

- The snapshot covers actor registration order and current states, the
  supervision tree (each actor's direct supervisor) and pause marks, every
  unacknowledged mailbox message with its priority and id, the pending
  supervision failures in production order with their current supervisor
  and supervision path, the ids removed by a supervision `drop`, the timed
  deliveries neither released nor expired, the dedup records, the logical
  clock, the next message id and the completion trace.
- Snapshot-safe data: `None`, booleans, integers, finite floats, strings,
  bytes and lists, tuples and string-keyed dicts composed recursively from
  those. Dicts with the same content in different key order produce
  identical bytes; two runtimes with identical observable state export
  byte-identical snapshots, and two runtimes restored from the same
  snapshot evolve identically under the same call sequence.
- Export raises `SnapshotError` on sets, custom objects, non-string mapping
  keys, NaN/infinite floats or circular references — and changes nothing:
  states, queues, clock, dedup records, trace and numbering stay as they
  were. In-memory behaviour for arbitrary deep-copyable objects is
  unchanged; the restriction only applies while exporting.
- Restore raises `SnapshotError` for truncated, tampered, unsupported
  version, missing-field or internally inconsistent bytes, and
  `LookupError` when `handlers` lacks an actor present in the snapshot
  (extra entries are ignored). A failed restore returns no runtime at all.
- After a successful restore every read-only query immediately reflects
  the snapshot moment, old dedup keys still confirm their first id, new
  deliveries continue numbering from the saved next id, timed deliveries
  keep their original deadlines and `run` keeps its registration-order,
  priority, rollback and derived-message commit rules.

### Semantics

- An actor is a unique non-empty string name, an initial state and a handler
  `handler(state, message, ctx) -> new_state`. Registering an empty or
  duplicate name raises `ValueError`. An optional fourth argument names an
  already registered direct supervisor; a non-string value raises
  `TypeError`, an empty name or self-supervision raises `ValueError` and an
  unknown name raises `LookupError`, with no state added on failure.
  Actors without a supervisor are roots.
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
  entry); earlier completed messages remain committed. For an actor
  without a supervisor `run` then raises `ActorExecutionError` (exposing
  `actor_name`, `message_id` and `original` — an `ActorDataCopyError` for
  copy failures). For a supervised actor the same rollback happens but the
  actor pauses and `run` returns the count completed so far, leaving a
  `FailureRecord` for `resolve_failure` instead of raising.
- `run(limit=None)` processes until idle; a non-positive or non-integer
  `limit` raises `ValueError` without consuming any message. `BaseException`
  subclasses (e.g. `KeyboardInterrupt`) propagate unwrapped.
- `get_state`, `pending_count` and `trace` are read-only and return
  independent copies; unknown actors raise `LookupError`. Each
  `TraceEntry` has `message_id`, `actor_name`, `priority`, `state_before`
  and `state_after`.

Only single-process semantics are provided: the supervision tree is
synchronous and explicit — failures pause one actor and wait for a
`resolve_failure` decision, there is no parallel scheduling, automatic
restart or implicit persistence — snapshots are exported and restored only
through the explicit `export_snapshot` / `restore_snapshot` entries.
Time is logical and only advances via explicit `advance` calls — there is
no wall-clock access and no threads.

## Tests

```bash
python3 -m unittest discover -s tests
```
