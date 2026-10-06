"""Deterministic replay regression tests over the public runtime API only.

The baseline already ships actor creation, mailboxes with priority, a logical
clock with timed delivery, derived (handler-context) sends and the observable
trace.  This module adds no product surface: it builds fixed scenarios only
through the documented entry points (``register``/``send``/``schedule``/
``advance``/``run`` and the read-only queries ``clock``/``get_state``/
``pending_count``/``scheduled_count``/``trace``) and the existing test replay
convention in :mod:`tests.replay_harness` (length-framed JSON journal +
``CorruptRecordError``).

Each scenario is recorded as a public *transcript* -- accepted deliveries
(target/priority/id/logical time), clock settlements (the public
``AdvanceResult``) and completion trace entries -- and then required to:

* repeat item-for-item when the same fixed inputs are driven through a fresh
  runtime at least twice under the same step (concurrency) configuration:
  event kind, actor id, message id, logical time, ordering and final state
  must all be equal;
* replay into a brand-new runtime from the first run's journal, without
  accepting any further external message and without any real timed wait
  (replay issues only the recorded logical ``advance`` calls; nothing sleeps),
  producing the same final queryable states and the same comparable events.

Corrupt journals -- a truncated tail, two records swapped out of their
recorded order, a delivery referencing an actor the journal never registers
-- must fail with the documented error type and must leave no partially
recovered, still-queryable state.  The existing conventions are:
``tests.replay_harness.CorruptRecordError`` for a structurally malformed
replay stream, and the runtime's own public :class:`LookupError` when a
delivery resolves an unknown target.

Comparison is always event-by-event (asserting the *index* of the first
divergence), never a final-state-only comparison, so a scheduling order drift
in the middle of a run cannot hide behind an equal end state.
"""
from __future__ import annotations

import json
import struct
import unittest

from actor_runtime import ActorContext, ActorRuntime

from tests.replay_harness import (
    A_COUNTER,
    A_EVENTS,
    HANDLERS,
    CorruptRecordError,
    initial_state_for,
)

# ---------------------------------------------------------------------------
# Two actors that interact.  The handlers are the existing, replay-proven
# fixture handlers, so the behaviour under test is the product's, not new
# test code:
#   events  records plain events and, for a message carrying ``then``, commits
#           a derived delivery to another actor *after* its own processing;
#   counter accumulates {"kind": "inc", "n": int}.
# ---------------------------------------------------------------------------

WORKER = A_EVENTS
PEER = A_COUNTER
ACTORS = (WORKER, PEER)

SCHEMA = 1
CODEC = "det-reg-v1"


def register_workers(rt: ActorRuntime) -> None:
    """Register the two interacting actors through the public entry point."""
    for name in ACTORS:
        rt.register(name, initial_state_for(name), HANDLERS[name])


# ---------------------------------------------------------------------------
# Public transcript
# ---------------------------------------------------------------------------
#
# Every item is built strictly from facts observed at public entry points --
# no private mailbox, scheduler field or persistence structure is read.
#
#   ("accept", target, priority, message_id, clock)
#       A delivery the runtime accepted (send/schedule); the id is the public
#       return value, captured at the call site of a fixed action sequence.
#   ("clock", time, released, expired)
#       One logical-clock settlement, taken verbatim from AdvanceResult.
#   ("done", message_id, actor_name, priority, state_before, state_after)
#       One completed processing, taken verbatim from a public TraceEntry.


def _normal_state(raw):
    """Canonicalise an actor state for comparison (sets -> sorted lists)."""
    if isinstance(raw, dict) and "seen" in raw and "arrivals" in raw:
        return {
            "seen": sorted(raw["seen"]),
            "arrivals": raw["arrivals"],
            "value": raw["value"],
        }
    return raw


def trace_events(rt: ActorRuntime) -> list:
    """All public completion events so far, in completion order."""
    return [
        (
            "done",
            e.message_id,
            e.actor_name,
            e.priority,
            _normal_state(e.state_before),
            _normal_state(e.state_after),
        )
        for e in rt.trace()
    ]


def final_snapshot(rt: ActorRuntime) -> dict:
    """Every final queryable public fact about a settled runtime."""
    return {
        "clock": rt.clock(),
        "states": {name: _normal_state(rt.get_state(name)) for name in ACTORS},
        "pending": {name: rt.pending_count(name) for name in ACTORS},
        "scheduled": {name: rt.scheduled_count(name) for name in ACTORS},
        "trace": trace_events(rt),
    }


def assert_transcripts_equal(testcase: unittest.TestCase,
                             first: list, second: list, label: str) -> None:
    """Compare two transcripts item by item, naming the first divergence."""
    testcase.assertEqual(
        len(first), len(second),
        f"{label}: transcript length differs near index "
        f"{min(len(first), len(second))}",
    )
    for index, (a, b) in enumerate(zip(first, second)):
        testcase.assertEqual(
            a, b,
            f"{label}: first transcript divergence at event index {index}:\n"
            f"  expected: {a!r}\n"
            f"  actual:   {b!r}",
        )


def assert_snapshots_equal(testcase: unittest.TestCase,
                           first: dict, second: dict, label: str) -> None:
    # Trace first (item by item, first mismatch named), then the other final
    # facts, so an intermediate scheduling drift is reported as a concrete
    # event rather than an opaque final-state inequality.
    assert_transcripts_equal(testcase, first["trace"], second["trace"], label)
    for key in ("clock", "states", "pending", "scheduled"):
        testcase.assertEqual(
            first[key], second[key],
            f"{label}: final {key} differs despite equal-length traces",
        )


# ---------------------------------------------------------------------------
# Fixed action script and its length-framed JSON journal
# ---------------------------------------------------------------------------
#
# The journal is test-local persistence (the same convention replay_harness
# establishes), not a product surface.  Each record carries a monotonic
# ``seq``: a stream whose frames arrive out of recorded order is structurally
# invalid and is rejected as CorruptRecordError before any runtime is built.
#
# Action tuples:
#   ("register", name)
#   ("send", target, message, priority)
#   ("schedule", target, message, delay, ttl, priority)
#   ("advance", ticks)
#   ("run", limit)            # None => drain to quiescence


def _json_dumps(obj) -> bytes:
    return (
        json.dumps(
            obj, ensure_ascii=True, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")


def _frame(rec: dict) -> bytes:
    body = _json_dumps(rec)
    return struct.pack("<I", len(body)) + body


def action_to_record(action: tuple, seq: int) -> dict:
    op = action[0]
    if op == "register":
        return {"t": "register", "seq": seq, "name": action[1]}
    if op == "send":
        return {"t": "send", "seq": seq, "target": action[1],
                "message": action[2], "priority": int(action[3])}
    if op == "schedule":
        return {"t": "schedule", "seq": seq, "target": action[1],
                "message": action[2], "delay": int(action[3]),
                "ttl": action[4], "priority": int(action[5])}
    if op == "advance":
        return {"t": "advance", "seq": seq, "ticks": int(action[1])}
    if op == "run":
        return {"t": "run", "seq": seq, "limit": action[1]}
    raise CorruptRecordError(f"unknown script action: {op!r}")


def encode_records_raw(records: list) -> bytes:
    """Frame a header followed by ``records`` in their given list order."""
    out = bytearray()
    out += _frame({"t": "header", "schema": SCHEMA, "codec": CODEC})
    for rec in records:
        out += _frame(rec)
    return bytes(out)


def encode_script(actions: list) -> bytes:
    """Length-framed JSON journal of the fixed action script."""
    return encode_records_raw(
        [action_to_record(action, seq) for seq, action in enumerate(actions)]
    )


def read_records(blob: bytes) -> list:
    """Parse frames, validating framing, the header and the seq ordering."""
    records: list = []
    pos = 0
    total = len(blob)
    saw_header = False
    while pos < total:
        if total - pos < 4:
            raise CorruptRecordError("truncated record header")
        (length,) = struct.unpack("<I", blob[pos:pos + 4])
        pos += 4
        if total - pos < length:
            raise CorruptRecordError("truncated record body")
        try:
            rec = json.loads(blob[pos:pos + length].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptRecordError(f"undecodable record: {exc}") from exc
        pos += length
        if not isinstance(rec, dict) or "t" not in rec:
            raise CorruptRecordError("record missing type tag")
        if not saw_header:
            if rec.get("t") != "header":
                raise CorruptRecordError("journal missing header frame")
            if rec.get("schema") != SCHEMA or rec.get("codec") != CODEC:
                raise CorruptRecordError("incompatible journal header")
            saw_header = True
            continue
        if rec["t"] == "header":
            raise CorruptRecordError("duplicate header frame")
        records.append(rec)
    if not saw_header:
        raise CorruptRecordError("empty journal")
    return records


REQUIRED_FIELDS = {
    "register": ("seq", "name"),
    "send": ("seq", "target", "message", "priority"),
    "schedule": ("seq", "target", "message", "delay", "ttl", "priority"),
    "advance": ("seq", "ticks"),
    "run": ("seq", "limit"),
}


def validate_records(records: list) -> None:
    """Structural validation before a runtime is created.

    Enforces a strictly increasing record sequence (so an out-of-order swap
    fails), registrations first and unique, and every delivery referencing an
    actor created by an earlier register record.
    """
    registered: set = set()
    saw_non_register = False
    prev_seq = -1
    for index, rec in enumerate(records):
        kind = rec.get("t")
        fields = REQUIRED_FIELDS.get(kind)
        if fields is None:
            raise CorruptRecordError(f"unknown record type: {kind!r}")
        for field in fields:
            if field not in rec:
                raise CorruptRecordError(
                    f"record {index} ({kind!r}) missing field {field!r}"
                )
        seq = rec["seq"]
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= prev_seq:
            raise CorruptRecordError(
                f"record {index} has out-of-order or invalid seq {seq!r}"
            )
        prev_seq = seq
        if kind == "register":
            if saw_non_register:
                raise CorruptRecordError(
                    f"register record {index} follows non-register records"
                )
            if rec["name"] in registered:
                raise CorruptRecordError(
                    f"duplicate registration for {rec['name']!r}"
                )
            registered.add(rec["name"])
        elif kind in ("send", "schedule") and rec["target"] not in registered:
            raise CorruptRecordError(
                f"record {index} references unregistered actor "
                f"{rec['target']!r}"
            )


# ---------------------------------------------------------------------------
# Driver: executes validated records while capturing the public transcript
# ---------------------------------------------------------------------------


def drive_records(records: list, *, known_actors=None, validate: bool = True):
    """Interpret records on a fresh runtime, returning ``(rt, transcript)``.

    Replay uses this same driver: it performs only the actions present in the
    journal, so it never receives an external message the first run did not
    record, and moves time only via the recorded logical ``advance`` calls --
    there is no wall-clock wait anywhere in the runtime or this driver.

    With ``validate=False`` the structural layer is skipped (used to assert
    the runtime's own public :class:`LookupError` for an unknown target);
    ``known_actors`` then names the actors to register up front.
    """
    if validate:
        validate_records(records)
        names = [r["name"] for r in records if r["t"] == "register"]
    else:
        names = list(known_actors or ())
    rt = ActorRuntime()
    for name in names:
        rt.register(name, initial_state_for(name), HANDLERS[name])

    transcript: list = []
    done_seen = 0
    for rec in records:
        kind = rec["t"]
        if kind == "register":
            continue
        if kind == "send":
            message_id = rt.send(
                rec["target"], rec["message"], priority=int(rec["priority"])
            )
            transcript.append(
                ("accept", rec["target"], int(rec["priority"]),
                 message_id, rt.clock())
            )
        elif kind == "schedule":
            ttl = rec["ttl"]
            message_id = rt.schedule(
                rec["target"], rec["message"], delay=int(rec["delay"]),
                ttl=None if ttl is None else int(ttl),
                priority=int(rec["priority"]),
            )
            transcript.append(
                ("accept", rec["target"], int(rec["priority"]),
                 message_id, rt.clock())
            )
        elif kind == "advance":
            result = rt.advance(int(rec["ticks"]))
            transcript.append(
                ("clock", result.time, tuple(result.released),
                 tuple(result.expired))
            )
        elif kind == "run":
            limit = rec["limit"]
            rt.run(limit=None if limit is None else int(limit))
            events = trace_events(rt)
            transcript.extend(events[done_seen:])
            done_seen = len(events)
    return rt, transcript


def execute_script(actions: list):
    """First execution: encode -> parse -> drive; return runtime + artifacts."""
    journal = encode_script(actions)
    records = read_records(journal)
    rt, transcript = drive_records(records)
    return rt, transcript, final_snapshot(rt), journal, records


def records_to_actions(records: list) -> list:
    """Inverse of action_to_record (drops seq) for canonical re-encoding."""
    actions = []
    for rec in records:
        kind = rec["t"]
        if kind == "register":
            actions.append(("register", rec["name"]))
        elif kind == "send":
            actions.append(("send", rec["target"], rec["message"],
                            rec["priority"]))
        elif kind == "schedule":
            actions.append(("schedule", rec["target"], rec["message"],
                            rec["delay"], rec["ttl"], rec["priority"]))
        elif kind == "advance":
            actions.append(("advance", rec["ticks"]))
        elif kind == "run":
            actions.append(("run", rec["limit"]))
    return actions


def replay_journal_bytes(blob: bytes):
    """Replay a journal onto a brand-new runtime; return its final snapshot."""
    records = read_records(blob)
    rt, transcript = drive_records(records)
    return rt, transcript, final_snapshot(rt)


# ---------------------------------------------------------------------------
# The fixed core scenario
# ---------------------------------------------------------------------------
#
# One script with two interacting actors, containing every required element:
#   * normal- and high-priority messages sharing a mailbox -- a high-priority
#     delivery accepted while low-priority mail is still queued overtakes the
#     not-yet-started low messages, but cannot interrupt a handler (handlers
#     are synchronous; the stepwise run(limit=1) assertions prove the high
#     message can only jump mail that has not started);
#   * a message whose handling produces a further delivery ("fanout" -> the
#     peer via ctx.send), enqueued only after that handling commits, at the
#     derived default priority 0;
#   * several timed messages with the same deadline released by one advance
#     in the runtime's stable order (priority, then id), including one that
#     expires at that very tick (expiry wins; it never enters a mailbox);
#   * both delivery modes: synchronous stepwise processing (a result is
#     publicly visible only after run(limit=1) returns) and plain asynchronous
#     acceptance (send/schedule returning an id means "accepted", never
#     "handled").
#
# All timing is logical (advance); ids come from the single fixed monotonic
# source driven by one fixed action list.  No wall-clock, randomness, thread
# or machine-speed assumption occurs.


def inc(n: int) -> dict:
    return {"kind": "inc", "n": n}


def core_scenario() -> list:
    return [
        ("register", WORKER),
        ("register", PEER),

        # Pre-queued normal-priority work before any high-priority mail.
        ("send", WORKER, {"kind": "record", "event": "low-1"}, 0),       # id 1
        ("send", WORKER, {"kind": "record", "event": "low-2"}, 0),       # id 2

        # Synchronous step: one message completes, visible only afterwards.
        ("run", 1),                                                      # low-1

        # High-priority mail accepted while low-2 has not started: it jumps
        # ahead of low-2 but cannot touch the already-committed low-1 step.
        ("send", WORKER, {"kind": "record", "event": "high"}, 10),       # id 3
        ("run", 1),                                                      # high

        # "fanout" commits, then its derived inc(7) is enqueued to the peer
        # at derived priority 0; the peer already holds an external inc(100).
        ("send", PEER, inc(100), 0),                                     # id 4
        ("send", WORKER, {
            "kind": "record", "event": "fanout",
            "then": {"to": PEER, "msg": inc(7)},
        }, 0),                                                           # id 5
        ("run", None),                          # drains; derived gets id 6

        # Several timed messages sharing deadline tick 4: two worker timers
        # (priority 8 then 0), one peer timer, and one worker delivery that
        # expires at that same tick (ttl == delay) and never reaches a handler.
        ("schedule", WORKER, {"kind": "timer", "token": "t-lo"},
         4, None, 0),                                                    # id 7
        ("schedule", WORKER, {"kind": "timer", "token": "t-hi"},
         4, None, 8),                                                    # id 8
        ("schedule", PEER, inc(40), 4, None, 0),                         # id 9
        ("schedule", WORKER, {"kind": "timer", "token": "t-stale"},
         4, 4, 0),                                                       # id 10
        ("advance", 4),
        ("run", None),
    ]


# Concrete expected completion order on the worker (high overtakes only
# queued, not-started mail; derived/follow-up work and the same-deadline
# timers follow priority then stable id).
EXPECTED_WORKER_ORDER = [
    "low-1",
    "high",
    "low-2",
    "fanout",
    {"timer": "t-hi"},
    {"timer": "t-lo"},
]

# Fixed id facts of this exact script.
DERIVED_ID = 6
TIMER_IDS = {"t-lo": 7, "t-hi": 8, "t-peer": 9, "t-stale": 10}


class DeterministicReplayRegressionTests(unittest.TestCase):
    """Same inputs/config -> identical transcript, states and journal replay."""

    def _drive_core(self):
        return execute_script(core_scenario())

    # -- concrete core-scenario ordering -----------------------------------

    def test_core_scenario_has_expected_shape(self):
        _, transcript, snapshot, journal, records = self._drive_core()

        # Both interacting actors receive traffic.
        targets = {r.get("target") for r in records
                   if r.get("t") in ("send", "schedule")}
        self.assertEqual(targets, {WORKER, PEER})

        # Worker completion order: the high-priority message overtakes only
        # not-yet-started low mail; the same-deadline timers keep priority
        # then stable id order.
        self.assertEqual(
            list(snapshot["states"][WORKER]["value"]),
            EXPECTED_WORKER_ORDER,
        )

        # The high-priority id completes before the earlier-accepted low-2 id.
        done_ids = [item[1] for item in transcript if item[0] == "done"]
        self.assertLess(done_ids.index(3), done_ids.index(2))

        # Derived inc(7) is produced while fanout is handled and applied to
        # the peer alongside its pre-queued inc(100) and the peer timer 40.
        self.assertEqual(snapshot["states"][PEER]["value"], 147)
        # The derived delivery took a fresh id right after its trigger.
        self.assertIn(("done", DERIVED_ID, PEER, 0),
                      [item[:4] for item in transcript if item[0] == "done"])

        # Exactly one same-tick expiry, never handled; three stable releases.
        clock_events = [item for item in transcript if item[0] == "clock"]
        self.assertEqual(len(clock_events), 1)
        _, time4, released, expired = clock_events[0]
        self.assertEqual((time4, released, expired),
                         (4, (7, 8, 9), (10,)))
        self.assertNotIn(
            {"timer": "t-stale"}, snapshot["states"][WORKER]["value"]
        )

        # Settled: no pending mail, no outstanding timers, clock at tick 4.
        self.assertEqual(snapshot["pending"], {WORKER: 0, PEER: 0})
        self.assertEqual(snapshot["scheduled"], {WORKER: 0, PEER: 0})
        self.assertEqual(snapshot["clock"], 4)

        self.assertTrue(journal)

    # -- repeatability ------------------------------------------------------

    def test_repeated_execution_is_itemwise_identical(self):
        # Same initial state, same step (concurrency) configuration, same
        # message sequence: run it three times on independent runtimes.
        _, first_tx, first_snap, _, _ = self._drive_core()
        for rep in range(2):
            _, transcript, snapshot, _, _ = self._drive_core()
            assert_transcripts_equal(
                self, first_tx, transcript,
                f"repeat {rep + 1} vs first execution",
            )
            assert_snapshots_equal(
                self, first_snap, snapshot,
                f"repeat {rep + 1} vs first execution",
            )

    # -- replay through the existing entry point ---------------------------

    def test_journal_replay_reproduces_states_and_events(self):
        _, first_tx, first_snap, journal, records = self._drive_core()

        # Replay into a fresh runtime from durable bytes alone.
        replay_rt, replay_tx, replay_snap = replay_journal_bytes(journal)

        # Same runtime-wide ids, completions, clock settlements and states.
        assert_transcripts_equal(
            self, first_tx, replay_tx, "journal replay vs first execution"
        )
        assert_snapshots_equal(
            self, first_snap, replay_snap, "journal replay vs first execution"
        )

        # The replay stream is canonical: parse -> encode is byte-identical.
        self.assertEqual(
            encode_script(records_to_actions(read_records(journal))), journal
        )

        # Replay accepts no external input beyond the journal: the recorded
        # run actions exhaust exactly all mail in the fresh runtime.
        self.assertEqual(replay_rt.pending_count(WORKER), 0)
        self.assertEqual(replay_rt.pending_count(PEER), 0)
        self.assertEqual(replay_rt.scheduled_count(WORKER), 0)
        self.assertEqual(replay_rt.scheduled_count(PEER), 0)

    def test_replay_uses_only_logical_time_never_real_waits(self):
        _, _, _, journal, records = self._drive_core()
        # Time moves solely through recorded advance records (no sleep exists
        # in the API), and replay ends at the recorded logical tick.
        self.assertTrue(any(r["t"] == "advance" for r in records))
        _, _, replay_snap = replay_journal_bytes(journal)
        self.assertEqual(replay_snap["clock"], 4)

    # -- synchronous vs asynchronous delivery semantics ---------------------

    def test_synchronous_result_visible_only_after_processing(self):
        rt = ActorRuntime()
        register_workers(rt)
        rt.send(WORKER, {"kind": "record", "event": "one"}, 0)
        # Accepted but not processed: nothing publicly visible yet.
        self.assertEqual(rt.get_state(WORKER)["value"], [])
        self.assertEqual(rt.trace(), [])
        done = rt.run(limit=1)
        self.assertEqual(done, 1)
        # The synchronous result is visible only after the call returns.
        self.assertEqual(rt.get_state(WORKER)["value"], ["one"])
        self.assertEqual(len(rt.trace()), 1)

    def test_asynchronous_acceptance_is_not_processing(self):
        rt = ActorRuntime()
        register_workers(rt)
        low_id = rt.send(WORKER, {"kind": "record", "event": "lo"}, 0)
        high_id = rt.send(WORKER, {"kind": "record", "event": "hi"}, 10)
        timer_id = rt.schedule(PEER, inc(5), delay=1, priority=0)
        # Accepted ids do not mean handled: no trace, no state, timer held
        # outside the mailbox.
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.get_state(WORKER)["value"], [])
        self.assertEqual(rt.pending_count(WORKER), 2)
        self.assertEqual(rt.scheduled_count(PEER), 1)
        # Releasing the timer still processes nothing.
        self.assertEqual(rt.advance(1).released, (timer_id,))
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.get_state(PEER)["value"], 0)
        # Only an explicit run commits; the high-priority message overtakes
        # the earlier accepted, not-yet-started low one.
        rt.run()
        self.assertEqual(rt.get_state(WORKER)["value"], ["hi", "lo"])
        self.assertEqual([e.message_id for e in rt.trace() if e.actor_name == WORKER],
                         [high_id, low_id])
        self.assertLess(low_id, high_id)

    def test_priority_cannot_interrupt_a_running_handler(self):
        # No preemption: a handler enqueues further mail "during" its own
        # execution via the context; that mail is picked up strictly after the
        # handler commits, never interleaved with it.
        seen: list = []

        def leader(state, message, ctx: ActorContext):
            seen.append(("start", message))
            ctx.send(WORKER, {"kind": "record", "event": "during"})
            seen.append(("end", message))
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("ord", [], leader)
        rt.register(WORKER, initial_state_for(WORKER), HANDLERS[WORKER])
        rt.send("ord", "leader")
        rt.run()
        self.assertEqual(seen, [("start", "leader"), ("end", "leader")])
        self.assertEqual(rt.get_state(WORKER)["value"], ["during"])
        self.assertEqual(
            [e.actor_name for e in rt.trace()], ["ord", WORKER]
        )

    # -- malformed replay: documented errors and failure atomicity ----------

    def test_truncated_journal_fails_with_documented_error(self):
        _, _, _, journal, _ = self._drive_core()
        with self.assertRaises(CorruptRecordError):
            replay_journal_bytes(journal[:-3])

    def test_swapped_event_order_fails_with_documented_error(self):
        _, _, _, _, records = self._drive_core()
        # Swap two adjacent send frames but keep their embedded seq values:
        # the stream now arrives out of recorded order and must be rejected.
        send_indexes = [i for i, r in enumerate(records) if r["t"] == "send"]
        tampered = list(records)
        i, j = send_indexes[0], send_indexes[1]
        tampered[i], tampered[j] = tampered[j], tampered[i]
        with self.assertRaises(CorruptRecordError):
            replay_journal_bytes(encode_records_raw(tampered))

    def test_unknown_actor_through_structural_layer_is_corrupt_record(self):
        # The replay persistence layer rejects a record referencing an actor
        # that was never registered, before constructing any runtime.
        records = [
            {"t": "register", "seq": 0, "name": WORKER},
            {"t": "send", "seq": 1, "target": "ghost",
             "message": "m", "priority": 0},
        ]
        with self.assertRaises(CorruptRecordError):
            drive_records(records, validate=True)

    def test_unknown_actor_at_runtime_boundary_is_lookup_error(self):
        # Bypassing the structural layer, the runtime's own documented public
        # contract for an unknown target is LookupError -- asserted as that
        # exact type, never as a bare Exception.
        records = [
            {"t": "send", "seq": 0, "target": "ghost",
             "message": {"kind": "record", "event": "x"}, "priority": 0},
        ]
        with self.assertRaises(LookupError):
            drive_records(records, known_actors=(), validate=False)

    def test_failed_replay_leaves_no_partial_queryable_state(self):
        _, _, _, journal, _ = self._drive_core()

        # A truncated replay raises and hands back no runtime/snapshot: the
        # caller cannot continue querying a half-recovered runtime.
        outcome = ("untouched",)
        with self.assertRaises(CorruptRecordError):
            outcome = replay_journal_bytes(journal[:-3])
        self.assertEqual(
            outcome, ("untouched",),
            "a failed replay must not return a queryable result",
        )

        # An unknown-target replay likewise returns nothing.
        outcome = ("untouched",)
        records = [
            {"t": "send", "seq": 0, "target": "ghost",
             "message": {"kind": "record", "event": "x"}, "priority": 0},
        ]
        with self.assertRaises(LookupError):
            outcome = drive_records(records, known_actors=(), validate=False)
        self.assertEqual(outcome, ("untouched",))

        # The failed attempts leave no residue: an intact journal still
        # replays to a fully settled snapshot on a fresh runtime.
        _, _, snapshot = replay_journal_bytes(journal)
        self.assertEqual(snapshot["pending"], {WORKER: 0, PEER: 0})
        self.assertEqual(snapshot["scheduled"], {WORKER: 0, PEER: 0})

    # -- first-mismatch diagnostics ----------------------------------------

    def test_comparison_names_the_first_diverging_event(self):
        base = [
            ("done", 1, WORKER, 0, [], ["a"]),
            ("done", 2, WORKER, 0, ["a"], ["a", "b"]),
            ("done", 3, PEER, 0, 0, 1),
        ]
        drifted = [
            ("done", 1, WORKER, 0, [], ["a"]),
            ("done", 2, PEER, 0, 0, 1),        # actor swapped at index 1
            ("done", 3, WORKER, 0, ["a"], ["a", "b"]),
        ]
        with self.assertRaises(AssertionError) as caught:
            assert_transcripts_equal(self, base, drifted, "drift-check")
        self.assertIn("event index 1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
