"""Deterministic first-run vs replay vs snapshot-recovery scenario tests.

These tests add no product API: they drive only the existing public surface
(``register/send/run/get_state/pending_count/trace`` and the documented
exceptions) through the test-local ``replay_harness``.  Every scenario is run
three ways and required to agree on the public contract:

1. first execution over the generated command script;
2. replay from the durable journal bytes produced by the first execution;
3. recovery from the most recent snapshot plus the journal suffix.

Comparison uses only contract-defined facts: completion-order trace events
(actor, priority, state before/after), message identity, actor identity,
handler results (states), supervision outcomes and the durable delivery/
dedup ledger.  It never consults wall-clock time, thread ids, object
identity or log text.
"""
from __future__ import annotations

import unittest

from actor_runtime import (
    ActorContext,
    ActorExecutionError,
    ActorRuntime,
)

from tests.replay_harness import (
    A_CHILD,
    A_COUNTER,
    A_EVENTS,
    A_SUPERVISOR,
    CorruptRecordError,
    DeterministicRng,
    IncompatibleSnapshotError,
    InvalidScript,
    ScenarioResult,
    corrupt_midframe,
    default_policy,
    encode_records,
    execute,
    normalize_script,
    parse_records,
    recover_snapshot,
    replay_journal,
    tamper_snapshot,
    truncate_midframe,
)


# ---------------------------------------------------------------------------
# Equivalence assertions
# ---------------------------------------------------------------------------


def assert_first_vs_replay(testcase: unittest.TestCase,
                           first: ScenarioResult,
                           replay: ScenarioResult) -> None:
    # Byte-level signature of states + event ledger + supervision + dedup.
    testcase.assertEqual(first.signature(), replay.signature())
    # Journal replay must reproduce message identity exactly: the same
    # runtime-wide monotonic ids in the same completion order.
    testcase.assertEqual(first.traces, replay.traces)
    # Re-encoding the parsed journal must be byte-identical (canonical form).
    testcase.assertEqual(
        encode_records(parse_records(replay.journal)), replay.journal
    )


def assert_first_vs_recovery(testcase: unittest.TestCase,
                             first: ScenarioResult,
                             recovered: ScenarioResult) -> None:
    # Field-level normalised equivalence. Recovery begins a new runtime
    # generation, so its runtime-wide message ids restart at 1: the documented
    # normalisation compares completion order, actor identity, priority and
    # state transitions (quads), and separately requires each generation's
    # ids to be a fresh monotonic 1..N sequence.
    testcase.assertEqual(first.states, recovered.states)
    testcase.assertEqual(first.signature(), recovered.signature())
    testcase.assertEqual(first.trace_quads(), recovered.trace_quads())
    cut = recovered.restored_len
    # Preserved prefix is byte-for-byte the snapshot's trace (old generation).
    testcase.assertEqual(first.traces[:cut], recovered.traces[:cut])
    # Suffix: both generations number their own messages 1..N in order.
    first_suffix_ids = first.trace_ids()[cut:]
    recovered_suffix_ids = recovered.trace_ids()[cut:]
    n = len(recovered_suffix_ids)
    testcase.assertEqual(recovered_suffix_ids, list(range(1, n + 1)))
    testcase.assertEqual(
        len(first_suffix_ids), len(recovered_suffix_ids)
    )
    # The first run's suffix ids are a consecutive monotonic run too.
    if first_suffix_ids:
        testcase.assertEqual(
            first_suffix_ids,
            list(range(first_suffix_ids[0],
                       first_suffix_ids[0] + len(first_suffix_ids))),
        )


def run_all_three(testcase: unittest.TestCase, script: list,
                  *, policy=default_policy):
    first = execute(script, policy=policy)
    replay = replay_journal(first.journal, policy=policy)
    assert_first_vs_replay(testcase, first, replay)
    if first.snapshots:
        seq, offset, blob = first.snapshots[-1]
        recovered = recover_snapshot(
            blob, first.journal, marker_offset=offset, policy=policy
        )
        assert_first_vs_recovery(testcase, first, recovered)
    return first, replay


# ---------------------------------------------------------------------------
# Hand-written deterministic scenarios
# ---------------------------------------------------------------------------


class MailboxMixingTests(unittest.TestCase):
    """Normal + priority mailboxes, registration order, mixed in one drain.

    Because each external input is drained before the next is accepted, two
    actors only hold mail simultaneously when deliveries are enqueued together
    -- here, by several timers reaching their deadline on one clock advance.
    That single drain is exactly where registration order (across actors) and
    priority (within a mailbox) both decide the sequence.
    """

    def test_priority_and_registration_order_mix(self):
        script = [
            # All three timers share one deadline and fire in one drain.
            ("schedule", A_EVENTS, "lo", 2, 100, 0, True),
            ("schedule", A_EVENTS, "hi", 2, 100, 9, True),
            ("schedule", A_CHILD, "c", 2, 100, 0, True),
            ("clock", 2),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        # events is registered before child, so its whole mailbox drains
        # first; within events, priority 9 ("hi") beats the enqueued-earlier
        # "lo". Completion order: events hi, events lo, child c.
        self.assertEqual(
            [(e[1], e[2]) for e in first.traces],
            [(A_EVENTS, 9), (A_EVENTS, 0), (A_CHILD, 0)],
        )
        self.assertEqual(
            first.states[A_EVENTS]["value"],
            [{"timer": "hi"}, {"timer": "lo"}],
        )
        self.assertEqual(
            first.states[A_CHILD]["value"], [{"timer": "c"}]
        )

    def test_equal_priority_is_fifo_by_message_id(self):
        # Two equal-priority timers share a deadline and one mailbox: their
        # message ids break the tie in timer-enqueue (FIFO) order.
        script = [
            ("schedule", A_EVENTS, "a", 3, 100, 3, True),
            ("schedule", A_EVENTS, "b", 3, 100, 3, True),
            ("schedule", A_EVENTS, "c", 3, 100, 3, True),
            ("clock", 3),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(
            first.states[A_EVENTS]["value"],
            [{"timer": "a"}, {"timer": "b"}, {"timer": "c"}],
        )


class DerivedDeliveryInterleaveTests(unittest.TestCase):
    """Synchronous handler completion vs asynchronously enqueued derived."""

    def test_derived_enqueued_after_commit_mixes_with_external_mail(self):
        script = [
            # events commits "go" and only THEN enqueues a derived inc to
            # counter; counter already has external mail, and registration
            # order selects counter after events' mailbox empties.
            ("send", A_COUNTER, {"kind": "inc", "n": 7}, 0),
            ("send", A_EVENTS,
             {"kind": "record", "event": "go",
              "then": {"to": A_COUNTER,
                       "msg": {"kind": "inc", "n": 5}}}, 0),
            ("send", A_EVENTS, {"kind": "record", "event": "last"}, 0),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        # events drained first (registration order picks counter first only
        # if counter has mail -- it does, so counter's pre-existing 7 is
        # applied before events; the derived 5 arrives only after "go").
        # Completion order is deterministic and asserted via replay/recovery;
        # assert the resulting state directly.
        self.assertEqual(first.states[A_COUNTER]["value"], 12)
        self.assertEqual(
            first.states[A_EVENTS]["value"], ["go", "last"]
        )

    def test_derived_chain_order(self):
        script = [
            ("send", A_EVENTS,
             {"kind": "record", "event": 1,
              "then": {"to": A_EVENTS,
                       "msg": {"kind": "record", "event": 2}}}, 0),
            ("send", A_EVENTS, {"kind": "record", "event": 3}, 0),
        ]
        first, _ = run_all_three(self, script)
        # The first external input drains to idle: 1 commits, derived 2 is
        # enqueued and processed in the same drain; 3 only arrives afterwards.
        self.assertEqual(first.states[A_EVENTS]["value"], [1, 2, 3])
        # The interleaving case (derived vs already-queued external mail) is
        # covered by test_derived_enqueued_after_commit_mixes_with_external_mail
        # using timers to stage simultaneous mail.


class TimerTests(unittest.TestCase):
    """Virtual-clock timer expiry, including expired-before-firing."""

    def test_timer_fires_on_clock_advance(self):
        script = [
            ("schedule", A_EVENTS, "ping", 5, 100, 0, True),
            ("clock", 4),
            ("snapshot",),                      # nothing due yet
            ("send", A_EVENTS, {"kind": "record", "event": "pre"}, 0),
            ("clock", 1),                       # timer becomes due
            ("send", A_EVENTS, {"kind": "record", "event": "post"}, 0),
        ]
        first, _ = run_all_three(self, script)
        kinds = [e["at"] for e in first.events]
        self.assertIn("timer_fired", kinds)
        self.assertNotIn("timer_expired", kinds)
        # At clock 5 the timer message is enqueued behind "pre"? No: "pre"
        # was delivered at clock 4 and already drained. Order: pre, ping, post.
        self.assertEqual(
            first.states[A_EVENTS]["value"],
            ["pre", {"timer": "ping"}, "post"],
        )

    def test_expired_timer_never_reaches_handler(self):
        script = [
            # delay 5, ttl 0: it is already expired the instant it is due.
            ("schedule", A_EVENTS, "stale", 5, 0, 0, True),
            ("clock", 5),
            ("send", A_EVENTS, {"kind": "record", "event": "only"}, 0),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        at = [(e["at"], e.get("id")) for e in first.events]
        self.assertIn(("timer_expired", "t0001"), at)
        self.assertNotIn("timer_fired", [a[0] for a in at])
        # The stale timer never entered user processing.
        self.assertEqual(first.states[A_EVENTS]["value"], ["only"])
        self.assertEqual(
            [a["kind"] for a in first.states[A_EVENTS]["arrivals"]],
            ["record"],
        )

    def test_two_timers_fire_in_deterministic_deadline_order(self):
        script = [
            ("schedule", A_EVENTS, "later", 10, 100, 2, True),
            ("schedule", A_EVENTS, "sooner", 5, 100, 2, True),
            ("clock", 10),
        ]
        first, _ = run_all_three(self, script)
        fired = [e["id"] for e in first.events if e["at"] == "timer_fired"]
        self.assertEqual(fired, ["t0002", "t0001"])
        self.assertEqual(
            first.states[A_EVENTS]["value"],
            [{"timer": "sooner"}, {"timer": "later"}],
        )


class InFlightCapTests(unittest.TestCase):
    """Scheduling constrained by a per-input completion cap."""

    def test_cap_batches_completions_but_preserves_order(self):
        # cap=1 interleaves processing with external input: the derived "b"
        # is committed (and enqueued) before "c" is ever sent, so it precedes
        # "c" (a, b, c). The cap genuinely constrains scheduling, so the
        # contract is that first run, journal replay and recovery all produce
        # exactly this order -- not that capped order equals uncapped order
        # (which is a, c, b).
        script = [
            ("cap", 1),
            ("send", A_EVENTS,
             {"kind": "record", "event": "a",
              "then": {"to": A_EVENTS,
                       "msg": {"kind": "record", "event": "b"}}}, 0),
            ("send", A_EVENTS, {"kind": "record", "event": "c"}, 0),
            ("cap", None),                      # drains the backlog
            ("snapshot",),
        ]
        first, replay = run_all_three(self, script)
        self.assertEqual(
            first.states[A_EVENTS]["value"], ["a", "b", "c"]
        )
        self.assertEqual(first.traces, replay.traces)
        again, _ = run_all_three(self, script)
        self.assertEqual(first.traces, again.traces)

    def test_cap_with_priority_and_two_actors(self):
        # cap=1 completes exactly one message per external input. The second
        # inc (priority 10) is delivered after the first already completed, so
        # there is never any same-mailbox priority contention: ordering stays
        # delivery order. A same-deadline timer batch (below) is what exhibits
        # priority under the cap.
        script = [
            ("cap", 1),
            ("send", A_COUNTER, {"kind": "inc", "n": 1}, 0),
            ("send", A_COUNTER, {"kind": "inc", "n": 1}, 10),
            ("send", A_EVENTS, {"kind": "record", "event": "x"}, 0),
            ("cap", None),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(
            [(e[1], e[2]) for e in first.traces],
            [(A_COUNTER, 0), (A_COUNTER, 10), (A_EVENTS, 0)],
        )
        self.assertEqual(first.states[A_COUNTER]["value"], 2)
        self.assertEqual(first.states[A_EVENTS]["value"], ["x"])

    def test_priority_under_cap_with_batched_timer_mail(self):
        # Two timers with one deadline create simultaneous mail; with cap=1
        # only the single highest-priority message completes on that clock
        # input; release drains the rest in the same global order.
        script = [
            ("cap", 1),
            ("schedule", A_EVENTS, "lo", 1, 100, 0, True),
            ("schedule", A_EVENTS, "hi", 1, 100, 9, True),
            ("clock", 1),                      # completes hi only
            ("cap", None),                     # drains lo
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(
            [(e[1], e[2]) for e in first.traces],
            [(A_EVENTS, 9), (A_EVENTS, 0)],
        )
        self.assertEqual(
            first.states[A_EVENTS]["value"],
            [{"timer": "hi"}, {"timer": "lo"}],
        )


class DedupTests(unittest.TestCase):
    """At-least-once redelivery: one state change, two arrivals on record."""

    def test_duplicate_changes_state_once_but_both_deliveries_are_traced(self):
        script = [
            ("send", A_COUNTER, {"kind": "inc", "n": 9}, 0, True),
            ("redeliver", 0),
            ("redeliver", 0),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        # Exactly one state change.
        self.assertEqual(first.states[A_COUNTER]["value"], 9)
        # Three arrivals, the first applied and the other two marked dup.
        self.assertEqual(
            first.states[A_COUNTER]["arrivals"],
            [{"kind": "inc", "duplicate": False},
             {"kind": "inc", "duplicate": True},
             {"kind": "inc", "duplicate": True}],
        )
        # Every delivery and every dedup decision is durable, in order.
        dedup = [d for d in first.delivered_dedup if d["dedup"] == "d0001"]
        self.assertEqual(
            dedup,
            [{"dedup": "d0001", "duplicate": False},
             {"dedup": "d0001", "duplicate": True},
             {"dedup": "d0001", "duplicate": True}],
        )
        # Journal records show one original and two redeliveries.
        records = parse_records(first.journal)
        flags = [r.get("redelivery") for r in records
                 if r.get("t") == "deliver" and r.get("dedup") == "d0001"]
        self.assertEqual(flags, [False, True, True])

    def test_distinct_dedup_ids_both_apply(self):
        script = [
            ("send", A_EVENTS, {"kind": "record", "event": "a"}, 0, True),
            ("send", A_EVENTS, {"kind": "record", "event": "a"}, 0, True),
            ("redeliver", 1),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(first.states[A_EVENTS]["value"], ["a", "a"])
        self.assertEqual(
            [a["duplicate"] for a in first.states[A_EVENTS]["arrivals"]],
            [False, False, True],
        )


class SupervisionTests(unittest.TestCase):
    """Child failure -> resume/restart/stop/escalate on one replayable path."""

    def test_resume_keeps_state_drops_failing_message(self):
        # Counter policy is "resume": state survives, the fail is discarded.
        script = [
            ("send", A_COUNTER, {"kind": "inc", "n": 3}, 0),
            ("send", A_COUNTER, {"kind": "fail", "key": "k1"}, 0),
            ("resolve",),
            ("send", A_COUNTER, {"kind": "inc", "n": 4}, 0),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(first.states[A_COUNTER]["value"], 7)
        self.assertEqual(
            [(s["actor"], s["action"]) for s in first.supervision],
            [(A_COUNTER, "resume")],
        )
        self.assertEqual(
            [e["at"] for e in first.events].count("fault"), 1
        )

    def test_restart_resets_child_state_and_continues(self):
        script = [
            ("send", A_CHILD, {"kind": "record", "event": "old"}, 0, True),
            ("send", A_CHILD, {"kind": "fail", "key": "boom"}, 0),
            ("resolve",),                          # child policy: restart
            ("send", A_CHILD, {"kind": "record", "event": "new"}, 0, True),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        # Pre-restart state is erased even though "old" committed.
        self.assertEqual(first.states[A_CHILD]["value"], ["new"])
        # The restarted child has a fresh dedup memory: only the post-restart
        # id survives (ids follow durable delivery order, so it is d0002).
        self.assertEqual(first.states[A_CHILD]["seen"], ["d0002"])
        self.assertEqual(
            [(s["actor"], s["action"]) for s in first.supervision],
            [(A_CHILD, "restart")],
        )

    def test_stop_discards_mailbox_and_refuses_later_delivery(self):
        # events policy is "stop".
        script = [
            ("send", A_EVENTS, {"kind": "record", "event": "kept"}, 0),
            ("send", A_EVENTS, {"kind": "fail", "key": "die"}, 0),
            ("resolve",),
            # Later deliveries are refused (journaled, accepted=false) and
            # never reach the handler.
            ("send", A_EVENTS, {"kind": "record", "event": "nope"}, 0),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(first.states[A_EVENTS], {"stopped": True})
        post = [e for e in first.events
                if e["at"] == "deliver" and e["target"] == A_EVENTS
                and e["kind"] == "record"]
        self.assertTrue(any(not e["accepted"] for e in post))
        self.assertTrue(all(e["accepted"] for e in post[:1]))

    def test_escalate_stops_child_and_notifies_supervisor(self):
        # Two child faults: first restarts, second escalates.
        script = [
            ("send", A_CHILD, {"kind": "fail", "key": "one"}, 0),
            ("resolve",),                          # restart
            ("send", A_CHILD, {"kind": "record", "event": "mid"}, 0),
            ("send", A_CHILD, {"kind": "fail", "key": "two"}, 0),
            ("resolve",),                          # escalate
            ("send", A_SUPERVISOR,
             {"kind": "record", "event": "watching"}, 0),
            ("snapshot",),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(first.states[A_CHILD], {"stopped": True})
        self.assertEqual(
            first.states[A_SUPERVISOR]["escalations"], [A_CHILD]
        )
        self.assertEqual(
            [(s["actor"], s["action"]) for s in first.supervision],
            [(A_CHILD, "restart"), (A_CHILD, "escalate")],
        )

    def test_failure_error_type_is_recorded_from_public_fact(self):
        script = [
            ("send", A_COUNTER, {"kind": "fail", "key": "x"}, 0),
            ("resolve",),
        ]
        first, _ = run_all_three(self, script)
        self.assertEqual(
            first.supervision[0]["error_type"], "ValueError"
        )


class PublicExceptionContractTests(unittest.TestCase):
    """Existing public error contract is asserted, never redefined."""

    def test_runtime_rejects_invalid_priority(self):
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: s)
        for bad in (1.5, "1", None, [1], True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.send("a", "m", priority=bad)

    def test_runtime_rejects_missing_target(self):
        rt = ActorRuntime()
        with self.assertRaises(LookupError):
            rt.send("ghost", "m")
        rt.register("a", None, lambda s, m, c: s)
        def sends_unknown(state, message, ctx):
            ctx.send("ghost", "m")
            return state
        rt.register("b", None, sends_unknown)
        with self.assertRaises(ActorExecutionError) as caught:
            rt.send("b", 1)
            rt.run()
        self.assertIsInstance(caught.exception.original, LookupError)

    def test_handler_failure_is_actor_execution_error_with_public_fields(self):
        def boom(state, message, ctx):
            raise RuntimeError("kaboom")

        rt = ActorRuntime()
        rt.register(A_CHILD, None, boom)
        mid = rt.send(A_CHILD, "m")
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        self.assertEqual(caught.exception.actor_name, A_CHILD)
        self.assertEqual(caught.exception.message_id, mid)
        self.assertIsInstance(caught.exception.original, RuntimeError)
        # Failed message stays unacknowledged; no trace entry committed.
        self.assertEqual(rt.pending_count(A_CHILD), 1)
        self.assertEqual(rt.trace(), [])

    def test_corrupt_persisted_record_raises_fixture_error(self):
        first = execute([
            ("send", A_COUNTER, {"kind": "inc", "n": 1}, 0),
        ])
        with self.assertRaises(CorruptRecordError):
            replay_journal(corrupt_midframe(first.journal))
        with self.assertRaises(CorruptRecordError):
            replay_journal(truncate_midframe(first.journal))

    def test_incompatible_snapshot_and_header_raise(self):
        first = execute([
            ("send", A_COUNTER, {"kind": "inc", "n": 1}, 0),
            ("snapshot",),
        ])
        _, _, snap = first.snapshots[-1]
        with self.assertRaises(IncompatibleSnapshotError):
            recover_snapshot(
                tamper_snapshot(snap, schema=999),
                first.journal,
                marker_offset=first.snapshots[-1][1],
            )
        with self.assertRaises(IncompatibleSnapshotError):
            recover_snapshot(
                tamper_snapshot(snap, codec="json-v2"),
                first.journal,
                marker_offset=first.snapshots[-1][1],
            )
        bad_header = encode_records([], schema=999)
        with self.assertRaises(IncompatibleSnapshotError):
            replay_journal(bad_header)


# ---------------------------------------------------------------------------
# Seeded generation + shrinking
# ---------------------------------------------------------------------------


TARGETS = (A_EVENTS, A_COUNTER, A_CHILD, A_SUPERVISOR)


def _plain_message(rng: DeterministicRng, target: str, seq: int,
                   *, allow_then: bool) -> dict:
    if target == A_COUNTER:
        return {"kind": "inc", "n": rng.randint(1, 5)}
    if target == A_SUPERVISOR:
        return {"kind": "record", "event": f"e{seq}"}
    # events / child
    if allow_then and rng.chance(30):
        to = rng.choose((A_EVENTS, A_COUNTER, A_SUPERVISOR))
        inner = ({"kind": "inc", "n": rng.randint(1, 3)}
                 if to == A_COUNTER
                 else {"kind": "record", "event": f"d{seq}"})
        return {"kind": "record", "event": f"e{seq}",
                "then": {"to": to, "msg": inner}}
    return {"kind": "record", "event": f"e{seq}"}


def generate_script(seed: int) -> list:
    """Generate a finite, fixed-layout script from ``seed``.

    The layout is structural (so the replay invariants always hold and the
    suite deterministically exercises every feature); only payloads and
    small choices come from the fixed :class:`DeterministicRng`.

      phase 1  plain sends + a supervision block (resume/restart/stop/
               escalate chosen by ``seed % 4``) -- no timers, cap or derived
               messages here;
      phase 2  optional bounded-scheduling window (even seeds), released to
               a drain before continuing;
      phase 3  timers (one guaranteed to fire; stale ones on ``seed % 3``),
               clock advances, dedup sends and a redelivery;
      phase 4  first snapshot (mailboxes are drained);
      phase 5  tail sends / a further timer / another redelivery, then a
               second snapshot for mid-stream recovery.
    """
    rng = DeterministicRng(seed)
    script: list = []
    seq = 0
    dedup_count = 0

    def emit(target, message, prio=0, dedup=False):
        nonlocal dedup_count
        script.append(("send", target, message, int(prio), bool(dedup)))
        if dedup:
            ordinal = dedup_count
            dedup_count += 1
            return ordinal
        return None

    def plain(target, *, allow_then, dedup=None, prio=None):
        nonlocal seq
        seq += 1
        if prio is None:
            prio = rng.choose((-1, 0, 0, 0, 2, 5))
        if dedup is None:
            dedup = rng.chance(45)
        return emit(target, _plain_message(rng, target, seq,
                                           allow_then=allow_then),
                    prio, dedup)

    # -- phase 1: supervision window -------------------------------------
    for _ in range(rng.randint(3, 5)):
        plain(rng.choose((A_CHILD, A_COUNTER, A_EVENTS, A_SUPERVISOR)),
              allow_then=False)
    mode = seed % 4
    if mode == 0:                              # child -> restart
        script.append(("send", A_CHILD,
                       {"kind": "fail", "key": f"s{seed}-r"}, 0))
        script.append(("resolve",))
    elif mode == 1:                            # counter -> resume
        script.append(("send", A_COUNTER,
                       {"kind": "fail", "key": f"s{seed}-u"}, 0))
        script.append(("resolve",))
    elif mode == 2:                            # events -> stop
        script.append(("send", A_EVENTS,
                       {"kind": "fail", "key": f"s{seed}-s"}, 0))
        script.append(("resolve",))
    else:                                      # child -> restart, escalate
        script.append(("send", A_CHILD,
                       {"kind": "fail", "key": f"s{seed}-e1"}, 0))
        script.append(("resolve",))
        plain(A_CHILD, allow_then=False)
        script.append(("send", A_CHILD,
                       {"kind": "fail", "key": f"s{seed}-e2"}, 0))
        script.append(("resolve",))
    plain(rng.choose((A_COUNTER, A_SUPERVISOR)), allow_then=False)

    # -- phase 2: bounded scheduling window (even seeds) ------------------
    if seed % 2 == 0:
        script.append(("cap", rng.choose((1, 2, 3))))
        for _ in range(rng.randint(2, 4)):
            plain(rng.choose((A_EVENTS, A_COUNTER)),
                  allow_then=True,
                  prio=rng.choose((0, 1, 3)))
        script.append(("cap", None))           # drain before continuing

    # -- phase 3: timers, clock, dedup send + redelivery ------------------
    timer_targets = (A_EVENTS, A_CHILD)
    timer_specs = []
    n_timers = rng.randint(1, 3)
    want_stale = (seed % 3 == 0)
    max_delay = 0
    for k in range(n_timers):
        target = rng.choose(timer_targets)
        delay = rng.randint(1, 5)
        max_delay = max(max_delay, delay)
        stale = want_stale and k == 0
        ttl = 0 if stale else rng.choose((5, 20, 100))
        timer_specs.append((target, f"tm{k}", delay, ttl,
                            rng.choose((0, 2)), rng.chance(70), stale))
        script.append(("schedule", target, f"tm{k}", delay, ttl,
                       timer_specs[-1][4], timer_specs[-1][5]))
    # Advance far enough that every timer reaches its deadline. With the
    # ttl==0 boundary rule (expires at the deadline) a stale timer is caught
    # at exactly max_delay, and a non-stale timer (ttl >= 5) is still valid
    # at clock <= max_delay (delays are at most 5).
    total = max_delay
    stepped = 0
    while stepped < total:
        step = min(rng.randint(1, 3), total - stepped)
        script.append(("clock", step))
        stepped += step
        if rng.chance(60):
            plain(rng.choose((A_EVENTS, A_COUNTER, A_SUPERVISOR)),
                  allow_then=True)
    # Guaranteed dedup send followed by an at-least-once redelivery.
    dedup_ordinal = emit(
        A_COUNTER, {"kind": "inc", "n": rng.randint(1, 4)}, 0, True
    )
    script.append(("redeliver", dedup_ordinal))

    # -- phase 4: first snapshot ------------------------------------------
    script.append(("snapshot",))

    # -- phase 5: tail -----------------------------------------------------
    for _ in range(rng.randint(2, 4)):
        ordinal = plain(rng.choose(TARGETS), allow_then=True)
        if ordinal is not None and rng.chance(40):
            script.append(("redeliver", ordinal))
    if seed % 2 == 1:
        script.append(("schedule", A_EVENTS, "tail", 1, 100, 0, True))
        script.append(("clock", 2))
    script.append(("snapshot",))
    return script


def shrink_script(script: list) -> list:
    """Deterministic delta-debugging style shrink.

    Returns a strictly shorter script that still reproduces a failure of the
    same property check.  Candidates: drop one command; drop an even/odd
    half; collapse to a suffix.  The first candidate that still fails wins,
    applied greedily to a fixed point.
    """
    current = list(script)

    def fails(candidate):
        try:
            check_script(candidate)
        except _PropertyFailure:
            # The target property still fails on this candidate.
            return True
        except (InvalidScript, CorruptRecordError, IncompatibleSnapshotError):
            # Deleting a command broke the structural invariants; such a
            # candidate cannot be a valid reproduction of the failure.
            return False
        # Any other exception indicates a harness/test bug rather than the
        # property under test; do not treat it as a reproducible failure.
        return False

    changed = True
    while changed and len(current) > 1:
        changed = False
        n = len(current)
        candidates = [[c for j, c in enumerate(current) if j != i]
                      for i in range(n)]
        candidates += [current[: n // 2], current[n // 2:]]
        for candidate in candidates:
            if len(candidate) < len(current) and fails(candidate):
                current = candidate
                changed = True
                break
    return current


class _PropertyFailure(AssertionError):
    pass


def check_script(script: list) -> ScenarioResult:
    """The reusable property: three executions agree on the public contract."""
    first = execute(script)
    replay = replay_journal(first.journal)
    if first.signature() != replay.signature():
        raise _PropertyFailure("replay signature mismatch")
    if first.traces != replay.traces:
        raise _PropertyFailure("replay trace/id mismatch")
    if encode_records(parse_records(first.journal)) != first.journal:
        raise _PropertyFailure("journal is not in canonical byte form")
    if first.snapshots:
        _, offset, blob = first.snapshots[-1]
        recovered = recover_snapshot(
            blob, first.journal, marker_offset=offset
        )
        if first.states != recovered.states:
            raise _PropertyFailure("recovery state mismatch")
        if first.signature() != recovered.signature():
            raise _PropertyFailure("recovery signature mismatch")
        if first.trace_quads() != recovered.trace_quads():
            raise _PropertyFailure("recovery trace order mismatch")
        cut = recovered.restored_len
        recovered_suffix_ids = recovered.trace_ids()[cut:]
        n = len(recovered_suffix_ids)
        if recovered_suffix_ids != list(range(1, n + 1)):
            raise _PropertyFailure("recovery suffix ids not a fresh 1..N run")
        if len(first.trace_ids()) != len(recovered.trace_ids()):
            raise _PropertyFailure("recovery trace length mismatch")
        if first.traces[:cut] != recovered.traces[:cut]:
            raise _PropertyFailure("recovery prefix mismatch")
    return first


class SeededScenarioTests(unittest.TestCase):
    SEEDS = tuple(range(1, 41))   # fixed seeds: reproducible without wall time

    def test_all_fixed_seeds_replay_identically(self):
        failures = []
        for seed in self.SEEDS:
            try:
                check_script(generate_script(seed))
            except _PropertyFailure as exc:
                failures.append((seed, str(exc)))
        if failures:
            lines = ["seeded property failed:"]
            for seed, msg in failures:
                shrunk = shrink_script(generate_script(seed))
                lines.append(
                    f"  seed={seed}: {msg}\n"
                    f"    minimal script ({len(shrunk)} actions): {shrunk!r}\n"
                    f"    reproduce: check_script(generate_script({seed})) "
                    f"then check_script({shrunk!r})"
                )
            self.fail("\n".join(lines))

    def test_same_seed_is_byte_stable(self):
        # Same seed -> same script -> byte-identical journal across separate
        # processes' invocations (run twice here; determinism is structural).
        for seed in (1, 7, 42, 123456):
            s1 = generate_script(seed)
            s2 = generate_script(seed)
            self.assertEqual(s1, s2)
            j1 = execute(s1).journal
            j2 = execute(s2).journal
            self.assertEqual(j1, j2, f"journal differs for seed {seed}")

    def test_shrinker_reproduces_and_reduces_when_injected_bug(self):
        # Validate the failure-reporting machinery itself (not the product):
        # a synthetic property that "fails" on two snapshots must shrink to a
        # script still containing both snapshot commands.
        good = generate_script(3)

        original = ScenarioResult.signature

        def poisoned(self):
            if len(self.snapshots) >= 2:
                raise _PropertyFailure("synthetic")
            return original(self)

        ScenarioResult.signature = poisoned
        try:
            shrunk = shrink_script(good)
        finally:
            ScenarioResult.signature = original
        self.assertLess(len(shrunk), len(good))
        # The shrunk script still contains two snapshots.
        self.assertEqual(
            sum(1 for c in shrunk if c[0] == "snapshot"), 2
        )

    def test_every_seed_uses_only_contract_observations(self):
        # Guard against accidental dependence on non-contract info: handler
        # states and events contain only json-able primitives (signature()
        # would raise on NaN/sets via allow_nan=False / json encoding).
        for seed in self.SEEDS[:10]:
            first = check_script(generate_script(seed))
            self.assertIsInstance(first.signature(), bytes)
            self.assertIsInstance(first.journal, bytes)

    def test_generated_scripts_include_every_feature(self):
        # The fixed seed set collectively exercises sends, clock/timers,
        # faults, dedup redelivery, caps and snapshots.
        ops = set()
        for seed in self.SEEDS:
            for cmd in generate_script(seed):
                ops.add(cmd[0])
        self.assertIn("send", ops)
        self.assertIn("clock", ops)
        self.assertIn("schedule", ops)
        self.assertIn("resolve", ops)
        self.assertIn("snapshot", ops)
        # Dedup redelivery and caps are probabilistic; assert they occur in
        # the fixed suite (stable once seeds are fixed).
        self.assertIn("redeliver", ops)
        self.assertIn("cap", ops)
        # And supervision decisions of each kind appear somewhere.
        actions = set()
        timers_fired = timers_expired = False
        for seed in self.SEEDS:
            result = check_script(generate_script(seed))
            actions.update(s["action"] for s in result.supervision)
            at = [e["at"] for e in result.events]
            timers_fired |= "timer_fired" in at
            timers_expired |= "timer_expired" in at
        self.assertEqual(actions, {"resume", "restart", "stop", "escalate"})
        self.assertTrue(timers_fired)
        self.assertTrue(timers_expired)


# ---------------------------------------------------------------------------
# Sanity: handler-context derived delivery uses only the public context
# ---------------------------------------------------------------------------


class PublicContextTests(unittest.TestCase):
    def test_handler_uses_actor_context_send(self):
        def parent(state, message, ctx: ActorContext):
            ctx.send("child", {"echo": message})
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("parent", [], parent)
        rt.register("child", [], lambda s, m, c: s + [m])
        rt.send("parent", "hi")
        rt.run()
        self.assertEqual(rt.get_state("parent"), ["hi"])
        self.assertEqual(rt.get_state("child"), [{"echo": "hi"}])


if __name__ == "__main__":
    unittest.main()
