"""Deterministic repeat-execution regression tests (public product API only).

These tests add no product surface and read no private field: they drive the
runtime exclusively through its documented entry points -- ``register``,
``send``, ``schedule``, ``advance``, ``run(limit=...)`` -- and observe it only
through ``clock``, ``get_state``, ``pending_count``, ``scheduled_count``,
``trace`` and the documented result/exception types.

The same initial state, the same runtime configuration (two interacting
actors registered in a fixed order, a fixed per-input completion cap) and the
same message sequence are executed several times.  Every repetition must be
item-for-item identical: the completion-order public trace (message id, actor
id, priority, state before/after), the logical clock, mailbox/scheduled
counts and the final queryable state.  Nothing depends on wall-clock time,
random numbers, thread timing or machine speed -- the clock is the runtime's
own logical clock advanced explicitly, and message identity is the runtime's
own fixed monotonic id sequence, so the expected ids are asserted literally.

The core scene contains, on purpose, every ordering hazard at once:

* two actors (``alpha``/``beta``) that exchange messages -- an ``alpha``
  follow-up is buffered by the handler and only enters ``beta``'s mailbox
  after the triggering message commits, and ``beta`` answers ``alpha``;
* ordinary (priority 0) and high-priority (priority 10) messages: the
  high-priority message overtakes only mail whose processing has not yet
  started, never a message already being handled (a handler commits
  atomically; ``run(limit=1)`` completes exactly one whole message);
* equal-priority messages running in stable mailbox-entry (id) order;
* several timed messages sharing one deadline, entering the mailbox in the
  runtime's existing stable (deadline then id) order, plus a message whose
  deadline and expiry coincide and so never reaches a handler;
* a synchronous delivery, whose result is visible strictly after that
  message's processing ends, and an asynchronous one (``schedule``), whose
  accepted id means only "held by the runtime", never "processed".

On any ordering drift the comparison below reports the first phase and,
within it, the first differing trace entry rather than only the end state, so
an intermediate scheduling divergence cannot hide behind an equal final state.
"""
from __future__ import annotations

import copy
import unittest

from actor_runtime import ActorRuntime

ALPHA = "alpha"
BETA = "beta"
ACTORS = (ALPHA, BETA)

# Fixed per-external-input completion bound: the "concurrency limit" of the
# deterministic scene. Every batch below completes at most this many messages.
CAP = 2

INITIAL_ALPHA = {"log": []}
INITIAL_BETA = {"log": [], "total": 0}


def alpha_handler(state, message, ctx):
    """Records notes; a note asks beta to bump; beta answers with an ack."""
    kind, tag = message
    log = list(state["log"])
    if kind == "note":
        log.append(("note", tag))
        # Follow-up delivery buffered now, admitted to beta's mailbox only
        # after this processing commits (derived priority is always 0).
        ctx.send(BETA, ("bump", tag))
    elif kind == "acked":
        log.append(("acked", tag))
    elif kind == "tick":
        log.append(("tick", tag))
    else:
        raise AssertionError(f"alpha got unexpected message: {message!r}")
    return {"log": log}


def beta_handler(state, message, ctx):
    """Each bump is counted and answered with an asynchronous ack to alpha."""
    kind, tag = message
    if kind != "bump":
        raise AssertionError(f"beta got unexpected message: {message!r}")
    ctx.send(ALPHA, ("acked", tag))
    return {
        "log": list(state["log"]) + [("bump", tag)],
        "total": state["total"] + 1,
    }


# ---------------------------------------------------------------------------
# Observation and comparison (public facts only)
# ---------------------------------------------------------------------------


def observe(rt: ActorRuntime) -> dict:
    """Snapshot every contract-visible fact at one point in the execution."""
    return {
        "clock": rt.clock(),
        "states": {name: copy.deepcopy(rt.get_state(name)) for name in ACTORS},
        "pending": {name: rt.pending_count(name) for name in ACTORS},
        "scheduled": {name: rt.scheduled_count(name) for name in ACTORS},
        "trace": [tuple(entry) for entry in rt.trace()],
    }


def _first_trace_difference(expected, actual):
    """Return (index, expected_entry, actual_entry) for the first trace drift."""
    for index in range(max(len(expected), len(actual))):
        exp = expected[index] if index < len(expected) else "<missing>"
        got = actual[index] if index < len(actual) else "<missing>"
        if exp != got:
            return index, exp, got
    return None


def assert_runs_identical(testcase: unittest.TestCase, runs: list) -> None:
    """Assert several (label, observation) phase streams are item-identical.

    A divergence is reported at the first phase and then at the first trace
    entry that differs, instead of as a bare end-state inequality, so an
    intermediate ordering drift is always localised.
    """
    baseline = runs[0]
    for repeat_index, run in enumerate(runs[1:], start=2):
        testcase.assertEqual(
            [label for label, _ in baseline],
            [label for label, _ in run],
            f"repeat {repeat_index}: phase labels differ",
        )
        for phase_index, ((label, base), (_, other)) in enumerate(
            zip(baseline, run)
        ):
            if base["clock"] != other["clock"]:
                testcase.fail(
                    f"repeat {repeat_index}, phase {phase_index!r} ({label}): "
                    f"logical clock differs: {base['clock']!r} != "
                    f"{other['clock']!r}"
                )
            for key in ("pending", "scheduled", "states"):
                if base[key] != other[key]:
                    testcase.fail(
                        f"repeat {repeat_index}, phase {phase_index!r} "
                        f"({label}): first divergence in {key!r}: "
                        f"{base[key]!r} != {other[key]!r}"
                    )
            drift = _first_trace_difference(base["trace"], other["trace"])
            if drift is not None:
                index, exp, got = drift
                testcase.fail(
                    f"repeat {repeat_index}, phase {phase_index!r} ({label}): "
                    f"first trace divergence at completion #{index}: "
                    f"expected {exp!r}, got {got!r}"
                )


def _drain_capped(rt: ActorRuntime, phases: list, label: str) -> list:
    """Process under the fixed cap, recording one observation per batch."""
    batch_ids = []
    batch = 0
    while any(rt.pending_count(name) for name in ACTORS):
        before = len(rt.trace())
        completed = rt.run(limit=CAP)
        # A started message always finishes inside the same call: the cap
        # bounds completions, never a half-run handler.
        assert completed == len(rt.trace()) - before
        assert 0 < completed <= CAP
        ids = [entry.message_id for entry in rt.trace()[before:]]
        batch_ids.append(ids)
        batch += 1
        phases.append((f"{label}-batch{batch}", observe(rt)))
    return batch_ids


# ---------------------------------------------------------------------------
# Synchronous delivery scene
# ---------------------------------------------------------------------------


def run_synchronous_scene() -> tuple[list, dict, list]:
    """Drive the synchronous scene; return (phases, accepted_ids, batch_ids)."""
    rt = ActorRuntime()
    rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
    rt.register(BETA, INITIAL_BETA, beta_handler)

    phases: list = [("initial", observe(rt))]
    accepted = {}
    accepted["note-a"] = rt.send(ALPHA, ("note", "a"))            # ordinary
    accepted["note-b"] = rt.send(ALPHA, ("note", "b"))            # ordinary
    accepted["note-hi"] = rt.send(ALPHA, ("note", "hi"), priority=10)
    accepted["bump-ext"] = rt.send(BETA, ("bump", "ext"))         # other actor
    phases.append(("after-sends-before-run", observe(rt)))

    # Synchronous semantics: nothing has been processed merely by sending.
    assert rt.trace() == []
    assert rt.get_state(ALPHA) == INITIAL_ALPHA
    assert rt.get_state(BETA) == INITIAL_BETA

    batch_ids = _drain_capped(rt, phases, "sync")
    phases.append(("sync-final", observe(rt)))
    return phases, accepted, batch_ids


SYNC_ACCEPTED_IDS = {"note-a": 1, "note-b": 2, "note-hi": 3, "bump-ext": 4}

# High-priority note-hi (id 3) overtakes the two unstarted ordinary notes
# (ids 1, 2); equal priority stays FIFO by id. Registration order then mixes
# in beta, while each note's derived bump (ids 5-7, priority 0) and beta's
# derived acks (ids 8-11) only appear after their triggering commit.
SYNC_BATCH_IDS = [
    [3, 1],
    [2, 4],
    [8, 5],
    [9, 6],
    [10, 7],
    [11],
]

SYNC_FINAL_TRACE = [
    (3, ALPHA, 10),
    (1, ALPHA, 0),
    (2, ALPHA, 0),
    (4, BETA, 0),
    (8, ALPHA, 0),
    (5, BETA, 0),
    (9, ALPHA, 0),
    (6, BETA, 0),
    (10, ALPHA, 0),
    (7, BETA, 0),
    (11, ALPHA, 0),
]

SYNC_FINAL_ALPHA_LOG = [
    ("note", "hi"), ("note", "a"), ("note", "b"),
    ("acked", "ext"), ("acked", "hi"), ("acked", "a"), ("acked", "b"),
]
SYNC_FINAL_BETA_LOG = [
    ("bump", "ext"), ("bump", "hi"), ("bump", "a"), ("bump", "b"),
]


class SynchronousDeliveryDeterminismTests(unittest.TestCase):
    def test_repeated_executions_are_item_for_item_identical(self):
        runs = [run_synchronous_scene()[0] for _ in range(3)]
        assert_runs_identical(self, runs)

    def test_accepted_ids_batches_states_and_trace_match_exact_contract(self):
        phases, accepted, batch_ids = run_synchronous_scene()
        self.assertEqual(accepted, SYNC_ACCEPTED_IDS)
        self.assertEqual(batch_ids, SYNC_BATCH_IDS)
        final = phases[-1][1]
        self.assertEqual(final["clock"], 0)          # sends/runs move no clock
        self.assertEqual(final["pending"], {ALPHA: 0, BETA: 0})
        self.assertEqual(
            [(t[0], t[1], t[2]) for t in final["trace"]], SYNC_FINAL_TRACE
        )
        self.assertEqual(final["states"][ALPHA]["log"], SYNC_FINAL_ALPHA_LOG)
        self.assertEqual(final["states"][BETA]["log"], SYNC_FINAL_BETA_LOG)
        self.assertEqual(final["states"][BETA]["total"], 4)

    def test_high_priority_overtakes_only_unstarted_mail(self):
        # One ordinary message waits while a later high-priority message is
        # admitted; the first single completion is the high-priority one and
        # the ordinary message is still untouched afterwards.
        rt = ActorRuntime()
        rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
        rt.register(BETA, INITIAL_BETA, beta_handler)
        low = rt.send(ALPHA, ("note", "low"))
        high = rt.send(ALPHA, ("note", "high"), priority=10)
        self.assertEqual(rt.run(limit=1), 1)
        head = rt.trace()[-1]
        self.assertEqual((head.message_id, head.actor_name, head.priority),
                         (high, ALPHA, 10))
        # The overtaken low message is still pending, unstarted.
        self.assertEqual(rt.pending_count(ALPHA), 1)
        self.assertNotIn(("note", "low"), rt.get_state(ALPHA)["log"])
        # The committed high message already admitted its own follow-up, but
        # the overtaken low message's follow-up has not: beta holds exactly
        # one (high's) unprocessed bump and has handled nothing yet.
        self.assertEqual(rt.pending_count(BETA), 1)
        self.assertEqual(rt.get_state(BETA)["log"], [])
        # Drain: the once-waiting low message now completes normally and only
        # then admits its derived mail; nothing ever reorders a started one.
        rt.run()
        self.assertEqual(low, 1)
        # The full drain also runs beta's two bumps and beta's answers back to
        # alpha; the note order (high then the once-waiting low) is preserved.
        self.assertEqual(
            rt.get_state(ALPHA)["log"],
            [("note", "high"), ("note", "low"),
             ("acked", "high"), ("acked", "low")],
        )
        self.assertEqual(
            rt.get_state(BETA)["log"], [("bump", "high"), ("bump", "low")]
        )

    def test_synchronous_result_is_visible_only_after_processing(self):
        # Acceptance (an id) is not completion: before run() the effect is
        # absent; after processing one message its committed state is visible.
        rt = ActorRuntime()
        rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
        rt.register(BETA, INITIAL_BETA, beta_handler)
        mid = rt.send(ALPHA, ("note", "visible-when-done"))
        self.assertIsInstance(mid, int)
        self.assertEqual(rt.get_state(ALPHA)["log"], [])
        self.assertEqual(rt.trace(), [])
        rt.run(limit=1)
        self.assertEqual(
            rt.get_state(ALPHA)["log"], [("note", "visible-when-done")]
        )
        self.assertEqual(rt.trace()[-1].message_id, mid)


# ---------------------------------------------------------------------------
# Asynchronous (timed) delivery scene
# ---------------------------------------------------------------------------


def run_asynchronous_scene() -> tuple[list, dict, list, list]:
    """Drive the timed scene; return (phases, accepted_ids, advances, batches)."""
    rt = ActorRuntime()
    rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
    rt.register(BETA, INITIAL_BETA, beta_handler)

    phases: list = [("initial", observe(rt))]
    scheduled = {}
    # Three alpha timers and one beta timer share deadline 3; one alpha
    # message (q) is high priority. stale expires on its deadline tick 2.
    scheduled["p"] = rt.schedule(ALPHA, ("tick", "p"), delay=3, priority=0)
    scheduled["q"] = rt.schedule(ALPHA, ("tick", "q"), delay=3, priority=10)
    scheduled["r"] = rt.schedule(ALPHA, ("tick", "r"), delay=3, priority=0)
    scheduled["bt"] = rt.schedule(BETA, ("bump", "t"), delay=3, priority=0)
    scheduled["stale"] = rt.schedule(
        ALPHA, ("tick", "stale"), delay=2, ttl=2, priority=0
    )
    scheduled["late"] = rt.schedule(ALPHA, ("tick", "late"), delay=4)
    phases.append(("after-schedule", observe(rt)))

    advances = []
    advances.append(tuple(rt.advance(2)))                 # stale expires at t2
    phases.append(("clock-2", observe(rt)))
    advances.append(tuple(rt.advance(1)))                 # four released at t3
    phases.append(("clock-3-released-not-run", observe(rt)))

    # Releasing invokes no handler: accepted/released is not processed.
    assert rt.trace() == []
    assert rt.get_state(ALPHA) == INITIAL_ALPHA

    batch_ids = _drain_capped(rt, phases, "async-t3")
    advances.append(tuple(rt.advance(1)))                 # late released at t4
    phases.append(("clock-4", observe(rt)))
    batch_ids += _drain_capped(rt, phases, "async-t4")
    phases.append(("async-final", observe(rt)))
    return phases, scheduled, advances, batch_ids


ASYNC_SCHEDULED_IDS = {
    "p": 1, "q": 2, "r": 3, "bt": 4, "stale": 5, "late": 6,
}
# (time, released, expired)
ASYNC_ADVANCES = [
    (2, (), (5,)),
    (3, (1, 2, 3, 4), ()),       # stable deadline-then-id release order
    (4, (6,), ()),
]
# Same-deadline batch: high-priority q overtakes unstarted p/r; FIFO ties.
ASYNC_BATCH_IDS = [
    [2, 1],
    [3, 4],
    [7],                          # beta's derived ack, committed after id 4
    [6],                          # late timer released at t4
]
ASYNC_FINAL_TRACE = [
    (2, ALPHA, 10),
    (1, ALPHA, 0),
    (3, ALPHA, 0),
    (4, BETA, 0),
    (7, ALPHA, 0),
    (6, ALPHA, 0),
]
ASYNC_FINAL_ALPHA_LOG = [
    ("tick", "q"), ("tick", "p"), ("tick", "r"),
    ("acked", "t"), ("tick", "late"),
]
ASYNC_FINAL_BETA_LOG = [("bump", "t")]


class AsynchronousDeliveryDeterminismTests(unittest.TestCase):
    def test_repeated_executions_are_item_for_item_identical(self):
        runs = [run_asynchronous_scene()[0] for _ in range(3)]
        assert_runs_identical(self, runs)

    def test_ids_advances_batches_states_and_trace_match_exact_contract(self):
        phases, scheduled, advances, batch_ids = run_asynchronous_scene()
        self.assertEqual(scheduled, ASYNC_SCHEDULED_IDS)
        self.assertEqual(advances, ASYNC_ADVANCES)
        self.assertEqual(batch_ids, ASYNC_BATCH_IDS)
        final = phases[-1][1]
        self.assertEqual(final["clock"], 4)
        self.assertEqual(final["scheduled"], {ALPHA: 0, BETA: 0})
        self.assertEqual(final["pending"], {ALPHA: 0, BETA: 0})
        self.assertEqual(
            [(t[0], t[1], t[2]) for t in final["trace"]], ASYNC_FINAL_TRACE
        )
        self.assertEqual(final["states"][ALPHA]["log"], ASYNC_FINAL_ALPHA_LOG)
        self.assertEqual(final["states"][BETA]["log"], ASYNC_FINAL_BETA_LOG)
        self.assertEqual(final["states"][BETA]["total"], 1)

    def test_same_deadline_timers_enter_by_stable_id_order(self):
        # The release list at one deadline is ordered by id regardless of the
        # mailbox priority that later decides completion order.
        rt = ActorRuntime()
        rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
        a = rt.schedule(ALPHA, ("tick", "a"), delay=1, priority=0)
        b = rt.schedule(ALPHA, ("tick", "b"), delay=1, priority=0)
        c = rt.schedule(ALPHA, ("tick", "c"), delay=1, priority=0)
        result = rt.advance(1)
        self.assertEqual(result.released, (a, b, c))
        rt.run()
        self.assertEqual(
            rt.get_state(ALPHA)["log"],
            [("tick", "a"), ("tick", "b"), ("tick", "c")],
        )

    def test_accepted_scheduled_message_is_not_yet_processed(self):
        # An accepted (id returned) scheduled delivery is merely held: not in
        # a mailbox, not processed, absent from state and trace. Even after its
        # release it remains ordinary unprocessed mailbox mail until run().
        rt = ActorRuntime()
        rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
        mid = rt.schedule(ALPHA, ("tick", "x"), delay=1)
        self.assertEqual(rt.scheduled_count(ALPHA), 1)
        self.assertEqual(rt.pending_count(ALPHA), 0)
        self.assertEqual(rt.get_state(ALPHA)["log"], [])
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.advance(1).released, (mid,))
        self.assertEqual(rt.pending_count(ALPHA), 1)   # now mailbox mail
        self.assertEqual(rt.get_state(ALPHA)["log"], [])
        self.assertEqual(rt.trace(), [])               # release ran no handler
        rt.run()
        self.assertEqual(rt.get_state(ALPHA)["log"], [("tick", "x")])
        self.assertEqual(rt.trace()[-1].message_id, mid)

    def test_same_tick_deadline_and_expiry_never_reaches_handler(self):
        rt = ActorRuntime()
        rt.register(ALPHA, INITIAL_ALPHA, alpha_handler)
        mid = rt.schedule(ALPHA, ("tick", "gone"), delay=3, ttl=3)
        result = rt.advance(3)
        self.assertEqual(result, (3, (), (mid,)))
        self.assertEqual(rt.pending_count(ALPHA), 0)
        self.assertEqual(rt.scheduled_count(ALPHA), 0)
        rt.run()
        self.assertEqual(rt.get_state(ALPHA)["log"], [])
        self.assertEqual(rt.trace(), [])


if __name__ == "__main__":
    unittest.main()
