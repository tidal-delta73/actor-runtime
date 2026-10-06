"""Cross-cutting regression tests for the unified delivery pipeline.

These scenarios deliberately interleave, inside one runtime, the features
whose internal paths the refactor unified:

* successful plain deliveries on the global monotonic id sequence;
* ``send_once`` first acceptance and duplicate confirmation (placeholders
  neither copied nor compared, reservations surviving rollback);
* timed release and same-tick expiry driven through ``advance``;
* failed processing (handler exception, non-``Exception`` ``BaseException``
  and deep-copy failure at the commit boundary) with full rollback: the
  message stays pending, no derived delivery or id becomes visible, and the
  later retry numbers derived mail continuously without inheriting the
  triggering message's priority.

Every assertion uses concrete, exhaustively enumerated values -- ids,
``DedupResult``, ``AdvanceResult``, counts, clock, full trace tuples and
states -- so any observable divergence between the old and new internal
paths fails the suite deterministically.
"""
import copy
import unittest

from actor_runtime import (
    ActorDataCopyError,
    ActorExecutionError,
    ActorRuntime,
    DedupResult,
)


def append_handler(state, message, ctx):
    return list(state) + [message]


class UncopyablePlaceholder:
    """A duplicate placeholder: it must never be copied or compared."""

    def __deepcopy__(self, memo):
        raise AssertionError("duplicate placeholder must not be copied")

    def __eq__(self, other):
        raise AssertionError("duplicate placeholder must not be compared")


class CopyBlocked:
    """Payload whose deep copy always fails."""

    def __deepcopy__(self, memo):
        raise RuntimeError("copy blocked")


class InterleavedDeliveryRegressionTests(unittest.TestCase):
    """Success, dedup confirmation, timers and failed rollback in one script."""

    @staticmethod
    def _build_and_drive():
        """Run the fixed interleaved script; return every public fact."""
        fail = {"on": True}

        def a_handler(state, message, ctx):
            if message == "boom":
                # Derived mail is buffered on both attempts; it may only
                # become visible (and take an id) once the retry commits.
                ctx.send("b", "d-boom")
                if fail["on"]:
                    raise RuntimeError("kaboom")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], a_handler)
        rt.register("b", [], append_handler)
        rt.register("c", [], append_handler)
        facts = []

        # -- mixed acceptance on the global id sequence ------------------
        facts.append(("send-a", rt.send("a", "m1")))                          # 1
        facts.append(("once-a",
                      tuple(rt.send_once("a", "k1", "once1"))))               # 2
        facts.append(("sched-b",
                      rt.schedule("b", "tb", delay=2, priority=3)))           # 3
        # Same-tick deadline and ttl: must expire, never reach a mailbox.
        facts.append(("sched-a-stale",
                      rt.schedule("a", "stale", delay=2, ttl=2)))             # 4
        facts.append(("sched-c", rt.schedule("c", "tc", delay=1)))            # 5
        facts.append(("once-c",
                      tuple(rt.send_once("c", "kc", "tc-once"))))             # 6
        facts.append(("once-c-dup",
                      tuple(rt.send_once("c", "kc",
                                         UncopyablePlaceholder()))))           # 6 dup

        # -- first clock step: c's timer is released ----------------------
        facts.append(("advance-1", tuple(rt.advance(1))))
        facts.append(("clock", rt.clock()))
        facts.append(("counts-t1",
                      (rt.pending_count("a"), rt.pending_count("b"),
                       rt.pending_count("c"), rt.scheduled_count("a"),
                       rt.scheduled_count("b"), rt.scheduled_count("c"))))

        # Process exactly one: registration order picks a, id order picks 1.
        facts.append(("run-limit1", rt.run(limit=1)))
        facts.append(("trace-1", [tuple(e) for e in rt.trace()]))

        # -- inject a failure, crossing a fresh send and dedup confirm ----
        facts.append(("send-a-boom",
                      rt.send("a", "boom", priority=5)))                       # 7
        facts.append(("send-b-later", rt.send("b", "later")))                 # 8
        facts.append(("once-a-dup-before-fail",
                      tuple(rt.send_once("a", "k1",
                                         UncopyablePlaceholder()))))           # 2 dup

        try:
            rt.run()
            facts.append(("run-boom", "NO-ERROR"))
        except ActorExecutionError as exc:
            facts.append(("run-boom",
                          (type(exc).__name__, exc.actor_name, exc.message_id,
                           type(exc.original).__name__,
                           type(exc.__cause__).__name__)))
        else:
            self.fail("handler failure was not wrapped")

        facts.append(("counts-after-fail",
                      (rt.pending_count("a"), rt.pending_count("b"),
                       rt.pending_count("c"), rt.scheduled_count("a"),
                       rt.scheduled_count("b"), rt.scheduled_count("c"))))
        facts.append(("trace-after-fail", [tuple(e) for e in rt.trace()]))
        # The reservation survives the rollback; duplicates keep confirming 2.
        facts.append(("once-a-dup-after-fail",
                      tuple(rt.send_once("a", "k1",
                                         UncopyablePlaceholder()))))
        # The failed attempt consumed no id: the buffered d-boom was never
        # numbered, so this external delivery gets the very next id.
        facts.append(("send-c-after-fail", rt.send("c", "x")))                # 9

        # -- second clock step (tick 1 -> 2): b's timer is released and
        # a's stale one expires at the very same tick (expiry wins) --
        facts.append(("advance-2", tuple(rt.advance(1))))
        facts.append(("clock-2", rt.clock()))
        facts.append(("counts-t2",
                      (rt.pending_count("a"), rt.pending_count("b"),
                       rt.pending_count("c"), rt.scheduled_count("a"),
                       rt.scheduled_count("b"), rt.scheduled_count("c"))))

        # -- retry commits: the boomed message completes once with its
        # original id/priority; d-boom is numbered continuously and enters
        # b at derived priority 0, not the triggering priority 5.
        fail["on"] = False
        facts.append(("run-recover", rt.run()))
        facts.append(("states",
                      (copy.deepcopy(rt.get_state("a")),
                       copy.deepcopy(rt.get_state("b")),
                       copy.deepcopy(rt.get_state("c")))))
        facts.append(("trace-full", [tuple(e) for e in rt.trace()]))
        facts.append(("counts-final",
                      (rt.pending_count("a"), rt.pending_count("b"),
                       rt.pending_count("c"), rt.scheduled_count("a"),
                       rt.scheduled_count("b"), rt.scheduled_count("c"))))
        facts.append(("once-a-late-dup",
                      tuple(rt.send_once("a", "k1",
                                         UncopyablePlaceholder()))))
        facts.append(("send-last", rt.send("a", "z")))
        return facts

    EXPECTED = [
        ("send-a", 1),
        ("once-a", (2, True)),
        ("sched-b", 3),
        ("sched-a-stale", 4),
        ("sched-c", 5),
        ("once-c", (6, True)),
        ("once-c-dup", (6, False)),
        ("advance-1", (1, (5,), ())),
        ("clock", 1),
        # pending a,b,c ; scheduled a,b,c
        ("counts-t1", (2, 0, 2, 1, 1, 0)),
        ("run-limit1", 1),
        ("trace-1", [(1, "a", 0, [], ["m1"])]),
        ("send-a-boom", 7),
        ("send-b-later", 8),
        ("once-a-dup-before-fail", (2, False)),
        ("run-boom", ("ActorExecutionError", "a", 7,
                      "RuntimeError", "RuntimeError")),
        # once1 (id 2) is still ahead behind the requeued boom; b holds only
        # the external id 8 while its timer is still queued; c's two
        # messages are untouched.
        ("counts-after-fail", (2, 1, 2, 1, 1, 0)),
        ("trace-after-fail", [(1, "a", 0, [], ["m1"])]),
        ("once-a-dup-after-fail", (2, False)),
        ("send-c-after-fail", 9),
        ("advance-2", (2, (3,), (4,))),
        ("clock-2", 2),
        # id 3 released into b; id 4 expired, so a keeps once1 + the boom.
        ("counts-t2", (2, 2, 3, 0, 0, 0)),
        ("run-recover", 8),
        ("states", (
            ["m1", "boom", "once1"],        # boom's priority 5 jumps once1
            ["tb", "later", "d-boom"],      # timer prio 3 first, then id 8/10
            ["tc", "tc-once", "x"],
        )),
        ("trace-full", [
            (1, "a", 0, [], ["m1"]),
            # The retried boom keeps its original id and priority 5, so it
            # completes ahead of the lower-priority once1 despite failing once.
            (7, "a", 5, ["m1"], ["m1", "boom"]),
            (2, "a", 0, ["m1", "boom"], ["m1", "boom", "once1"]),
            # Within b the released timer wins on priority; derived d-boom is
            # id 10 at priority 0, never inheriting the priority-5 trigger.
            (3, "b", 3, [], ["tb"]),
            (8, "b", 0, ["tb"], ["tb", "later"]),
            (10, "b", 0, ["tb", "later"], ["tb", "later", "d-boom"]),
            (5, "c", 0, [], ["tc"]),
            (6, "c", 0, ["tc"], ["tc", "tc-once"]),
            (9, "c", 0, ["tc", "tc-once"], ["tc", "tc-once", "x"]),
        ]),
        ("counts-final", (0, 0, 0, 0, 0, 0)),
        ("once-a-late-dup", (2, False)),
        ("send-last", 11),
    ]

    def test_interleaved_script_matches_exact_contract(self):
        self.assertEqual(self._build_and_drive(), self.EXPECTED)

    def test_interleaved_script_is_deterministic(self):
        first = self._build_and_drive()
        for _ in range(3):
            self.assertEqual(self._build_and_drive(), first)


class DerivedCopyFailureRollbackTests(unittest.TestCase):
    """A copy failure at the derived-message boundary hides every derived."""

    @staticmethod
    def _build_and_drive():
        copy_fail = {"on": True}

        def a_handler(state, message, ctx):
            if message == "go":
                ctx.send("b", "d1")
                # On the failed attempt the second derived cannot be copied;
                # the retry instead buffers a plain string.
                if copy_fail["on"]:
                    ctx.send("b", CopyBlocked())
                else:
                    ctx.send("b", "d3")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], a_handler)
        rt.register("b", [], append_handler)
        out = {}
        out["id_ext1"] = rt.send("b", "external1")                        # 1
        # Priority 8 must not leak to the priority-0 derived messages.
        out["id_go"] = rt.send("a", "go", priority=8)                     # 2

        try:
            rt.run()
            out["failure"] = None
        except ActorExecutionError as exc:
            out["failure"] = (
                exc.actor_name, exc.message_id,
                isinstance(exc.original, ActorDataCopyError),
                exc.original.what,
            )
        out["b_pending_after_fail"] = rt.pending_count("b")  # only external1
        out["a_pending_after_fail"] = rt.pending_count("a")  # go stays
        out["trace_after_fail"] = [tuple(e) for e in rt.trace()]

        # Dedup bookkeeping still works across the failed processing.
        out["once"] = tuple(rt.send_once("b", "kb", "once"))              # 3
        out["once_dup"] = tuple(rt.send_once("b", "kb",
                                             UncopyablePlaceholder()))
        out["id_ext2"] = rt.send("b", "external2")                        # 4

        copy_fail["on"] = False
        out["recovered"] = rt.run()
        out["state_b"] = rt.get_state("b")
        out["trace"] = [(e.message_id, e.actor_name, e.priority)
                        for e in rt.trace()]
        return out

    def test_failed_commit_hides_all_derived_and_consumes_no_id(self):
        out = self._build_and_drive()
        self.assertEqual(out["id_ext1"], 1)
        self.assertEqual(out["id_go"], 2)
        self.assertEqual(out["failure"], ("a", 2, True, "derived message"))
        self.assertEqual(out["a_pending_after_fail"], 1)
        self.assertEqual(out["b_pending_after_fail"], 1)
        self.assertEqual(out["trace_after_fail"], [])
        self.assertEqual(out["once"], (3, True))
        self.assertEqual(out["once_dup"], (3, False))
        self.assertEqual(out["id_ext2"], 4)
        # Retry: a completes (id 2, priority 8); the two derived messages are
        # born afterwards with continuous ids 5 and 6 at priority 0. Order in
        # b is plain id order: external1, once, external2, d1, d3.
        self.assertEqual(out["recovered"], 6)
        self.assertEqual(out["state_b"],
                         ["external1", "once", "external2", "d1", "d3"])
        self.assertEqual(out["trace"], [
            (2, "a", 8),
            (1, "b", 0),
            (3, "b", 0),
            (4, "b", 0),
            (5, "b", 0),
            (6, "b", 0),
        ])

    def test_copy_rollback_deterministic(self):
        first = self._build_and_drive()
        for _ in range(2):
            self.assertEqual(self._build_and_drive(), first)


class BaseExceptionInterleaveTests(unittest.TestCase):
    """A non-Exception BaseException propagates unchanged, mail retained."""

    def test_base_exception_crossing_timers_and_dedup_retries_cleanly(self):
        stop = SystemExit(9)
        hard = {"on": True}

        def handler(state, message, ctx):
            if message == "hard" and hard["on"]:
                raise stop
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.register("b", [], append_handler)
        self.assertEqual(rt.send("a", "ok"), 1)
        hard_id = rt.send("a", "hard", priority=4)                        # 2
        timer = rt.schedule("b", "tb", delay=1, priority=2)               # 3
        # The priority-4 "hard" mail is selected before the priority-0 "ok",
        # so the very first processing aborts and both messages stay queued.
        self.assertEqual(hard_id, 2)
        with self.assertRaises(SystemExit) as caught:
            rt.run()
        self.assertIs(caught.exception, stop)
        # The exact BaseException leaves every mailbox and the trace
        # untouched: no commit happened at all.
        self.assertEqual(rt.pending_count("a"), 2)
        self.assertEqual(rt.trace(), [])
        # Cross the pending failure with duplicate confirmation and new mail;
        # no id was consumed by the aborted attempt.
        self.assertEqual(tuple(rt.send_once("a", "k", "once")), (4, True))
        self.assertEqual(tuple(rt.send_once("a", "k",
                                            UncopyablePlaceholder())),
                         (4, False))
        self.assertEqual(rt.send("b", "plain"), 5)
        self.assertEqual(tuple(rt.advance(1)), (1, (timer,), ()))

        hard["on"] = False
        done = rt.run()
        # a drains first (registration order): the retried hard keeps id 2
        # and priority 4, so it completes before ok and once; b then drains
        # the released priority-2 timer before the plain priority-0 mail.
        self.assertEqual(done, 5)
        self.assertEqual(rt.get_state("a"), ["hard", "ok", "once"])
        self.assertEqual(rt.get_state("b"), ["tb", "plain"])
        self.assertEqual(
            [(e.message_id, e.actor_name, e.priority) for e in rt.trace()],
            [(2, "a", 4), (1, "a", 0), (4, "a", 0), (3, "b", 2), (5, "b", 0)],
        )


if __name__ == "__main__":
    unittest.main()
