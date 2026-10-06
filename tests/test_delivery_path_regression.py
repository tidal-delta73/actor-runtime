"""Cross-cutting regression for the unified internal delivery path.

The refactor behind this module made ``send``, ``send_once``, ``schedule``
and ``ActorContext.send`` share one private validate/resolve/copy/number/
commit boundary without touching the public surface.  This scenario drives
all four paths -- a successful delivery, an idempotent first delivery plus
duplicate confirmations, timed release and expiry, and a handler failure
with rollback followed by a retry -- interleaved in one replay and asserts
both the concrete outcome and that the same input sequence reproduces it
byte-for-observably across independent runtimes.
"""
import copy
import unittest

from actor_runtime import (
    ActorContext,
    ActorDataCopyError,
    ActorExecutionError,
    ActorRuntime,
    AdvanceResult,
    DedupResult,
)


class UncopyablePlaceholder:
    """A duplicate-call placeholder that must never be copied or compared."""

    def __deepcopy__(self, memo):
        raise AssertionError("duplicate placeholder must not be copied")

    def __eq__(self, other):
        raise AssertionError("duplicate placeholder must not be compared")


def build_scenario():
    """Drive one fixed interleaved script; return the observable signature.

    Everything contract-visible is captured in the returned tuple so two
    independent executions can be compared for exact equivalence.
    """
    flag = {"fail": True}

    def a_handler(state, message, ctx: ActorContext):
        state = list(state)
        if message == "boom":
            # Derived send happens before the raise: on the failing attempt it
            # must be buffered, copied at commit staging and then dropped with
            # the rest of the transaction; on the retry it commits.
            ctx.send("b", "derived:boom")
            if flag["fail"]:
                raise RuntimeError("boom")
        state.append(message)
        return state

    def b_handler(state, message, ctx):
        return list(state) + [message]

    rt = ActorRuntime()
    rt.register("a", [], a_handler)
    rt.register("b", [], b_handler)

    dedup_results = []
    validation_errors = []
    advances = []
    failure = None

    # -- successful ordinary + idempotent deliveries ----------------------
    id_s1 = rt.send("a", "s1")
    dedup_results.append(rt.send_once("a", "k1", "once1"))
    # Duplicate while pending: placeholder is neither copied nor compared and
    # no id is consumed.
    dedup_results.append(rt.send_once("a", "k1", UncopyablePlaceholder()))
    first_run = rt.run(limit=2)

    # -- timed deliveries share the id sequence ---------------------------
    id_later = rt.schedule("b", "later1", delay=3)
    id_stale = rt.schedule("b", "stale", delay=2, ttl=2)  # expires at tick 2
    id_boom = rt.send("a", "boom", priority=5)

    # Validation failures of every kind, interspersed: none may move the id
    # counter, a mailbox, the scheduled queue, the clock or the dedup record.
    def reject(action, *exc_types):
        try:
            action()
        except exc_types as exc:
            validation_errors.append(type(exc).__name__)
        else:  # pragma: no cover - scenario guard
            raise AssertionError("expected a validation error")

    reject(lambda: rt.send("a", "x", priority=1.5), TypeError)
    reject(lambda: rt.schedule("a", "x", delay=-1), ValueError)
    reject(lambda: rt.schedule("a", "x", delay=True), TypeError)
    reject(lambda: rt.schedule("a", "x", delay=1, ttl=0), ValueError)
    reject(lambda: rt.send_once("ghost", 123, "m"), TypeError)
    reject(lambda: rt.send_once("ghost", "", "m"), ValueError)
    reject(lambda: rt.send("ghost", "x"), LookupError)

    # -- advance: expiry wins on the deadline tick; the other stays held ---
    advances.append(tuple(rt.advance(2)))

    # -- the high-priority failing message is selected and rolls back ------
    pre_failure_trace = [tuple(e) for e in rt.trace()]
    pre_failure_pending = (rt.pending_count("a"), rt.pending_count("b"))
    try:
        rt.run()
    except ActorExecutionError as exc:
        failure = (exc.actor_name, exc.message_id,
                   type(exc.original).__name__)
    # A duplicate confirmation after the rollback still resolves to the first
    # id; the key reservation survived.
    dedup_results.append(rt.send_once("a", "k1", UncopyablePlaceholder()))

    # The failed processing consumed no id: the next external send takes it.
    id_probe = rt.send("a", "probe")

    # -- release the surviving scheduled delivery, then retry -------------
    advances.append(tuple(rt.advance(1)))
    flag["fail"] = False
    second_run = rt.run()
    id_tail = rt.send("a", "tail")
    rt.run()
    dedup_results.append(rt.send_once("a", "k1", "again"))

    return {
        "ids": (id_s1, id_later, id_stale, id_boom, id_probe, id_tail),
        "first_run": first_run,
        "second_run": second_run,
        "dedup": [tuple(r) for r in dedup_results],
        "validation_errors": validation_errors,
        "advances": advances,
        "failure": failure,
        "pre_failure_trace": pre_failure_trace,
        "pre_failure_pending": pre_failure_pending,
        "clock": rt.clock(),
        "states": (
            copy.deepcopy(rt.get_state("a")),
            copy.deepcopy(rt.get_state("b")),
        ),
        "pending": (rt.pending_count("a"), rt.pending_count("b")),
        "scheduled": (rt.scheduled_count("a"), rt.scheduled_count("b")),
        "trace": [tuple(e) for e in rt.trace()],
    }


EXPECTED_TRACE = [
    (1, "a", 0, [], ["s1"]),
    (2, "a", 0, ["s1"], ["s1", "once1"]),
    # The retried high-priority failure keeps id 5 and priority 5.
    (5, "a", 5, ["s1", "once1"], ["s1", "once1", "boom"]),
    # The external probe that proved no id was consumed is still a-mail and
    # therefore drains before b is ever selected.
    (6, "a", 0, ["s1", "once1", "boom"],
     ["s1", "once1", "boom", "probe"]),
    # Released scheduled mail (id 3, priority 0) precedes the committed
    # derived message (id 7) in b's mailbox by smaller id.
    (3, "b", 0, [], ["later1"]),
    (7, "b", 0, ["later1"], ["later1", "derived:boom"]),
    (8, "a", 0, ["s1", "once1", "boom", "probe"],
     ["s1", "once1", "boom", "probe", "tail"]),
]


class UnifiedDeliveryPathRegressionTests(unittest.TestCase):
    def test_interleaved_scenario_matches_expected_observations(self):
        sig = build_scenario()

        # Global monotonic id sequence shared by every path; the expired
        # delivery's id is simply absent later, ids are never reused.
        self.assertEqual(
            sig["ids"], (1, 3, 4, 5, 6, 8)
        )

        # First drain completed exactly the two early messages; the retry
        # drain completed boom, probe, the released timer and the derived
        # message (tail is sent afterwards).
        self.assertEqual(sig["first_run"], 2)
        self.assertEqual(sig["second_run"], 4)

        # Duplicate confirmations all resolve to the first id (2, since the
        # plain send "s1" took id 1), including after the handler rollback;
        # the first call is the sole acceptance.
        self.assertEqual(
            sig["dedup"],
            [(2, True), (2, False), (2, False), (2, False)],
        )

        self.assertEqual(
            sig["validation_errors"],
            ["TypeError", "ValueError", "TypeError", "ValueError",
             "TypeError", "ValueError", "LookupError"],
        )

        # Tick 2: the delay=2/ttl=2 delivery expires exactly on its deadline
        # and never enters a mailbox; the delay=3 one is still held. Tick 3
        # releases it with its original scheduled id.
        self.assertEqual(sig["advances"], [
            tuple(AdvanceResult(2, (), (4,))),
            tuple(AdvanceResult(3, (3,), ())),
        ])

        # The failure is reported with actor, id and original; beforehand only
        # the first two messages had committed, and no derived mail existed.
        self.assertEqual(sig["failure"], ("a", 5, "RuntimeError"))
        self.assertEqual(
            sig["pre_failure_trace"],
            [(1, "a", 0, [], ["s1"]),
             (2, "a", 0, ["s1"], ["s1", "once1"])],
        )
        self.assertEqual(sig["pre_failure_pending"], (1, 0))

        self.assertEqual(sig["clock"], 3)
        self.assertEqual(
            sig["states"],
            (["s1", "once1", "boom", "probe", "tail"],
             ["later1", "derived:boom"]),
        )
        self.assertEqual(sig["pending"], (0, 0))
        self.assertEqual(sig["scheduled"], (0, 0))
        self.assertEqual(sig["trace"], EXPECTED_TRACE)

    def test_same_input_sequence_is_deterministic(self):
        first = build_scenario()
        for _ in range(3):
            self.assertEqual(build_scenario(), first)

    def test_dedup_result_types_are_preserved(self):
        # The refactor must keep returning the documented public result
        # types, not internal equivalents.
        rt = ActorRuntime()
        rt.register("a", [], lambda s, m, c: list(s) + [m])
        first = rt.send_once("a", "k", "m")
        duplicate = rt.send_once("a", "k", UncopyablePlaceholder())
        self.assertIsInstance(first, DedupResult)
        self.assertIsInstance(duplicate, DedupResult)
        self.assertEqual(first, DedupResult(1, True))
        self.assertEqual(duplicate, DedupResult(1, False))

    def test_copy_failure_on_entry_still_changes_nothing(self):
        # The unified boundary keeps the entry-point copy-failure contract:
        # ActorDataCopyError propagates unwrapped and consumes no id or key.
        rt = ActorRuntime()
        rt.register("a", [], lambda s, m, c: list(s) + [m])
        boom = RuntimeError("no copy")

        class Bad:
            def __deepcopy__(self, memo):
                raise boom

        with self.assertRaises(ActorDataCopyError) as caught:
            rt.send("a", Bad())
        self.assertIs(caught.exception.original, boom)
        with self.assertRaises(ActorDataCopyError) as caught:
            rt.send_once("a", "k", Bad())
        self.assertIs(caught.exception.original, boom)
        self.assertEqual(rt.send("a", "plain"), 1)
        self.assertEqual(rt.send_once("a", "k", "fixed"), DedupResult(2, True))


if __name__ == "__main__":
    unittest.main()
