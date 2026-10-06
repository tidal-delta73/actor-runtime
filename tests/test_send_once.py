"""Tests for idempotent delivery via ActorRuntime.send_once."""
import unittest

from actor_runtime import (
    ActorDataCopyError,
    ActorExecutionError,
    ActorRuntime,
    DedupResult,
)


def noop_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    state = list(state)
    state.append(message)
    return state


class Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("must never be copied")


class SendOnceAcceptanceTests(unittest.TestCase):
    def test_first_call_accepted_like_send(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        result = rt.send_once("a", "k", "m1", priority=2)
        self.assertIsInstance(result, DedupResult)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, DedupResult(message_id=1, accepted=True))
        self.assertEqual(rt.pending_count("a"), 1)
        # Comes from the same monotonic sequence as send.
        self.assertEqual(rt.send("a", "plain"), 2)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["m1", "plain"])
        entry = rt.trace()[0]
        self.assertEqual((entry.message_id, entry.priority), (1, 2))

    def test_result_is_immutable(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        result = rt.send_once("a", "k", None)
        with self.assertRaises(AttributeError):
            result.accepted = False
        with self.assertRaises(AttributeError):
            result.message_id = 9

    def test_message_is_copied_on_first_call(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        payload = ["m1"]
        rt.send_once("a", "k", payload)
        payload.append("mutated-after-delivery")
        rt.run()
        self.assertEqual(rt.get_state("a"), [["m1"]])

    def test_priority_rules_apply_to_first_delivery(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send_once("a", "low", "low", priority=0)
        rt.send_once("a", "high", "high", priority=5)
        rt.run()
        self.assertEqual(
            [t.message_id for t in rt.trace()], [2, 1]
        )
        self.assertEqual(rt.get_state("a"), ["high", "low"])


class SendOnceDuplicateTests(unittest.TestCase):
    def test_duplicate_while_pending_confirms_first_id(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        first = rt.send_once("a", "k", "m1", priority=3)
        again = rt.send_once("a", "k", "m1-different")
        self.assertEqual(first, DedupResult(1, True))
        self.assertEqual(again, DedupResult(1, False))
        # Not re-enqueued, no id consumed.
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.send("a", "plain"), 2)

    def test_duplicate_placeholder_is_never_copied_or_compared(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send_once("a", "k", "m1")
        # An object that cannot be copied (and is not equal to anything) is
        # perfectly acceptable as a duplicate placeholder.
        result = rt.send_once("a", "k", Uncopyable(), priority=-99)
        self.assertEqual(result, DedupResult(1, False))
        self.assertEqual(rt.pending_count("a"), 1)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["m1"])

    def test_duplicate_does_not_change_original_priority(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send_once("a", "k", "first", priority=5)
        rt.send("a", "plain-high", priority=4)
        # Neither the claimed priority nor the placeholder has any effect.
        rt.send_once("a", "k", "ignored", priority=-100)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["first", "plain-high"])
        self.assertEqual(
            [(t.message_id, t.priority) for t in rt.trace()],
            [(1, 5), (2, 4)],
        )

    def test_duplicate_after_completion_changes_nothing(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send_once("a", "k", "m1")
        rt.run()
        again = rt.send_once("a", "k", Uncopyable())
        self.assertEqual(again, DedupResult(1, False))
        self.assertEqual(rt.pending_count("a"), 0)
        self.assertEqual(rt.send("a", "next"), 2)  # no id was consumed
        rt.run()
        self.assertEqual(rt.get_state("a"), ["m1", "next"])
        self.assertEqual([t.message_id for t in rt.trace()], [1, 2])

    def test_duplicate_while_awaiting_failure_retry(self):
        attempts = {"n": 0}

        def handler(state, message, ctx):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("boom")
            return append_handler(state, message, ctx)

        rt = ActorRuntime()
        rt.register("a", [], handler)
        first = rt.send_once("a", "k", "m1")
        self.assertEqual(first, DedupResult(1, True))
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        self.assertEqual(caught.exception.message_id, 1)
        # The key reservation survives; resubmission is only a confirmation.
        again = rt.send_once("a", "k", Uncopyable())
        self.assertEqual(again, DedupResult(1, False))
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.run(), 1)
        # One successful handling -> at most one trace entry, one state
        # effect, and no id was spent on the duplicate.
        self.assertEqual(rt.get_state("a"), ["m1"])
        self.assertEqual([t.message_id for t in rt.trace()], [1])

    def test_records_cover_full_runtime_lifetime(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send_once("a", "k", "m1")
        rt.run()
        for _ in range(3):
            self.assertEqual(
                rt.send_once("a", "k", Uncopyable()),
                DedupResult(1, False),
            )
        self.assertEqual(rt.pending_count("a"), 0)
        self.assertEqual([t.message_id for t in rt.trace()], [1])


class SendOnceScopeTests(unittest.TestCase):
    def test_same_key_for_different_actors_is_independent(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.register("b", [], append_handler)
        first_a = rt.send_once("a", "shared", "to-a")
        first_b = rt.send_once("b", "shared", "to-b")
        self.assertEqual(first_a, DedupResult(1, True))
        self.assertEqual(first_b, DedupResult(2, True))
        self.assertEqual(rt.send_once("a", "shared", None),
                         DedupResult(1, False))
        self.assertEqual(rt.send_once("b", "shared", None),
                         DedupResult(2, False))
        rt.run()
        self.assertEqual(rt.get_state("a"), ["to-a"])
        self.assertEqual(rt.get_state("b"), ["to-b"])

    def test_plain_send_never_participates_in_dedup(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        self.assertEqual(rt.send("a", "m1"), 1)
        # A plain send with identical content does not reserve the key.
        self.assertEqual(rt.send_once("a", "k", "m1"),
                         DedupResult(2, True))
        rt.run()
        self.assertEqual(rt.get_state("a"), ["m1", "m1"])


class SendOnceValidationTests(unittest.TestCase):
    def _rt(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        return rt

    def test_priority_must_be_non_boolean_integer(self):
        rt = self._rt()
        for bad in (True, False, 1.5, "0", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.send_once("a", "k", None, priority=bad)
        # Nothing was reserved or enqueued.
        self.assertEqual(rt.pending_count("a"), 0)
        self.assertEqual(rt.send_once("a", "k", None), DedupResult(1, True))

    def test_delivery_key_must_be_string(self):
        rt = self._rt()
        for bad in (1, 1.0, b"k", None, ("k",), ["k"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.send_once("a", bad, None)

    def test_empty_delivery_key_raises_value_error(self):
        rt = self._rt()
        with self.assertRaises(ValueError):
            rt.send_once("a", "", None)

    def test_unknown_target_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            ActorRuntime().send_once("ghost", "k", None)

    def test_validation_precedes_dedup_lookup(self):
        rt = self._rt()
        rt.send_once("a", "k", "m1")
        # A duplicate with invalid arguments is rejected, never confirmed.
        with self.assertRaises(TypeError):
            rt.send_once("a", "k", None, priority=True)
        with self.assertRaises(ValueError):
            rt.send_once("a", "", None)
        # Unknown target is checked before the existing key could matter;
        # the same key string on a missing actor is a LookupError.
        with self.assertRaises(LookupError):
            rt.send_once("ghost", "k", None)
        # Bad priority outranks the unknown-target lookup.
        with self.assertRaises(TypeError):
            rt.send_once("ghost", "k", None, priority=True)
        # The valid duplicate still confirms the first id.
        self.assertEqual(rt.send_once("a", "k", None),
                         DedupResult(1, False))


class SendOnceCopyFailureTests(unittest.TestCase):
    def test_first_copy_failure_reserves_neither_key_nor_id(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        with self.assertRaises(ActorDataCopyError):
            rt.send_once("a", "k", Uncopyable())
        # No id consumed ...
        self.assertEqual(rt.send("a", "plain"), 1)
        # ... and the key is still free for a corrected first attempt,
        # which is accepted and takes the next id.
        result = rt.send_once("a", "k", "m1")
        self.assertEqual(result, DedupResult(2, True))
        rt.run()
        self.assertEqual(rt.get_state("a"), ["plain", "m1"])

    def test_copy_failure_checks_arguments_first(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        with self.assertRaises(TypeError):
            rt.send_once("a", "k", Uncopyable(), priority=1.5)
        with self.assertRaises(ValueError):
            rt.send_once("a", "", Uncopyable())
        with self.assertRaises(LookupError):
            rt.send_once("ghost", "k", Uncopyable())
        self.assertEqual(rt.send_once("a", "k", "ok"), DedupResult(1, True))


class SendOnceDeterminismTests(unittest.TestCase):
    def test_independent_runtimes_agree(self):
        def scenario():
            rt = ActorRuntime()
            rt.register("events", [], append_handler)
            rt.register("other", [], append_handler)
            results = []
            results.append(rt.send_once("events", "k1", "a", priority=2))
            results.append(rt.send_once("events", "k2", "b"))
            results.append(rt.send_once("events", "k1", Uncopyable()))
            results.append(rt.send_once("other", "k1", "c"))
            results.append(("plain-send", rt.send("events", "plain")))
            rt.run()
            results.append(rt.send_once("events", "k2", Uncopyable()))
            rt.run()
            return (
                [tuple(r) for r in results],
                rt.get_state("events"),
                rt.get_state("other"),
                [
                    (t.message_id, t.actor_name, t.priority)
                    for t in rt.trace()
                ],
            )

        self.assertEqual(scenario(), scenario())


if __name__ == "__main__":
    unittest.main()
