"""Tests for the logical clock and expiring timed deliveries."""
import copy
import unittest

from actor_runtime import (
    ActorDataCopyError,
    ActorRuntime,
    AdvanceResult,
)


def noop_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    return list(state) + [message]


class ClockTests(unittest.TestCase):
    def test_clock_starts_at_zero(self):
        rt = ActorRuntime()
        self.assertEqual(rt.clock(), 0)

    def test_send_run_and_schedule_do_not_move_the_clock(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.send("a", 1)
        rt.schedule("a", 2, delay=5)
        rt.run()
        self.assertEqual(rt.clock(), 0)

    def test_advance_returns_current_time_in_immutable_result(self):
        rt = ActorRuntime()
        result = rt.advance(3)
        self.assertIsInstance(result, AdvanceResult)
        self.assertEqual(result.time, 3)
        self.assertEqual(result.released, ())
        self.assertEqual(result.expired, ())
        self.assertEqual(rt.clock(), 3)
        # The id sequences are immutable tuples.
        self.assertIsInstance(result.released, tuple)
        self.assertIsInstance(result.expired, tuple)
        with self.assertRaises(AttributeError):
            result.released = (1,)
        with self.assertRaises(TypeError):
            result.expired[0] = 1

    def test_advance_is_cumulative(self):
        rt = ActorRuntime()
        rt.advance(2)
        self.assertEqual(rt.advance(4).time, 6)
        self.assertEqual(rt.clock(), 6)

    def test_advance_requires_positive_non_boolean_integer(self):
        rt = ActorRuntime()
        for bad in (0, -1, -10):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rt.advance(bad)
        for bad in (1.0, 1.5, "1", None, [1], True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.advance(bad)
        # A failed advance moves neither clock nor queues.
        self.assertEqual(rt.clock(), 0)


class ScheduleValidationTests(unittest.TestCase):
    def setUp(self):
        self.rt = ActorRuntime()
        self.rt.register("a", None, noop_handler)

    def test_schedule_ids_share_the_global_monotonic_sequence(self):
        ids = [
            self.rt.send("a", "s0"),
            self.rt.schedule("a", "d1", delay=2),
            self.rt.schedule("a", "d2", delay=1, ttl=5, priority=3),
            self.rt.send("a", "s3", priority=9),
        ]
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_unknown_target_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            ActorRuntime().schedule("ghost", "m", delay=1)

    def test_scheduled_count_unknown_actor_raises(self):
        with self.assertRaises(LookupError):
            ActorRuntime().scheduled_count("ghost")

    def test_priority_must_be_non_boolean_integer(self):
        for bad in (1.0, "1", None, [1], True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.rt.schedule("a", "m", delay=1, priority=bad)

    def test_delay_must_be_non_negative_non_boolean_integer(self):
        for bad in (1.0, "1", None, [1], True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.rt.schedule("a", "m", delay=bad)
        for bad in (-1, -10):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.rt.schedule("a", "m", delay=bad)

    def test_ttl_must_be_none_or_positive_non_boolean_integer(self):
        for bad in (1.0, "1", [], True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.rt.schedule("a", "m", delay=1, ttl=bad)
        for bad in (0, -1, -5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.rt.schedule("a", "m", delay=1, ttl=bad)

    def test_validation_precedes_target_lookup_and_copy(self):
        class Bad:
            def __deepcopy__(self, memo):
                raise RuntimeError("no copy")

        for kwargs in (
            {"delay": 1, "priority": 1.5},
            {"delay": 1.5},
            {"delay": 1, "ttl": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    self.rt.schedule("ghost", Bad(), **kwargs)

    def test_failed_schedule_consumes_no_id_and_queues_nothing(self):
        class Bad:
            def __deepcopy__(self, memo):
                raise RuntimeError("no copy")

        self.assertEqual(self.rt.send("a", "ok"), 1)
        with self.assertRaises(ActorDataCopyError):
            self.rt.schedule("a", Bad(), delay=1)
        with self.assertRaises(LookupError):
            self.rt.schedule("ghost", "x", delay=1)
        self.assertEqual(self.rt.send("a", "ok2"), 2)
        self.assertEqual(self.rt.scheduled_count("a"), 0)

    def test_failed_advance_consumes_nothing(self):
        self.rt.schedule("a", "m", delay=2)
        with self.assertRaises(ValueError):
            self.rt.advance(0)
        with self.assertRaises(TypeError):
            self.rt.advance(True)
        self.assertEqual(self.rt.clock(), 0)
        self.assertEqual(self.rt.scheduled_count("a"), 1)
        self.assertEqual(self.rt.pending_count("a"), 0)


class TimedDeliveryTests(unittest.TestCase):
    def test_zero_delay_enters_mailbox_immediately(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        mid = rt.schedule("a", "now", delay=0, ttl=1)
        self.assertEqual(mid, 1)
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.scheduled_count("a"), 0)
        result = rt.advance(1)
        self.assertEqual(result.released, ())
        self.assertEqual(result.expired, ())

    def test_delayed_message_stays_out_of_mailbox_until_deadline(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        mid = rt.schedule("a", "m", delay=5)
        self.assertEqual(rt.pending_count("a"), 0)
        self.assertEqual(rt.scheduled_count("a"), 1)
        self.assertEqual(rt.advance(4), AdvanceResult(4, (), ()))
        self.assertEqual(rt.scheduled_count("a"), 1)
        result = rt.advance(1)
        self.assertEqual(result, AdvanceResult(5, (mid,), ()))
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.scheduled_count("a"), 0)

    def test_delay_is_counted_from_the_current_tick(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.advance(10)
        mid = rt.schedule("a", "m", delay=3)
        self.assertEqual(rt.advance(2), AdvanceResult(12, (), ()))
        self.assertEqual(rt.advance(1), AdvanceResult(13, (mid,), ()))

    def test_scheduled_count_is_per_actor(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.register("b", None, noop_handler)
        rt.schedule("a", 1, delay=2)
        rt.schedule("a", 2, delay=3)
        rt.schedule("b", 3, delay=2)
        self.assertEqual(rt.scheduled_count("a"), 2)
        self.assertEqual(rt.scheduled_count("b"), 1)
        rt.advance(2)
        self.assertEqual(rt.scheduled_count("a"), 1)
        self.assertEqual(rt.scheduled_count("b"), 0)
        # Released mail counts as pending, never as scheduled.
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.pending_count("b"), 1)

    def test_released_message_keeps_scheduled_id_and_priority(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(m), s)[1])
        rt.send("a", "low")                                   # id 1, prio 0
        hi = rt.schedule("a", "high", delay=1, priority=10)   # id 2, prio 10
        result = rt.advance(1)
        self.assertEqual(result.released, (hi,))
        rt.run()
        self.assertEqual(seen, ["high", "low"])

    def test_releases_mix_with_existing_mail_by_registration_order(self):
        seen = []

        def make(tag):
            def handler(state, message, ctx):
                seen.append((tag, message))
                return state
            return handler

        rt = ActorRuntime()
        rt.register("first", None, make("f"))
        rt.register("second", None, make("s"))
        rt.schedule("second", "s1", delay=1)
        rt.schedule("first", "f1", delay=1)
        rt.advance(1)
        rt.run()
        self.assertEqual(seen, [("f", "f1"), ("s", "s1")])

    def test_release_without_run_does_not_invoke_handler_or_trace(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", [], lambda s, m, c: (seen.append(m), s + [m])[1])
        mid = rt.schedule("a", "m", delay=1)
        rt.advance(1)
        self.assertEqual(seen, [])
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.get_state("a"), [])
        # Processing succeeds afterwards under the ordinary run contract and
        # then produces a normal trace entry carrying the schedule id.
        rt.run()
        self.assertEqual(seen, ["m"])
        self.assertEqual([t.message_id for t in rt.trace()], [mid])
        self.assertEqual([t.priority for t in rt.trace()], [0])

    def test_scheduled_message_is_deep_copied(self):
        payload = [1, 2]
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: m)
        rt.schedule("a", payload, delay=1)
        payload.append(3)
        rt.advance(1)
        rt.run()
        self.assertEqual(rt.get_state("a"), [1, 2])


class ExpiryTests(unittest.TestCase):
    def test_message_expiring_at_deadline_never_enters_mailbox(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", [], lambda s, m, c: (seen.append(m), s + [m])[1])
        mid = rt.schedule("a", "gone", delay=3, ttl=3)
        before = rt.advance(2)
        self.assertEqual(before, AdvanceResult(2, (), ()))
        result = rt.advance(1)
        self.assertEqual(result, AdvanceResult(3, (), (mid,)))
        self.assertEqual(rt.pending_count("a"), 0)
        self.assertEqual(rt.scheduled_count("a"), 0)
        rt.run()
        self.assertEqual(seen, [])
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.trace(), [])

    def test_message_released_before_ttl_survives_a_later_expiry_tick(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        mid = rt.schedule("a", "m", delay=5, ttl=10)
        # A single jump crosses the deadline (5) and then the expiry tick
        # (10): the delivery was released at tick 5, so it delivers.
        result = rt.advance(10)
        self.assertEqual(result, AdvanceResult(10, (mid,), ()))
        rt.run()
        self.assertEqual(rt.get_state("a"), ["m"])

    def test_advance_just_to_expiry_after_release_keeps_mail(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        mid = rt.schedule("a", "m", delay=5, ttl=6)
        self.assertEqual(rt.advance(5), AdvanceResult(5, (mid,), ()))
        self.assertEqual(rt.advance(1), AdvanceResult(6, (), ()))
        self.assertEqual(rt.pending_count("a"), 1)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["m"])

    def test_expiry_does_not_consume_future_ids(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        stale = rt.schedule("a", "stale", delay=1, ttl=1)
        rt.advance(1)
        # The expired id is simply gone from the sequence; the next id is
        # fresh and higher.
        self.assertEqual(rt.send("a", "next"), stale + 1)


class AdvanceOrderingTests(unittest.TestCase):
    def test_events_stably_ordered_by_tick_then_id(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.register("b", None, noop_handler)
        m1 = rt.schedule("a", 1, delay=2)
        m2 = rt.schedule("a", 2, delay=2, ttl=2)   # expires tick 2
        m3 = rt.schedule("b", 3, delay=2, ttl=2)   # expires tick 2
        m4 = rt.schedule("a", 4, delay=1)
        m5 = rt.schedule("b", 5, delay=1, ttl=1)   # expires tick 1
        result = rt.advance(2)
        self.assertEqual(result.time, 2)
        # Released: tick 1 -> m4, tick 2 -> m1.
        self.assertEqual(result.released, (m4, m1))
        # Expired: tick 1 -> m5, tick 2 -> m2 then m3 by id.
        self.assertEqual(result.expired, (m5, m2, m3))

    def test_releases_in_one_advance_respect_priority_not_release_order(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(m), s)[1])
        rt.schedule("a", "low", delay=1, priority=0)
        rt.schedule("a", "high", delay=1, priority=10)
        rt.advance(1)
        rt.run()
        self.assertEqual(seen, ["high", "low"])

    def test_repeated_advances_are_stable(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        m1 = rt.schedule("a", 1, delay=1)
        m2 = rt.schedule("a", 2, delay=2)
        m3 = rt.schedule("a", 3, delay=2, ttl=2)
        self.assertEqual(rt.advance(1), AdvanceResult(1, (m1,), ()))
        self.assertEqual(rt.advance(1), AdvanceResult(2, (m2,), (m3,)))
        self.assertEqual(rt.advance(1), AdvanceResult(3, (), ()))


class ClockDeterminismTests(unittest.TestCase):
    def test_identical_sequences_replay_identically(self):
        def build_and_run():
            rt = ActorRuntime()
            rt.register("a", [], append_handler)
            rt.register("b", [], append_handler)
            rt.send("a", "seed")
            rt.schedule("b", "t1", delay=2, ttl=10, priority=4)
            rt.schedule("a", "t2", delay=3)
            rt.schedule("a", "stale", delay=3, ttl=3)
            advances = [rt.advance(2), rt.advance(1)]
            rt.run()
            rt.schedule("b", "t3", delay=1)
            rt.advance(1)
            rt.run()
            return (
                copy.deepcopy(rt.clock()),
                copy.deepcopy([tuple(r) for r in advances]),
                copy.deepcopy(rt.get_state("a")),
                copy.deepcopy(rt.get_state("b")),
                copy.deepcopy(rt.pending_count("a")),
                copy.deepcopy(rt.pending_count("b")),
                [tuple(e) for e in rt.trace()],
            )

        first = build_and_run()
        for _ in range(3):
            self.assertEqual(build_and_run(), first)

    def test_advance_results_match_across_independent_runtimes(self):
        def drive():
            rt = ActorRuntime()
            rt.register("a", [], append_handler)
            results = []
            for delay, ttl in ((1, None), (2, 2), (3, 10), (1, 1)):
                rt.schedule("a", f"m{delay}-{ttl}", delay=delay, ttl=ttl)
            results.append(rt.advance(2))
            results.append(rt.advance(3))
            rt.run()
            return [tuple(r) for r in results], rt.get_state("a"), \
                [e.message_id for e in rt.trace()]

        self.assertEqual(drive(), drive())


if __name__ == "__main__":
    unittest.main()
