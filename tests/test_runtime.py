"""Tests for the in-memory actor runtime."""
import copy
import unittest

from actor_runtime import (
    ActorContext,
    ActorExecutionError,
    ActorRuntime,
    TraceEntry,
)


def noop_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    state = list(state)
    state.append(message)
    return state


class RegistrationTests(unittest.TestCase):
    def test_empty_name_raises(self):
        rt = ActorRuntime()
        with self.assertRaises(ValueError):
            rt.register("", None, noop_handler)

    def test_duplicate_name_raises_and_keeps_original(self):
        rt = ActorRuntime()
        rt.register("a", 0, noop_handler)
        with self.assertRaises(ValueError):
            rt.register("a", 1, append_handler)
        self.assertEqual(rt.get_state("a"), 0)

    def test_send_unknown_raises_lookup_error(self):
        rt = ActorRuntime()
        with self.assertRaises(LookupError):
            rt.send("ghost", "hi")

    def test_get_state_unknown_raises(self):
        with self.assertRaises(LookupError):
            ActorRuntime().get_state("ghost")

    def test_pending_count_unknown_raises(self):
        with self.assertRaises(LookupError):
            ActorRuntime().pending_count("ghost")


class DeliveryTests(unittest.TestCase):
    def test_message_ids_monotonic(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        ids = [rt.send("a", i) for i in range(5)]
        self.assertEqual(ids, [1, 2, 3, 4, 5])

    def test_priority_must_be_integer(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        for bad in (1.0, "1", None, [1], True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.send("a", "m", priority=bad)

    def test_bad_delivery_consumes_no_id(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        self.assertEqual(rt.send("a", "ok"), 1)
        with self.assertRaises(TypeError):
            rt.send("a", "bad", priority=1.5)
        with self.assertRaises(LookupError):
            rt.send("ghost", "bad")
        self.assertEqual(rt.send("a", "ok2"), 2)

    def test_pending_count(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.send("a", 1)
        rt.send("a", 2)
        self.assertEqual(rt.pending_count("a"), 2)
        rt.run(limit=1)
        self.assertEqual(rt.pending_count("a"), 1)


class SchedulingTests(unittest.TestCase):
    def test_higher_priority_first(self):
        seen = []

        def rec(state, message, ctx):
            seen.append(message)
            return state

        rt = ActorRuntime()
        rt.register("a", None, rec)
        rt.send("a", "low", priority=0)
        rt.send("a", "high", priority=10)
        rt.send("a", "mid", priority=5)
        rt.run()
        self.assertEqual(seen, ["high", "mid", "low"])

    def test_equal_priority_fifo_by_id(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(m), s)[1])
        rt.send("a", "first", priority=3)
        rt.send("a", "second", priority=3)
        rt.send("a", "third", priority=3)
        rt.run()
        self.assertEqual(seen, ["first", "second", "third"])

    def test_actor_registration_order_selection(self):
        seen = []

        def make(tag):
            def handler(state, message, ctx):
                seen.append((tag, message))
                return state
            return handler

        rt = ActorRuntime()
        rt.register("first", None, make("f"))
        rt.register("second", None, make("s"))
        rt.send("second", "s1")
        rt.send("first", "f1")
        rt.send("second", "s2")
        rt.send("first", "f2")
        rt.run()
        # All of first's mailbox drains before second is ever selected.
        self.assertEqual(seen, [("f", "f1"), ("f", "f2"),
                                ("s", "s1"), ("s", "s2")])

    def test_limit_stops_processing(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        for i in range(5):
            rt.send("a", i)
        done = rt.run(limit=2)
        self.assertEqual(done, 2)
        self.assertEqual(rt.get_state("a"), [0, 1])
        self.assertEqual(rt.pending_count("a"), 3)
        done = rt.run()
        self.assertEqual(done, 3)
        self.assertEqual(rt.get_state("a"), [0, 1, 2, 3, 4])

    def test_bad_limit_consumes_nothing(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send("a", 1)
        for bad in (0, -1, 1.5, True, "2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rt.run(limit=bad)
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.trace(), [])

    def test_run_empty_returns_zero(self):
        self.assertEqual(ActorRuntime().run(), 0)

    def test_negative_priority_orders_below_default(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(m), s)[1])
        rt.send("a", "neg", priority=-5)
        rt.send("a", "zero")
        rt.run()
        self.assertEqual(seen, ["zero", "neg"])


class DerivedMessagesTests(unittest.TestCase):
    def test_derived_messages_enqueue_after_current_completes(self):
        order = []

        def a_handler(state, message, ctx: ActorContext):
            order.append(("a-in", message))
            ctx.send("b", f"from-a:{message}")
            order.append(("a-out", message))
            return state

        def b_handler(state, message, ctx):
            order.append(("b", message))
            return state

        rt = ActorRuntime()
        rt.register("a", None, a_handler)
        rt.register("b", None, b_handler)
        rt.send("a", 1)
        rt.run()
        self.assertEqual(
            order,
            [("a-in", 1), ("a-out", 1), ("b", "from-a:1")],
        )

    def test_send_to_unknown_from_handler_aborts(self):
        def bad(state, message, ctx):
            ctx.send("ghost", "x")
            return state

        rt = ActorRuntime()
        rt.register("a", None, bad)
        mid = rt.send("a", 1)
        with self.assertRaises(ActorExecutionError):
            rt.run()
        # Message stays unacknowledged; no trace entry.
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.trace(), [])

    def test_chained_delivery(self):
        def relay(state, message, ctx):
            nxt = message + 1
            if nxt <= 3:
                ctx.send("a", nxt)
            return message

        rt = ActorRuntime()
        rt.register("a", 0, relay)
        rt.send("a", 1)
        rt.run()
        self.assertEqual(rt.get_state("a"), 3)
        self.assertEqual([t.message_id for t in rt.trace()], [1, 2, 3])


class FailureTests(unittest.TestCase):
    def test_handler_exception_wrapped(self):
        class Boom(Exception):
            pass

        def bad(state, message, ctx):
            raise Boom("kaboom")

        rt = ActorRuntime()
        rt.register("a", None, bad)
        mid = rt.send("a", "m")
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        err = caught.exception
        self.assertEqual(err.actor_name, "a")
        self.assertEqual(err.message_id, mid)
        self.assertIsInstance(err.original, Boom)

    def test_failure_commits_nothing_keeps_prior_results(self):
        def handler(state, message, ctx):
            if message == "fail":
                raise RuntimeError("nope")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.send("a", "ok1")
        rt.send("a", "fail")
        rt.send("a", "ok2")
        with self.assertRaises(ActorExecutionError):
            rt.run()
        # First message committed; failing one stays in the mailbox and
        # blocks everything behind it.
        self.assertEqual(rt.get_state("a"), ["ok1"])
        self.assertEqual(rt.pending_count("a"), 2)
        self.assertEqual(len(rt.trace()), 1)

    def test_in_place_mutation_before_raise_is_rolled_back(self):
        def handler(state, message, ctx):
            state.append("dirty")
            if message == "fail":
                raise RuntimeError("nope")
            return state

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.send("a", "fail")
        with self.assertRaises(ActorExecutionError):
            rt.run()
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.pending_count("a"), 1)
        # Retrying the same message with a new runtime is still possible;
        # here verify re-run fails again without accumulating dirt.
        with self.assertRaises(ActorExecutionError):
            rt.run()
        self.assertEqual(rt.get_state("a"), [])

    def test_derived_messages_dropped_on_failure(self):
        received = []

        def bad(state, message, ctx):
            ctx.send("b", "derived")
            raise RuntimeError("nope")

        rt = ActorRuntime()
        rt.register("a", None, bad)
        rt.register("b", None, lambda s, m, c: (received.append(m), s)[1])
        rt.send("a", 1)
        with self.assertRaises(ActorExecutionError):
            rt.run()
        self.assertEqual(received, [])
        self.assertEqual(rt.pending_count("b"), 0)
        self.assertEqual(rt.pending_count("a"), 1)

    def test_base_exception_propagates_unwrapped(self):
        def handler(state, message, ctx):
            raise KeyboardInterrupt

        rt = ActorRuntime()
        rt.register("a", None, handler)
        rt.send("a", 1)
        with self.assertRaises(KeyboardInterrupt):
            rt.run()

    def test_base_exception_reraises_same_object(self):
        boom = KeyboardInterrupt("stop")

        def handler(state, message, ctx):
            raise boom

        rt = ActorRuntime()
        rt.register("a", None, handler)
        rt.send("a", 1)
        with self.assertRaises(KeyboardInterrupt) as caught:
            rt.run()
        self.assertIs(caught.exception, boom)

    def test_base_exception_keeps_message_pending(self):
        class CustomBase(BaseException):
            pass

        def handler(state, message, ctx):
            raise CustomBase()

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.send("a", "m", priority=7)
        with self.assertRaises(CustomBase):
            rt.run()
        # Nothing committed; the message is still pending.
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.trace(), [])

    def test_base_exception_rolls_back_mutation_and_derived(self):
        received = []

        def handler(state, message, ctx):
            state.append("dirty")
            ctx.send("b", "derived")
            raise SystemExit(1)

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.register("b", [], lambda s, m, c: (received.append(m), s)[1])
        rt.send("a", "m")
        with self.assertRaises(SystemExit):
            rt.run()
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(received, [])
        self.assertEqual(rt.pending_count("b"), 0)
        self.assertEqual(rt.trace(), [])

    def test_base_exception_preserves_prior_commits(self):
        def handler(state, message, ctx):
            if message == "fail":
                raise KeyboardInterrupt
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.send("a", "ok1")
        rt.send("a", "fail")
        rt.send("a", "ok2")
        with self.assertRaises(KeyboardInterrupt):
            rt.run()
        self.assertEqual(rt.get_state("a"), ["ok1"])
        self.assertEqual(rt.pending_count("a"), 2)
        self.assertEqual(len(rt.trace()), 1)

    def test_base_exception_retry_after_recovery(self):
        attempts = []

        def flaky(state, message, ctx):
            attempts.append(message)
            if len(attempts) == 1:
                raise KeyboardInterrupt
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], flaky)
        mid = rt.send("a", "m", priority=3)
        with self.assertRaises(KeyboardInterrupt):
            rt.run()
        self.assertEqual(rt.pending_count("a"), 1)
        # Same message retried with original id and priority; succeeds once.
        self.assertEqual(rt.run(), 1)
        self.assertEqual(rt.get_state("a"), ["m"])
        self.assertEqual(rt.pending_count("a"), 0)
        entries = rt.trace()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].message_id, mid)
        self.assertEqual(entries[0].priority, 3)

    def test_base_exception_does_not_consume_derived_ids(self):
        def bad(state, message, ctx):
            ctx.send("a", "derived")
            raise KeyboardInterrupt

        rt = ActorRuntime()
        rt.register("a", [], bad)
        rt.send("a", "m")
        with self.assertRaises(KeyboardInterrupt):
            rt.run()
        # The buffered derived send was dropped without consuming an id.
        self.assertEqual(rt.send("a", "next"), 2)

    def test_base_exception_determinism_across_reruns(self):
        def build_and_run():
            rt = ActorRuntime()
            attempts = []

            def flaky(state, message, ctx):
                if message == "flaky" and not attempts:
                    attempts.append(message)
                    ctx.send("b", "dropped")
                    raise KeyboardInterrupt
                if message == "flaky":
                    ctx.send("b", "derived")
                return list(state) + [message]

            rt.register("a", [], flaky)
            rt.register("b", [], append_handler)
            rt.send("a", "ok")
            rt.send("a", "flaky")
            with self.assertRaises(KeyboardInterrupt):
                rt.run()
            rt.run()
            return (
                rt.get_state("a"),
                rt.get_state("b"),
                rt.pending_count("a"),
                rt.pending_count("b"),
                [tuple(e) for e in rt.trace()],
            )

        first = build_and_run()
        for _ in range(3):
            self.assertEqual(build_and_run(), first)


class TraceTests(unittest.TestCase):
    def test_trace_fields_and_completion_order(self):
        rt = ActorRuntime()
        rt.register("a", 0, lambda s, m, c: s + m)
        id1 = rt.send("a", 1, priority=5)
        id2 = rt.send("a", 10)
        rt.run()
        entries = rt.trace()
        self.assertEqual(
            entries,
            [
                TraceEntry(id1, "a", 5, 0, 1),
                TraceEntry(id2, "a", 0, 1, 11),
            ],
        )

    def test_trace_and_queries_return_independent_copies(self):
        rt = ActorRuntime()
        rt.register("a", {"v": 0}, lambda s, m, c: s)
        rt.send("a", 1)
        rt.run()

        t1 = rt.trace()
        t1[0].state_after["v"] = 999
        t1.append("junk")
        self.assertEqual(rt.get_state("a"), {"v": 0})
        t2 = rt.trace()
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].state_after, {"v": 0})

        snap = rt.get_state("a")
        snap["v"] = 123
        self.assertEqual(rt.get_state("a"), {"v": 0})

    def test_initial_state_is_isolated_from_registration_input(self):
        initial = {"v": 1}
        rt = ActorRuntime()
        rt.register("a", initial, lambda s, m, c: s)
        initial["v"] = 42
        self.assertEqual(rt.get_state("a"), {"v": 1})

    def test_message_not_aliased_into_mailbox(self):
        payload = [1, 2]
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: m)
        rt.send("a", payload)
        payload.append(3)
        rt.run()
        self.assertEqual(rt.get_state("a"), [1, 2])


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_identical(self):
        def build_and_run():
            rt = ActorRuntime()

            def counter(state, message, ctx: ActorContext):
                state = dict(state)
                state["n"] = state.get("n", 0) + 1
                if state["n"] == 1:
                    ctx.send("b", "go")
                return state

            rt.register("a", {}, counter)
            rt.register("b", [], append_handler)
            rt.send("a", "x", priority=2)
            rt.send("b", "seed")
            rt.send("a", "y")
            rt.run()
            return (
                copy.deepcopy(rt.get_state("a")),
                copy.deepcopy(rt.get_state("b")),
                rt.pending_count("a"),
                rt.pending_count("b"),
                [tuple(e) for e in rt.trace()],
            )

        first = build_and_run()
        for _ in range(3):
            self.assertEqual(build_and_run(), first)


if __name__ == "__main__":
    unittest.main()
