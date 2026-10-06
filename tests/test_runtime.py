"""Tests for the in-memory actor runtime."""
import copy
import unittest

from actor_runtime import (
    ActorContext,
    ActorDataCopyError,
    ActorExecutionError,
    ActorRuntime,
    DedupResult,
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

    def test_base_exception_keeps_message_pending(self):
        class CustomBase(BaseException):
            pass

        boom = CustomBase("stop")

        def handler(state, message, ctx):
            raise boom

        rt = ActorRuntime()
        rt.register("a", [], handler)
        mid = rt.send("a", "m", priority=7)
        rt.send("a", "later", priority=1)
        with self.assertRaises(CustomBase) as caught:
            rt.run()
        # The exact same exception object propagates, unwrapped.
        self.assertIs(caught.exception, boom)
        # The failing message stays pending and blocks what is behind it.
        self.assertEqual(rt.pending_count("a"), 2)
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.trace(), [])
        # Failing again keeps the same message at the head of the mailbox.
        with self.assertRaises(CustomBase):
            rt.run()
        self.assertEqual(rt.pending_count("a"), 2)

    def test_base_exception_commits_nothing_and_preserves_prior_results(self):
        flag = {"fail": False}

        def handler(state, message, ctx):
            state = list(state)
            state.append(message)
            if message == "boom":
                ctx.send("b", "derived")
                raise SystemExit(3)
            return state

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.register("b", [], append_handler)
        rt.send("a", "ok")
        rt.send("a", "boom")
        with self.assertRaises(SystemExit):
            rt.run()
        # Prior completion stays committed; the failed message commits no
        # state change, no derived delivery and no trace entry.
        self.assertEqual(rt.get_state("a"), ["ok"])
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.pending_count("b"), 0)
        self.assertEqual([t.message_id for t in rt.trace()], [1])

        # Recover: the handler stops failing, the same message is retried
        # and completes exactly once with continuous message ids.
        def recovering(state, message, ctx):
            state = list(state)
            state.append(message)
            if message == "boom" and flag["fail"]:
                ctx.send("b", "derived")
                raise KeyboardInterrupt
            if message == "boom":
                ctx.send("b", "derived")
            return state

        rt2 = ActorRuntime()
        rt2.register("a", [], recovering)
        rt2.register("b", [], append_handler)
        rt2.send("a", "ok")
        rt2.send("a", "boom")
        flag["fail"] = True
        with self.assertRaises(KeyboardInterrupt):
            rt2.run()
        flag["fail"] = False
        done = rt2.run()
        self.assertEqual(done, 2)  # retried "boom" plus its derived message
        self.assertEqual(rt2.get_state("a"), ["ok", "boom"])
        self.assertEqual(rt2.get_state("b"), ["derived"])
        self.assertEqual(rt2.pending_count("a"), 0)
        # The failed attempt consumed no message id and left no trace; the
        # retried message keeps its original id and the derived one follows.
        self.assertEqual([t.message_id for t in rt2.trace()], [1, 2, 3])
        self.assertEqual([t.actor_name for t in rt2.trace()], ["a", "a", "b"])

    def test_base_exception_failure_is_deterministic_across_runs(self):
        def build_and_run():
            flag = {"fail": True}

            def handler(state, message, ctx):
                state = dict(state)
                state["n"] = state.get("n", 0) + 1
                if message == "risky":
                    ctx.send("b", "go")
                    if flag["fail"]:
                        raise KeyboardInterrupt
                return state

            rt = ActorRuntime()
            rt.register("a", {}, handler)
            rt.register("b", [], append_handler)
            rt.send("a", "first", priority=2)
            rt.send("a", "risky", priority=1)
            with self.assertRaises(KeyboardInterrupt):
                rt.run()
            flag["fail"] = False
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


class CopyFailureTests(unittest.TestCase):
    """Copy failures at the isolation boundary are transactional."""

    @staticmethod
    def make_flaky(flag):
        """A value whose deepcopy fails while flag['fail'] is set."""

        class Flaky:
            def __init__(self, value):
                self.value = value

            def __deepcopy__(self, memo):
                if flag["fail"]:
                    raise RuntimeError("copy blocked")
                return Flaky(self.value)

            def __eq__(self, other):
                return isinstance(other, Flaky) and self.value == other.value

        return Flaky

    def test_send_copy_failure_raises_and_consumes_no_id(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        self.assertEqual(rt.send("a", "ok"), 1)
        boom = RuntimeError("no copy")

        class Bad:
            def __deepcopy__(self, memo):
                raise boom

        with self.assertRaises(ActorDataCopyError) as caught:
            rt.send("a", Bad())
        err = caught.exception
        self.assertIs(err.original, boom)
        self.assertIs(err.__cause__, boom)
        # Mailbox, trace, state and next message id are all unchanged.
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.send("a", "ok2"), 2)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["ok", "ok2"])

    def test_send_validation_precedes_copy(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)

        class Bad:
            def __deepcopy__(self, memo):
                raise RuntimeError("no copy")

        with self.assertRaises(TypeError):
            rt.send("a", Bad(), priority=1.5)
        with self.assertRaises(LookupError):
            rt.send("ghost", Bad())
        self.assertEqual(rt.send("a", "ok"), 1)

    def test_state_before_copy_failure_is_retryable(self):
        flag = {"fail": False}
        Flaky = self.make_flaky(flag)

        def handler(state, message, ctx):
            return Flaky(state.value + [message])

        rt = ActorRuntime()
        rt.register("a", Flaky([]), handler)
        first = rt.send("a", "m1", priority=5)
        rt.send("a", "m2", priority=1)
        flag["fail"] = True
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        err = caught.exception
        self.assertEqual(err.actor_name, "a")
        self.assertEqual(err.message_id, first)
        self.assertIsInstance(err.original, ActorDataCopyError)
        self.assertIsInstance(err.original.original, RuntimeError)
        # Nothing committed; both messages stay pending.
        self.assertEqual(rt.pending_count("a"), 2)
        self.assertEqual(rt.trace(), [])
        # Fixing the copy source lets the same messages complete once,
        # keeping the original priority and ids.
        flag["fail"] = False
        self.assertEqual(rt.get_state("a"), Flaky([]))
        self.assertEqual(rt.run(), 2)
        self.assertEqual(rt.get_state("a"), Flaky(["m1", "m2"]))
        self.assertEqual(
            [(t.message_id, t.priority) for t in rt.trace()],
            [(first, 5), (first + 1, 1)],
        )

    def test_handler_message_copy_failure_keeps_message_pending(self):
        flag = {"fail": False}
        Flaky = self.make_flaky(flag)
        seen = []

        def handler(state, message, ctx):
            seen.append(message.value)
            return state

        rt = ActorRuntime()
        rt.register("a", None, handler)
        mid = rt.send("a", Flaky("m"))
        flag["fail"] = True
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        self.assertIsInstance(caught.exception.original, ActorDataCopyError)
        self.assertEqual(caught.exception.message_id, mid)
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.trace(), [])
        self.assertEqual(seen, [])
        flag["fail"] = False
        self.assertEqual(rt.run(), 1)
        self.assertEqual(seen, ["m"])
        self.assertEqual(len(rt.trace()), 1)

    def test_returned_state_copy_failure_commits_nothing(self):
        flag = {"fail": False}
        Flaky = self.make_flaky(flag)
        received = []

        def handler(state, message, ctx):
            ctx.send("b", "derived")
            if flag["fail"]:
                return Flaky("uncopyable")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], handler)
        rt.register("b", [], lambda s, m, c: (received.append(m), s)[1])
        mid = rt.send("a", "m1")
        flag["fail"] = True
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        err = caught.exception
        self.assertEqual((err.actor_name, err.message_id), ("a", mid))
        self.assertIsInstance(err.original, ActorDataCopyError)
        # No state, no derived delivery, no trace, no consumed id.
        self.assertEqual(rt.get_state("a"), [])
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.pending_count("b"), 0)
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.send("a", "m2"), 2)
        # Retry after the fix commits exactly once per message.
        flag["fail"] = False
        self.assertEqual(rt.run(), 4)
        self.assertEqual(rt.get_state("a"), ["m1", "m2"])
        self.assertEqual(received, ["derived", "derived"])
        self.assertEqual([t.message_id for t in rt.trace()], [1, 2, 3, 4])

    def test_last_derived_copy_failure_hides_all_derived(self):
        flag = {"fail": True}
        Flaky = self.make_flaky(flag)
        received = []

        def handler(state, message, ctx):
            ctx.send("b", "d1")
            ctx.send("b", "d2")
            ctx.send("b", Flaky("d3") if flag["fail"] else "d3")
            return state

        rt = ActorRuntime()
        rt.register("a", None, handler)
        rt.register("b", [], lambda s, m, c: (received.append(m), s)[1])
        rt.send("a", "go")
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        self.assertIsInstance(caught.exception.original, ActorDataCopyError)
        # Even though only the last derived message cannot be copied, none
        # of them became visible and no id was consumed.
        self.assertEqual(rt.pending_count("b"), 0)
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.send("b", "external"), 2)
        # After the fix the retry delivers each derived message once.
        flag["fail"] = False
        rt.run()
        self.assertEqual(received, ["external", "d1", "d2", "d3"])
        self.assertEqual(rt.pending_count("a"), 0)

    def test_success_path_isolated_from_later_mutation(self):
        held = {}

        def handler(state, message, ctx):
            new_state = {"v": [1]}
            derived = {"payload": [1]}
            held["state"] = new_state
            held["derived"] = derived
            ctx.send("b", derived)
            return new_state

        rt = ActorRuntime()
        rt.register("a", None, handler)
        rt.register("b", None, lambda s, m, c: m)
        rt.send("a", "go")
        self.assertEqual(rt.run(limit=1), 1)
        # Mutating the objects the handler produced must not leak into the
        # runtime state, the queued derived message or the trace entry.
        held["state"]["v"].append(999)
        held["derived"]["payload"].append(999)
        self.assertEqual(rt.get_state("a"), {"v": [1]})
        self.assertEqual(rt.trace()[0].state_after, {"v": [1]})
        rt.run()
        self.assertEqual(rt.get_state("b"), {"payload": [1]})


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


class SendOnceDedupTests(unittest.TestCase):
    """Idempotent delivery via send_once / DedupResult."""

    @staticmethod
    def make_uncopyable(boom=RuntimeError("no copy")):
        class Bad:
            def __deepcopy__(self, memo):
                raise boom

            def __eq__(self, other):
                raise AssertionError("duplicate placeholder must not be compared")

        return Bad

    def test_first_call_accepted_like_send(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        result = rt.send_once("a", "key-1", "hello", priority=3)
        self.assertEqual(result, DedupResult(1, True))
        self.assertIsInstance(result, DedupResult)
        self.assertEqual(rt.pending_count("a"), 1)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["hello"])
        self.assertEqual(
            [tuple(e) for e in rt.trace()],
            [(1, "a", 3, [], ["hello"])],
        )

    def test_dedup_result_is_immutable(self):
        result = DedupResult(7, True)
        self.assertEqual((result.message_id, result.accepted), (7, True))
        with self.assertRaises(AttributeError):
            result.accepted = False

    def test_duplicate_while_pending_confirms_first_id(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        first = rt.send_once("a", "k", "first")
        self.assertEqual(first, DedupResult(1, True))
        # The placeholder must not be copied or compared.
        dup = rt.send_once("a", "k", self.make_uncopyable()(), priority=9)
        self.assertEqual(dup, DedupResult(1, False))
        # Nothing re-enqueued, no id consumed, priority of the original kept.
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.send("a", "plain"), 2)
        rt.run()
        self.assertEqual(
            [(t.message_id, t.priority) for t in rt.trace()],
            [(1, 0), (2, 0)],
        )

    def test_duplicate_message_never_reaches_handler(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(m), s)[1])
        rt.send_once("a", "k", {"v": 1})
        rt.send_once("a", "k", {"v": 2})
        rt.send_once("a", "k", {"v": 3})
        rt.run()
        self.assertEqual(seen, [{"v": 1}])
        self.assertEqual(len(rt.trace()), 1)

    def test_duplicate_after_completion_still_confirms(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        first = rt.send_once("a", "k", "m1")
        rt.run()
        dup = rt.send_once("a", "k", self.make_uncopyable()())
        self.assertEqual(dup.message_id, first.message_id)
        self.assertFalse(dup.accepted)
        self.assertEqual(rt.pending_count("a"), 0)
        self.assertEqual(rt.get_state("a"), ["m1"])
        self.assertEqual(len(rt.trace()), 1)
        # Dedup record lives for the whole runtime lifetime.
        self.assertEqual(rt.send_once("a", "k", "m2"), DedupResult(1, False))

    def test_duplicate_while_waiting_failure_retry_keeps_reservation(self):
        flag = {"fail": True}

        def handler(state, message, ctx):
            if flag["fail"]:
                raise RuntimeError("boom")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("a", [], handler)
        first = rt.send_once("a", "k", "original")
        self.assertEqual(first, DedupResult(1, True))
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        self.assertEqual(caught.exception.message_id, 1)
        # Key reservation survives the rollback: re-submit only confirms the
        # duplicate, the original stays pending and no id is consumed.
        dup = rt.send_once("a", "k", self.make_uncopyable()())
        self.assertEqual(dup, DedupResult(1, False))
        self.assertEqual(rt.pending_count("a"), 1)
        self.assertEqual(rt.send("a", "plain"), 2)
        self.assertEqual(rt.trace(), [])
        # The later run retries the original message; at most one trace entry
        # is ever produced for the idempotent delivery.
        flag["fail"] = False
        rt.run()
        self.assertEqual(rt.get_state("a"), ["original", "plain"])
        self.assertEqual(
            [(t.message_id, t.actor_name) for t in rt.trace()],
            [(1, "a"), (2, "a")],
        )
        self.assertEqual(rt.send_once("a", "k", "again"), DedupResult(1, False))

    def test_dedup_scope_is_target_and_key(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.register("b", [], append_handler)
        self.assertEqual(rt.send_once("a", "shared", "x"), DedupResult(1, True))
        self.assertEqual(rt.send_once("b", "shared", "x"), DedupResult(2, True))
        self.assertEqual(rt.send_once("a", "shared", "y"), DedupResult(1, False))
        self.assertEqual(rt.send_once("b", "shared", "y"), DedupResult(2, False))
        rt.run()
        self.assertEqual(rt.get_state("a"), ["x"])
        self.assertEqual(rt.get_state("b"), ["x"])

    def test_distinct_keys_distinct_deliveries(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        self.assertEqual(rt.send_once("a", "k1", "a"), DedupResult(1, True))
        self.assertEqual(rt.send_once("a", "k2", "b"), DedupResult(2, True))
        self.assertEqual(rt.send_once("a", "k1", "c"), DedupResult(1, False))
        self.assertEqual(rt.pending_count("a"), 2)

    def test_ids_interleave_with_send_on_global_sequence(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        self.assertEqual(rt.send("a", "p"), 1)
        self.assertEqual(rt.send_once("a", "k", "m"), DedupResult(2, True))
        self.assertEqual(rt.send_once("a", "k", "m"), DedupResult(2, False))
        self.assertEqual(rt.send("a", "p2"), 3)

    def test_duplicate_keeps_original_priority(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(m), s)[1])
        rt.send_once("a", "high", "high", priority=10)
        rt.send_once("a", "high", "ignored", priority=0)  # dup, no re-enqueue
        rt.send_once("a", "low", "low", priority=0)
        rt.run()
        self.assertEqual(seen, ["high", "low"])
        self.assertEqual(
            [t.priority for t in rt.trace()], [10, 0]
        )

    def test_first_call_copies_message_and_isolates_caller(self):
        payload = [1, 2]
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: m)
        result = rt.send_once("a", "k", payload)
        self.assertTrue(result.accepted)
        payload.append(3)
        rt.run()
        self.assertEqual(rt.get_state("a"), [1, 2])

    def test_priority_validation(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.send_once("a", "k", "m")
        for bad in (1.0, "1", None, [1], True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.send_once("a", "k", "m", priority=bad)

    def test_delivery_key_must_be_non_empty_string(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        for bad in (1, 1.0, None, b"k", ["k"], object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.send_once("a", bad, "m")
        with self.assertRaises(ValueError):
            rt.send_once("a", "", "m")

    def test_unknown_target_raises_lookup_error(self):
        rt = ActorRuntime()
        with self.assertRaises(LookupError):
            rt.send_once("ghost", "k", "m")

    def test_validation_order_precedes_dedup_lookup(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.send_once("a", "k", "m")
        # Priority is checked first, even with a non-string key.
        with self.assertRaises(TypeError):
            rt.send_once("a", 123, "m", priority=1.5)
        # Key type/emptiness is checked before the target exists.
        with self.assertRaises(TypeError):
            rt.send_once("ghost", 123, "m")
        with self.assertRaises(ValueError):
            rt.send_once("ghost", "", "m")
        # An existing key is still subject to priority validation, and a
        # rejected call changes nothing.
        with self.assertRaises(TypeError):
            rt.send_once("a", "k", "m", priority=True)
        self.assertEqual(rt.send_once("a", "k", "m"), DedupResult(1, False))

    def test_copy_failure_on_first_reserves_no_key_consumes_no_id(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        boom = RuntimeError("no copy")

        class Bad:
            def __deepcopy__(self, memo):
                raise boom

        with self.assertRaises(ActorDataCopyError) as caught:
            rt.send_once("a", "k", Bad())
        self.assertIs(caught.exception.original, boom)
        self.assertEqual(rt.pending_count("a"), 0)
        # No id consumed.
        self.assertEqual(rt.send("a", "plain"), 1)
        # Same key is still free: corrected retry is the first acceptance.
        self.assertEqual(rt.send_once("a", "k", "fixed"), DedupResult(2, True))
        rt.run()
        self.assertEqual(rt.get_state("a"), ["plain", "fixed"])

    def test_plain_send_does_not_deduplicate(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        self.assertEqual(rt.send_once("a", "k", "once"), DedupResult(1, True))
        self.assertEqual(rt.send("a", "once"), 2)
        self.assertEqual(rt.send("a", "once"), 3)
        rt.run()
        self.assertEqual(rt.get_state("a"), ["once", "once", "once"])

    def test_determinism_across_independent_runtimes(self):
        def build_and_run():
            rt = ActorRuntime()

            def handler(state, message, ctx):
                return list(state) + [message]

            rt.register("a", [], handler)
            rt.register("b", [], handler)
            results = []
            results.append(rt.send_once("a", "k1", "a1", priority=5))
            results.append(rt.send("b", "seed"))
            results.append(rt.send_once("a", "k1", "dup"))
            results.append(rt.send_once("b", "k1", "b1"))
            results.append(rt.send_once("a", "k2", "a2"))
            results.append(rt.send_once("b", "k1", "dup"))
            rt.run()
            results.append(rt.send_once("a", "k2", "late-dup"))
            return (
                [
                    tuple(r) if isinstance(r, DedupResult) else (r, None)
                    for r in results
                ],
                copy.deepcopy(rt.get_state("a")),
                copy.deepcopy(rt.get_state("b")),
                [tuple(e) for e in rt.trace()],
            )

        first = build_and_run()
        for _ in range(3):
            self.assertEqual(build_and_run(), first)
        expected_results = [
            (1, True), (2, None), (1, False), (3, True),
            (4, True), (3, False), (4, False),
        ]
        self.assertEqual(first[0], expected_results)


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
