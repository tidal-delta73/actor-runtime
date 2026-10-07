"""Tests for supervised actors: registration, failures and resolution."""
import copy
import hashlib
import json
import unittest

from actor_runtime import (
    ActorContext,
    ActorDataCopyError,
    ActorExecutionError,
    ActorRuntime,
    FailureRecord,
    SnapshotError,
    SupervisionError,
)


def noop_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    state = list(state)
    state.append(message)
    return state


def make_always_failing(exc=RuntimeError("boom")):
    def handler(state, message, ctx):
        raise exc
    return handler


def make_flaky(flag):
    """A value whose deepcopy fails while flag['fail'] is set."""

    class Flaky:
        def __init__(self, value):
            self.value = value

        def __deepcopy__(self, memo):
            if flag["fail"]:
                raise RuntimeError("copy blocked")
            return Flaky(list(self.value) if isinstance(self.value, list)
                         else self.value)

        def __eq__(self, other):
            return isinstance(other, Flaky) and self.value == other.value

    return Flaky


def tree_runtime():
    """A four-level chain: root -> mid -> low -> leaf, all registered."""
    rt = ActorRuntime()
    rt.register("root", [], append_handler)
    rt.register("mid", [], append_handler, supervisor="root")
    rt.register("low", [], append_handler, supervisor="mid")
    rt.register("leaf", [], append_handler, supervisor="low")
    return rt


class RegistrationTests(unittest.TestCase):
    def test_unsupervised_actors_are_roots_by_default(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        self.assertIsNone(rt._actors["a"].supervisor)

    def test_supervisor_must_be_string_or_none(self):
        rt = ActorRuntime()
        rt.register("sup", None, noop_handler)
        for bad in (1, 1.5, ["sup"], {"sup": 1}, True, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.register("child-" + str(bad), None, noop_handler,
                                supervisor=bad)

    def test_empty_supervisor_raises_value_error(self):
        rt = ActorRuntime()
        rt.register("sup", None, noop_handler)
        with self.assertRaises(ValueError):
            rt.register("child", None, noop_handler, supervisor="")

    def test_self_supervision_raises_value_error(self):
        rt = ActorRuntime()
        with self.assertRaises(ValueError):
            rt.register("solo", None, noop_handler, supervisor="solo")

    def test_unknown_supervisor_raises_lookup_error(self):
        rt = ActorRuntime()
        with self.assertRaises(LookupError):
            rt.register("child", None, noop_handler, supervisor="ghost")

    def test_name_checks_precede_supervisor_checks(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        # Empty name is rejected before supervisor type/emptiness.
        with self.assertRaises(ValueError):
            rt.register("", None, noop_handler, supervisor=123)
        # Duplicate name is rejected before the supervisor edge.
        with self.assertRaises(ValueError):
            rt.register("a", None, noop_handler, supervisor="ghost")

    def test_failed_registration_changes_nothing(self):
        rt = ActorRuntime()
        rt.register("a", 0, noop_handler)
        with self.assertRaises(TypeError):
            rt.register("b", 1, append_handler, supervisor=7)
        with self.assertRaises(ValueError):
            rt.register("c", 1, append_handler, supervisor="")
        with self.assertRaises(ValueError):
            rt.register("a", 1, append_handler, supervisor="a")
        with self.assertRaises(LookupError):
            rt.register("d", 1, append_handler, supervisor="ghost")
        # Only 'a' exists, its state is intact and the next order slot is 1.
        self.assertEqual(set(rt._actors), {"a"})
        self.assertEqual(rt.get_state("a"), 0)
        rt.register("b", 2, append_handler, supervisor="a")
        self.assertEqual(rt._actors["b"].order, 1)

    def test_supervision_edge_consumes_no_message_id(self):
        rt = tree_runtime()
        self.assertEqual(rt.send("root", "m"), 1)


class SchedulingNonInterferenceTests(unittest.TestCase):
    def test_tree_does_not_change_registration_order_scheduling(self):
        seen = []

        def rec(tag):
            def handler(state, message, ctx):
                seen.append(tag)
                return state
            return handler

        rt = ActorRuntime()
        rt.register("root", None, rec("root"))
        rt.register("mid", None, rec("mid"), supervisor="root")
        rt.register("leaf", None, rec("leaf"), supervisor="mid")
        rt.send("leaf", 1)
        rt.send("root", 1)
        rt.send("mid", 1)
        rt.run()
        # Supervision is not scheduling: registration order still decides.
        self.assertEqual(seen, ["root", "mid", "leaf"])


class FailureTests(unittest.TestCase):
    def test_root_failure_still_raises_execution_error(self):
        rt = ActorRuntime()
        rt.register("a", [], make_always_failing())
        mid = rt.send("a", "m")
        with self.assertRaises(ActorExecutionError) as caught:
            rt.run()
        self.assertEqual(caught.exception.message_id, mid)
        self.assertEqual(rt.pending_count("a"), 1)

    def test_supervised_failure_returns_completed_count_and_records(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send("root", "done")
        mid = rt.send("child", "boom")
        done = rt.run()
        self.assertEqual(done, 1)  # root completed, child failure ends the run
        records = rt.failures()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(
            record,
            FailureRecord(
                actor="child",
                message_id=mid,
                supervisor="root",
                error_type="RuntimeError",
                error_text="boom",
                supervision_path=("root",),
            ),
        )

    def test_record_is_immutable_and_query_returns_independent_list(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send("child", 1)
        rt.run()
        record = rt.failures()[0]
        with self.assertRaises(AttributeError):
            record.supervisor = "someone-else"
        first = rt.failures()
        first.clear()
        self.assertEqual(len(rt.failures()), 1)
        self.assertEqual(rt.failures()[0], record)

    def test_rollback_semantics_apply_to_supervised_failure(self):
        received = []

        def bad(state, message, ctx):
            ctx.send("other", "derived")
            raise RuntimeError("nope")

        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], bad, supervisor="root")
        rt.register("other", [],
                    lambda s, m, c: (received.append(m), s)[1])
        mid = rt.send("child", "fail", priority=4)
        self.assertEqual(rt.run(), 0)
        # Nothing committed: no state, derived delivery, trace or id; the
        # message keeps its id and priority; the actor is paused.
        self.assertEqual(rt.get_state("child"), [])
        self.assertEqual(rt.pending_count("child"), 1)
        self.assertEqual(rt.pending_count("other"), 0)
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.send("other", "after"), 2)
        self.assertTrue(rt._actors["child"].paused)

    def test_copy_failure_becomes_failure_record(self):
        flag = {"fail": False}
        Flaky = make_flaky(flag)

        def handler(state, message, ctx):
            return Flaky(state.value + [message])

        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", Flaky([]), handler, supervisor="root")
        mid = rt.send("child", "m")
        flag["fail"] = True
        done = rt.run()
        self.assertEqual(done, 0)
        record = rt.failures()[0]
        self.assertEqual(record.message_id, mid)
        self.assertEqual(record.error_type, "ActorDataCopyError")
        self.assertIn("deep-copy", record.error_text)
        self.assertEqual(rt.pending_count("child"), 1)
        # The same message retries after the copy source is fixed.
        flag["fail"] = False
        rt.resolve_failure("root", "child", mid, "retry")
        self.assertEqual(rt.run(), 1)
        self.assertEqual(rt.get_state("child"), Flaky(["m"]))

    def test_base_exception_propagates_unwrapped_for_supervised_actor(self):
        class Halt(BaseException):
            pass

        boom = Halt()

        def bad(state, message, ctx):
            raise boom

        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], bad, supervisor="root")
        rt.send("child", 1)
        with self.assertRaises(Halt) as caught:
            rt.run()
        self.assertIs(caught.exception, boom)
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["child"].paused)
        self.assertEqual(rt.pending_count("child"), 1)

    def test_paused_actor_keeps_receiving_mail_but_is_skipped(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send("child", "boom")
        rt.send("root", "r1")
        self.assertEqual(rt.run(), 1)  # child pauses, root completes
        # New deliveries (plain, idempotent, released timers) still arrive.
        rt.send("child", "later")
        self.assertEqual(
            rt.send_once("child", "k", "once").message_id, 4
        )
        mid_timer = rt.schedule("child", "timed", delay=2)
        result = rt.advance(2)
        self.assertEqual(result.released, (mid_timer,))
        # boom, later, once and the released timer all wait in the mailbox.
        self.assertEqual(rt.pending_count("child"), 4)
        # run skips the paused child and would have nothing else to do.
        self.assertEqual(rt.run(), 0)
        self.assertEqual(rt.run(limit=10), 0)
        # Root's mailbox still processed independently earlier; state stands.
        self.assertEqual(rt.get_state("root"), ["r1"])

    def test_other_actors_keep_running_in_later_runs(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.register("sibling", [], append_handler, supervisor="root")
        rt.send("child", "boom")
        rt.send("sibling", "s1")
        rt.send("sibling", "s2")
        self.assertEqual(rt.run(), 0)  # failure ends the run immediately
        self.assertEqual(rt.run(), 2)  # child skipped, siblings drain
        self.assertEqual(rt.get_state("sibling"), ["s1", "s2"])

    def test_pausing_supervisor_does_not_pause_its_children(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("mid", [], make_always_failing(), supervisor="root")
        rt.register("leaf", [], append_handler, supervisor="mid")
        rt.send("mid", "boom")
        rt.send("leaf", "l1")
        rt.run()  # mid selected first and pauses
        rt.run()  # the paused mid is skipped, its child leaf still runs
        self.assertTrue(rt._actors["mid"].paused)
        self.assertFalse(rt._actors["leaf"].paused)
        self.assertEqual(rt.get_state("leaf"), ["l1"])

    def test_failures_keep_production_order_and_resolve_independently(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("a", [], make_always_failing(RuntimeError("a-boom")),
                    supervisor="root")
        rt.register("b", [], make_always_failing(RuntimeError("b-boom")),
                    supervisor="root")
        rt.send("a", 1)
        rt.run()
        rt.send("b", 1)
        rt.run()
        records = rt.failures()
        self.assertEqual([r.actor for r in records], ["a", "b"])
        # Resolve b first; a's record stays put in first position.
        returned = rt.resolve_failure("root", "b", 2, "drop")
        self.assertEqual(returned.actor, "b")
        records = rt.failures()
        self.assertEqual([r.actor for r in records], ["a"])
        self.assertTrue(rt._actors["a"].paused)
        self.assertFalse(rt._actors["b"].paused)

    def test_failed_supervision_event_consumes_no_id(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("leaf", [], make_always_failing(), supervisor="root")
        self.assertEqual(rt.send("leaf", "boom"), 1)
        rt.run()
        # A root cannot escalate: the rejected attempts change nothing, and
        # neither the failure, its record nor the escalation attempts number
        # a message.
        with self.assertRaises(SupervisionError):
            rt.resolve_failure("root", "leaf", 1, "escalate")
        self.assertEqual(rt.failures()[0].message_id, 1)
        self.assertEqual(rt.send("root", "after"), 2)


class RetryTests(unittest.TestCase):
    def test_retry_resumes_and_retries_same_message(self):
        flag = {"fail": True}

        def handler(state, message, ctx):
            if flag["fail"]:
                raise RuntimeError("boom")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("child", [], handler, supervisor="root")
        mid = rt.send("child", "m", priority=3)
        rt.send("child", "later")
        rt.run()
        returned = rt.resolve_failure("root", "child", mid, "retry")
        self.assertEqual(returned.message_id, mid)
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["child"].paused)
        # Before the fix the run pauses again; after it the same id retries.
        rt.run()
        self.assertEqual(rt.failures()[0].message_id, mid)
        flag["fail"] = False
        rt.resolve_failure("root", "child", mid, "retry")
        self.assertEqual(rt.run(), 2)
        self.assertEqual(rt.get_state("child"), ["m", "later"])
        self.assertEqual(
            [(t.message_id, t.priority) for t in rt.trace()],
            [(mid, 3), (2, 0)],
        )

    def test_retry_keeps_priority_position(self):
        flag = {"fail": True}

        def handler(state, message, ctx):
            if message == "fail" and flag["fail"]:
                raise RuntimeError("boom")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], handler, supervisor="root")
        rt.send("child", "fail", priority=10)
        rt.send("child", "low", priority=0)
        rt.run()
        flag["fail"] = False
        rt.resolve_failure("root", "child", 1, "retry")
        rt.run()
        # The retried message kept its priority and goes first.
        self.assertEqual(rt.get_state("child"), ["fail", "low"])


class DropTests(unittest.TestCase):
    def test_drop_removes_failing_message_and_resumes(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], append_handler, supervisor="root")
        rt.send("child", "keep-1", priority=9)
        mid = rt.send("child", "fail", priority=0)

        def handler(state, message, ctx):
            if message == "fail":
                raise RuntimeError("boom")
            return append_handler(state, message, ctx)

        rt._actors["child"].handler = handler
        # Higher-priority keep-1 completes first, then the failure.
        self.assertEqual(rt.run(), 1)
        self.assertEqual(rt.failures()[0].message_id, mid)
        returned = rt.resolve_failure("root", "child", mid, "drop")
        self.assertEqual(returned.message_id, mid)
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["child"].paused)
        self.assertEqual(rt.pending_count("child"), 0)
        # No trace entry for the dropped id and no id consumption.
        self.assertEqual([t.message_id for t in rt.trace()], [1])
        self.assertEqual(rt.send("child", "after"), 3)
        rt.run()
        self.assertEqual(rt.get_state("child"), ["keep-1", "after"])

    def test_drop_retains_send_once_key(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        result = rt.send_once("child", "order-1", {"n": 1})
        self.assertTrue(result.accepted)
        rt.run()
        rt.resolve_failure("root", "child", result.message_id, "drop")
        # The key still confirms the dropped first id, it is not re-accepted.
        self.assertEqual(
            rt.send_once("child", "order-1", {"n": 2}),
            (result.message_id, False),
        )


class EscalationTests(unittest.TestCase):
    def test_escalate_moves_record_up_keeping_position_and_pause(self):
        rt = tree_runtime()
        rt._actors["leaf"].handler = make_always_failing()
        mid = rt.send("leaf", 1)
        rt.send("root", "r")
        rt.run()
        record = rt.failures()[0]
        self.assertEqual(record.supervision_path, ("root", "mid", "low"))
        moved = rt.resolve_failure("low", "leaf", mid, "escalate")
        self.assertEqual(moved.supervisor, "mid")
        self.assertEqual(moved.supervision_path, ("root", "mid"))
        self.assertEqual(moved.actor, "leaf")
        self.assertEqual(moved.message_id, mid)
        self.assertEqual(moved.error_type, "RuntimeError")
        self.assertEqual(moved.error_text, "boom")
        # In-place rewrite: same production slot; failed actor stays paused.
        self.assertEqual(len(rt.failures()), 1)
        self.assertTrue(rt._actors["leaf"].paused)
        self.assertEqual(rt.pending_count("leaf"), 1)
        # The old supervisor can no longer act; the new one can.
        with self.assertRaises(LookupError):
            rt.resolve_failure("low", "leaf", mid, "drop")
        moved = rt.resolve_failure("mid", "leaf", mid, "escalate")
        self.assertEqual(moved.supervisor, "root")
        self.assertEqual(moved.supervision_path, ("root",))

    def test_escalate_chain_to_root_then_supervision_error(self):
        rt = tree_runtime()
        rt._actors["leaf"].handler = make_always_failing()
        mid = rt.send("leaf", 1)
        rt.run()
        rt.resolve_failure("low", "leaf", mid, "escalate")
        rt.resolve_failure("mid", "leaf", mid, "escalate")
        record = rt.failures()[0]
        self.assertEqual(record.supervisor, "root")
        with self.assertRaises(SupervisionError):
            rt.resolve_failure("root", "leaf", mid, "escalate")
        # Root escalation changes nothing: record and pause retained.
        self.assertEqual(rt.failures(), [record])
        self.assertTrue(rt._actors["leaf"].paused)
        # It can still be retried/dropped by the root afterwards.
        rt.resolve_failure("root", "leaf", mid, "drop")
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["leaf"].paused)

    def test_supervisor_root_directly_supervising_raises_on_escalate(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send("child", 1)
        rt.run()
        with self.assertRaises(SupervisionError):
            rt.resolve_failure("root", "child", 1, "escalate")
        self.assertEqual(len(rt.failures()), 1)


class ResolutionValidationTests(unittest.TestCase):
    def _failing(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send("child", 1)
        rt.run()
        return rt

    def test_unknown_record_raises_lookup_error(self):
        rt = self._failing()
        with self.assertRaises(LookupError):
            rt.resolve_failure("root", "child", 999, "retry")
        with self.assertRaises(LookupError):
            rt.resolve_failure("root", "ghost", 1, "retry")
        self.assertEqual(len(rt.failures()), 1)

    def test_wrong_supervisor_identity_raises_lookup_error(self):
        rt = ActorRuntime()
        rt.register("root", [], noop_handler)
        rt.register("other", [], noop_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send("child", 1)
        rt.run()
        with self.assertRaises(LookupError):
            rt.resolve_failure("other", "child", 1, "drop")
        # Even the failed actor itself cannot resolve its own failure.
        with self.assertRaises(LookupError):
            rt.resolve_failure("child", "child", 1, "drop")
        self.assertEqual(rt.failures()[0].supervisor, "root")

    def test_invalid_action_raises_value_error(self):
        rt = self._failing()
        for bad in ("restart", "RESUME", "", "resume", None, 7):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rt.resolve_failure("root", "child", 1, bad)
        # A rejected action changes nothing and the record still resolves.
        self.assertEqual(len(rt.failures()), 1)
        self.assertTrue(rt._actors["child"].paused)
        rt.resolve_failure("root", "child", 1, "drop")
        self.assertEqual(rt.failures(), [])

    def test_resolving_twice_raises_lookup_error(self):
        rt = self._failing()
        rt.resolve_failure("root", "child", 1, "drop")
        with self.assertRaises(LookupError):
            rt.resolve_failure("root", "child", 1, "drop")

    def test_root_escalation_failure_changes_nothing(self):
        rt = self._failing()
        before = rt.failures()[0]
        with self.assertRaises(SupervisionError):
            rt.resolve_failure("root", "child", 1, "escalate")
        self.assertEqual(rt.failures(), [before])
        self.assertTrue(rt._actors["child"].paused)
        self.assertEqual(rt.pending_count("child"), 1)


# -- snapshots -------------------------------------------------------------


HANDLERS = {
    "root": append_handler,
    "mid": append_handler,
    "low": append_handler,
}


def reseal(payload):
    """Re-serialise a modified payload with a fresh valid digest."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    envelope = {"payload": payload, "digest": hashlib.sha256(canonical).hexdigest()}
    return json.dumps(
        envelope, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


class SupervisionSnapshotTests(unittest.TestCase):
    def test_pending_failure_round_trip(self):
        flag = {"fail": True}
        Flaky = make_flaky(flag)

        def handler(state, message, ctx):
            if flag["fail"]:
                raise ValueError("nope")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("mid", [], append_handler, supervisor="root")
        rt.register("leaf", [], handler, supervisor="mid")
        rt.send("leaf", "boom", priority=4)
        rt.send("root", "r1")
        rt.run()
        rt.send("leaf", "while-paused")
        rt.schedule("leaf", "timed", delay=5)

        data = rt.export_snapshot()
        handlers = {"root": append_handler, "mid": append_handler,
                    "leaf": handler}
        restored = ActorRuntime.restore_snapshot(data, handlers)
        self.assertEqual(restored.failures(), rt.failures())
        self.assertTrue(restored._actors["leaf"].paused)
        self.assertEqual(restored.pending_count("leaf"), 2)
        self.assertEqual(restored.scheduled_count("leaf"), 1)
        # Same action sequence and inputs on both runtimes agree everywhere.
        flag["fail"] = False
        for runtime in (rt, restored):
            runtime.resolve_failure("mid", "leaf", 1, "retry")
            runtime.advance(5)
            runtime.run()
        self.assertEqual(restored.get_state("leaf"), rt.get_state("leaf"))
        self.assertEqual(restored.get_state("root"), rt.get_state("root"))
        self.assertEqual(restored.trace(), rt.trace())
        self.assertEqual(restored.export_snapshot(), rt.export_snapshot())

    def test_escalated_record_round_trip(self):
        rt = tree_runtime()
        rt._actors["leaf"].handler = make_always_failing()
        rt.send("leaf", 1)
        rt.run()
        rt.resolve_failure("low", "leaf", 1, "escalate")
        data = rt.export_snapshot()
        handlers = {name: append_handler for name in ("root", "mid", "low")}
        handlers["leaf"] = make_always_failing()
        restored = ActorRuntime.restore_snapshot(data, handlers)
        record = restored.failures()[0]
        self.assertEqual(record.supervisor, "mid")
        self.assertEqual(record.supervision_path, ("root", "mid"))
        self.assertTrue(restored._actors["leaf"].paused)
        # mid -> root works, then root escalation errors; same actions on
        # the original runtime produce identical results.
        for runtime in (rt, restored):
            runtime.resolve_failure("mid", "leaf", 1, "escalate")
            with self.assertRaises(SupervisionError):
                runtime.resolve_failure("root", "leaf", 1, "escalate")
            runtime.resolve_failure("root", "leaf", 1, "drop")
        self.assertEqual(restored.failures(), rt.failures())
        self.assertEqual(restored.export_snapshot(), rt.export_snapshot())

    def test_drop_round_trip_keeps_dedup_key_and_numbering(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("child", [], make_always_failing(), supervisor="root")
        rt.send_once("child", "k", "boom")
        rt.run()
        rt.resolve_failure("root", "child", 1, "drop")
        restored = ActorRuntime.restore_snapshot(
            rt.export_snapshot(),
            {"root": append_handler, "child": make_always_failing()},
        )
        self.assertEqual(restored.failures(), [])
        self.assertFalse(restored._actors["child"].paused)
        self.assertEqual(restored.pending_count("child"), 0)
        self.assertEqual(
            restored.send_once("child", "k", "again"), (1, False)
        )
        self.assertEqual(restored.send("root", "fresh"), 2)

    def test_failure_order_survives_snapshot(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("a", [], make_always_failing(RuntimeError("a")),
                    supervisor="root")
        rt.register("b", [], make_always_failing(RuntimeError("b")),
                    supervisor="root")
        rt.send("a", 1)
        rt.run()
        rt.send("b", 1)
        rt.run()
        restored = ActorRuntime.restore_snapshot(
            rt.export_snapshot(),
            {"root": append_handler,
             "a": make_always_failing(RuntimeError("a")),
             "b": make_always_failing(RuntimeError("b"))},
        )
        self.assertEqual(
            [(r.actor, r.supervisor) for r in restored.failures()],
            [("a", "root"), ("b", "root")],
        )

    def test_identical_supervised_state_exports_identical_bytes(self):
        def build():
            rt = tree_runtime()
            rt._actors["leaf"].handler = make_always_failing()
            rt.send("leaf", 1)
            rt.run()
            rt.resolve_failure("low", "leaf", 1, "escalate")
            return rt

        data = build().export_snapshot()
        for _ in range(3):
            self.assertEqual(build().export_snapshot(), data)

    def test_legacy_snapshot_without_supervision_fields_restores(self):
        # Bytes written before supervision existed simply omit the new
        # fields; stripping them from current bytes reproduces them.
        rt = ActorRuntime()
        rt.register("alpha", {"count": 0},
                    lambda s, m, c: dict(s, count=s["count"] + 1))
        rt.register("beta", [], append_handler)
        rt.send("alpha", "first", priority=5)
        rt.send("beta", "second")
        rt.run()
        envelope = json.loads(rt.export_snapshot().decode("ascii"))
        payload = envelope["payload"]
        for actor in payload["actors"]:
            actor.pop("supervisor")
            actor.pop("paused")
        payload.pop("failures")
        payload.pop("dropped_message_ids")
        restored = ActorRuntime.restore_snapshot(
            reseal(payload),
            {"alpha": lambda s, m, c: dict(s, count=s["count"] + 1),
             "beta": append_handler},
        )
        # Legacy actors are roots: an ordinary handler failure raises.
        restored._actors["alpha"].handler = make_always_failing()
        restored.send("alpha", "boom")
        with self.assertRaises(ActorExecutionError):
            restored.run()

    def test_tampered_paused_without_record_rejected(self):
        rt = tree_runtime()
        rt._actors["leaf"].handler = make_always_failing()
        rt.send("leaf", 1)
        rt.run()
        payload = json.loads(rt.export_snapshot().decode("ascii"))["payload"]
        payload["failures"] = []
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(
                reseal(payload),
                {name: append_handler for name in ("root", "mid", "low")}
                | {"leaf": make_always_failing()},
            )

    def test_tampered_failure_supervisor_rejected(self):
        rt = tree_runtime()
        rt._actors["leaf"].handler = make_always_failing()
        rt.send("leaf", 1)
        rt.run()
        payload = json.loads(rt.export_snapshot().decode("ascii"))["payload"]
        # Detach the leaf from the chain but keep the failure holder "low".
        for actor in payload["actors"]:
            if actor["name"] == "leaf":
                actor["supervisor"] = "root"
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(
                reseal(payload),
                {name: append_handler for name in ("root", "mid", "low")}
                | {"leaf": make_always_failing()},
            )

    def test_tampered_failure_message_missing_rejected(self):
        rt = tree_runtime()
        rt._actors["leaf"].handler = make_always_failing()
        rt.send("leaf", 1)
        rt.run()
        payload = json.loads(rt.export_snapshot().decode("ascii"))["payload"]
        for actor in payload["actors"]:
            if actor["name"] == "leaf":
                actor["mailbox"] = []
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(
                reseal(payload),
                {name: append_handler for name in ("root", "mid", "low")}
                | {"leaf": make_always_failing()},
            )

    def test_tampered_dropped_live_id_rejected(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("child", [], append_handler, supervisor="root")
        rt.send("root", "live")
        payload = json.loads(rt.export_snapshot().decode("ascii"))["payload"]
        payload["dropped_message_ids"] = [1]  # id 1 sits in root's mailbox
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(
                reseal(payload),
                {"root": append_handler, "child": append_handler},
            )


if __name__ == "__main__":
    unittest.main()
