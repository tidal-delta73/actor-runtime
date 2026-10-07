"""Tests for the supervision tree: registration, failures and snapshots."""
import copy
import hashlib
import json
import unittest

from actor_runtime import (
    ActorExecutionError,
    ActorRuntime,
    DedupResult,
    FailureRecord,
    SnapshotError,
    SupervisionError,
)


def noop_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    return list(state) + [message]


def make_failing_on(predicate):
    """Handler appending every message, raising on predicate(message)."""

    def handler(state, message, ctx):
        if predicate(message):
            raise RuntimeError(f"boom:{message}")
        return list(state) + [message]

    return handler


fail_handler = make_failing_on(lambda m: m == "fail")


def reseal(data, transform):
    """Tamper with a snapshot payload and reseal an intact digest."""
    envelope = json.loads(data.decode("ascii"))
    transform(envelope["payload"])
    canonical = json.dumps(
        envelope["payload"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")
    envelope["digest"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(
        envelope, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def build_chain(worker_handler=fail_handler):
    """root -> mid -> worker(root registered first); return the runtime."""
    rt = ActorRuntime()
    rt.register("root", [], append_handler)
    rt.register("mid", [], append_handler, supervisor="root")
    rt.register("worker", [], worker_handler, supervisor="mid")
    return rt


class SupervisorRegistrationTests(unittest.TestCase):
    def test_supervisor_must_be_string_or_none(self):
        rt = ActorRuntime()
        rt.register("p", None, noop_handler)
        for bad in (1, 1.0, True, b"p", ["p"], object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    rt.register("c", None, noop_handler, supervisor=bad)

    def test_empty_supervisor_raises_value_error(self):
        rt = ActorRuntime()
        rt.register("p", None, noop_handler)
        with self.assertRaises(ValueError):
            rt.register("c", None, noop_handler, supervisor="")

    def test_self_supervision_raises_value_error(self):
        rt = ActorRuntime()
        with self.assertRaises(ValueError):
            rt.register("solo", None, noop_handler, supervisor="solo")

    def test_unknown_supervisor_raises_lookup_error(self):
        rt = ActorRuntime()
        with self.assertRaises(LookupError):
            rt.register("c", None, noop_handler, supervisor="ghost")

    def test_failed_registration_stores_nothing(self):
        rt = ActorRuntime()
        rt.register("p", 0, noop_handler)
        with self.assertRaises(LookupError):
            rt.register("c", 1, noop_handler, supervisor="ghost")
        with self.assertRaises(LookupError):
            rt.get_state("c")
        # The rejected state did not replace anything and the actor is not
        # a valid delivery target.
        self.assertEqual(rt.get_state("p"), 0)
        with self.assertRaises(LookupError):
            rt.send("c", "m")

    def test_failed_registration_preserves_order(self):
        seen = []
        rt = ActorRuntime()
        rt.register("a", None, lambda s, m, c: (seen.append(("a", m)), s)[1])
        with self.assertRaises(LookupError):
            rt.register("b", None, noop_handler, supervisor="ghost")
        rt.register("c", None, lambda s, m, c: (seen.append(("c", m)), s)[1])
        rt.send("c", 1)
        rt.send("a", 2)
        rt.run()
        # "b" never existed; selection is a then c in registration order.
        self.assertEqual(seen, [("a", 2), ("c", 1)])

    def test_default_is_root_and_baseline_behaviour_kept(self):
        rt = ActorRuntime()
        rt.register("a", [], fail_handler)
        rt.send("a", "fail")
        with self.assertRaises(ActorExecutionError):
            rt.run()
        self.assertEqual(rt.failures(), [])

    def test_supervision_chain_registration(self):
        rt = build_chain()
        self.assertEqual(
            rt._actors["worker"].supervisor_name, "mid"
        )
        self.assertEqual(
            rt._supervision_path(rt._actors["worker"]),
            ("worker", "mid", "root"),
        )


class FailureRecordTests(unittest.TestCase):
    def test_failure_record_is_immutable_namedtuple(self):
        record = FailureRecord("c", 3, "p", "ValueError", "x", ("c", "p"))
        self.assertEqual(
            record,
            FailureRecord("c", 3, "p", "ValueError", "x", ("c", "p")),
        )
        self.assertEqual(record.actor_name, "c")
        self.assertEqual(record.message_id, 3)
        self.assertEqual(record.supervisor, "p")
        self.assertEqual(record.error_type, "ValueError")
        self.assertEqual(record.error_text, "x")
        self.assertEqual(record.supervision_path, ("c", "p"))
        with self.assertRaises(AttributeError):
            record.supervisor = "other"

    def test_supervised_failure_produces_record_not_raise(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        mid = rt.send("c", "fail", priority=4)
        done = rt.run()
        self.assertEqual(done, 0)
        records = rt.failures()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.actor_name, "c")
        self.assertEqual(record.message_id, mid)
        self.assertEqual(record.supervisor, "p")
        self.assertEqual(record.error_type, "RuntimeError")
        self.assertEqual(record.error_text, "boom:fail")
        self.assertEqual(record.supervision_path, ("c", "p"))

    def test_record_holds_full_path_on_a_chain(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        record = rt.failures()[0]
        self.assertEqual(record.supervisor, "mid")
        self.assertEqual(record.supervision_path, ("worker", "mid", "root"))

    def test_earlier_completions_counted_when_failure_ends_run(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        rt.send("c", "fail")
        rt.send("p", "p1")
        rt.send("p", "p2")
        # p is registered first: both of its messages commit, then c fails.
        self.assertEqual(rt.run(), 2)
        self.assertEqual(rt.get_state("p"), ["p1", "p2"])
        self.assertEqual(len(rt.failures()), 1)

    def test_rollback_message_state_derived_trace_id(self):
        received = []

        def bad(state, message, ctx):
            ctx.send("p", "derived")
            raise RuntimeError("nope")

        rt = ActorRuntime()
        rt.register("p", [], lambda s, m, c: (received.append(m), s)[1])
        rt.register("c", [], bad, supervisor="p")
        mid = rt.send("c", "go", priority=3)
        rt.run()
        # Message back with same id and priority; nothing else committed.
        self.assertEqual(rt.pending_count("c"), 1)
        self.assertEqual(rt.pending_count("p"), 0)
        self.assertEqual(received, [])
        self.assertEqual(rt.trace(), [])
        self.assertEqual(rt.get_state("c"), [])
        # No id consumed by the failure machinery.
        self.assertEqual(rt.send("p", "after"), mid + 1)

    def test_copy_failure_is_recorded_for_supervised_actor(self):
        flag = {"fail": True}

        class Flaky:
            def __init__(self, value):
                self.value = value

            def __deepcopy__(self, memo):
                if flag["fail"]:
                    raise RuntimeError("copy blocked")
                return Flaky(self.value)

        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", None, lambda s, m, c: Flaky(m), supervisor="p")
        mid = rt.send("c", "m", priority=2)
        # No ActorExecutionError escapes: the failure is a record instead.
        self.assertEqual(rt.run(), 0)
        record = rt.failures()[0]
        self.assertEqual((record.actor_name, record.message_id), ("c", mid))
        self.assertEqual(record.error_type, "ActorDataCopyError")
        self.assertIn("failed to deep-copy", record.error_text)
        self.assertEqual(rt.pending_count("c"), 1)
        self.assertEqual(rt.trace(), [])

    def test_base_exception_propagates_for_supervised_actor(self):
        class Stop(BaseException):
            pass

        stop = Stop()

        def bad(state, message, ctx):
            raise stop

        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], bad, supervisor="p")
        rt.send("c", 1)
        with self.assertRaises(Stop) as caught:
            rt.run()
        self.assertIs(caught.exception, stop)
        # No record, no pause: message is pending and the actor still runs.
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["c"].paused)
        self.assertEqual(rt.pending_count("c"), 1)

    def test_unsupervised_actor_still_raises_execution_error(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler)  # root, even with p registered
        rt.send("c", "fail")
        with self.assertRaises(ActorExecutionError):
            rt.run()
        self.assertEqual(rt.failures(), [])


class PausedSchedulingTests(unittest.TestCase):
    def _failing_runtime(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        rt.send("c", "fail")
        rt.run()
        return rt

    def test_run_skips_paused_and_drains_others(self):
        rt = self._failing_runtime()
        rt.send("p", "p1")
        rt.send("p", "p2")
        self.assertEqual(rt.run(), 2)
        self.assertEqual(rt.get_state("p"), ["p1", "p2"])
        # c is still paused with its failing message.
        self.assertEqual(rt.pending_count("c"), 1)
        self.assertEqual(len(rt.failures()), 1)
        self.assertEqual(rt.run(), 0)

    def test_new_delivery_enters_paused_mailbox(self):
        rt = self._failing_runtime()
        new_id = rt.send("c", "later", priority=7)
        self.assertEqual(rt.pending_count("c"), 2)
        # Still skipped.
        self.assertEqual(rt.run(), 0)
        rt.resolve_failure("c", "p", "drop")
        self.assertEqual(rt.pending_count("c"), 1)
        self.assertEqual(rt.run(), 1)
        self.assertEqual(rt.get_state("c"), ["later"])
        self.assertEqual(rt.trace()[0].message_id, new_id)
        self.assertEqual(rt.trace()[0].priority, 7)

    def test_due_timed_message_enters_paused_mailbox(self):
        rt = self._failing_runtime()
        timed_id = rt.schedule("c", "timed", delay=3, priority=5)
        result = rt.advance(3)
        self.assertEqual(result.released, (timed_id,))
        # Released into the paused actor's mailbox, not handled.
        self.assertEqual(rt.pending_count("c"), 2)
        self.assertEqual(rt.run(), 0)
        rt.resolve_failure("c", "p", "drop")
        self.assertEqual(rt.run(), 1)
        # The released message keeps its scheduled id and priority.
        self.assertEqual(rt.trace()[0].message_id, timed_id)
        self.assertEqual(rt.trace()[0].priority, 5)
        self.assertEqual(rt.get_state("c"), ["timed"])

    def test_send_once_still_dedupes_while_paused(self):
        rt = self._failing_runtime()
        self.assertEqual(
            rt.send_once("c", "k", "first"), DedupResult(2, True)
        )
        self.assertEqual(
            rt.send_once("c", "k", "again"), DedupResult(2, False)
        )
        self.assertEqual(rt.pending_count("c"), 2)

    def test_multiple_failures_appear_in_production_order(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("a", [], fail_handler, supervisor="p")
        rt.register("b", [], fail_handler, supervisor="p")
        rt.send("a", "fail")
        rt.send("b", "fail")
        # First run ends at a's failure; second skips a and fails b.
        self.assertEqual(rt.run(), 0)
        self.assertEqual([f.actor_name for f in rt.failures()], ["a"])
        self.assertEqual(rt.run(), 0)
        self.assertEqual([f.actor_name for f in rt.failures()], ["a", "b"])

    def test_failures_query_returns_independent_copies(self):
        rt = self._failing_runtime()
        first = rt.failures()
        first.append("junk")
        first[0].error_text  # immutable tuple, no mutation possible
        second = rt.failures()
        self.assertEqual(len(second), 1)
        self.assertIsNot(first, second)
        self.assertEqual(second, first[:1])


class ResolveFailureTests(unittest.TestCase):
    def test_retry_deletes_record_resumes_and_retries_message(self):
        flag = {"fail": True}

        def handler(state, message, ctx):
            if flag["fail"]:
                raise RuntimeError("still bad")
            return list(state) + [message]

        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], handler, supervisor="p")
        mid = rt.send("c", "m", priority=4)
        rt.run()
        self.assertEqual(len(rt.failures()), 1)
        result = rt.resolve_failure("c", "p", "retry")
        self.assertIsNone(result)
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["c"].paused)
        # Message still pending with the same id...
        self.assertEqual(rt.pending_count("c"), 1)
        # ...and a run re-attempts it, failing again with the same identity.
        rt.run()
        self.assertEqual(rt.failures()[0].message_id, mid)
        # Once the handler recovers, retry completes it once, original id.
        rt.resolve_failure("c", "p", "retry")
        flag["fail"] = False
        self.assertEqual(rt.run(), 1)
        self.assertEqual(rt.failures(), [])
        self.assertEqual(rt.get_state("c"), ["m"])
        self.assertEqual(
            [(t.message_id, t.priority) for t in rt.trace()], [(mid, 4)]
        )

    def test_drop_deletes_message_record_and_resumes(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        rt.send("c", "fail")
        rt.send("c", "keep")
        rt.run()
        self.assertIsNone(rt.resolve_failure("c", "p", "drop"))
        self.assertEqual(rt.failures(), [])
        self.assertFalse(rt._actors["c"].paused)
        self.assertEqual(rt.pending_count("c"), 1)
        self.assertEqual(rt.run(), 1)
        self.assertEqual(rt.get_state("c"), ["keep"])

    def test_drop_keeps_heap_order_for_remaining_mail(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], append_handler, supervisor="p")

        def handler(state, message, ctx):
            if message == "fail":
                raise RuntimeError("boom")
            return list(state) + [message]

        rt._actors["c"].handler = handler
        rt.send("c", "fail", priority=0)
        rt.run()
        rt.send("c", "high", priority=10)
        rt.send("c", "low", priority=0)
        rt.resolve_failure("c", "p", "drop")
        rt.run()
        self.assertEqual(rt.get_state("c"), ["high", "low"])

    def test_drop_keeps_send_once_reservation(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        first = rt.send_once("c", "k", "fail")
        self.assertEqual(first, DedupResult(1, True))
        rt.run()
        rt.resolve_failure("c", "p", "drop")
        # The message is gone, but the key still confirms the dropped id.
        self.assertEqual(rt.pending_count("c"), 0)
        self.assertEqual(rt.send_once("c", "k", "retry"), DedupResult(1, False))
        # Plain deliveries are unaffected.
        self.assertEqual(rt.send("c", "plain"), 2)

    def test_escalate_hands_same_record_to_parent(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        original = rt.failures()[0]
        escalated = rt.resolve_failure("worker", "mid", "escalate")
        self.assertIsInstance(escalated, FailureRecord)
        self.assertIs(escalated, rt.failures()[0])
        self.assertEqual(escalated.supervisor, "root")
        # Identity fields are untouched and the path is not shortened.
        self.assertEqual(escalated.actor_name, "worker")
        self.assertEqual(escalated.message_id, original.message_id)
        self.assertEqual(escalated.error_type, "RuntimeError")
        self.assertEqual(escalated.supervision_path,
                         ("worker", "mid", "root"))
        # Actor still paused, message still pending, order preserved.
        self.assertTrue(rt._actors["worker"].paused)
        self.assertEqual(rt.pending_count("worker"), 1)
        # The old supervisor no longer has standing.
        with self.assertRaises(LookupError):
            rt.resolve_failure("worker", "mid", "retry")
        # The new supervisor resolves it.
        self.assertIsNone(rt.resolve_failure("worker", "root", "drop"))
        self.assertEqual(rt.failures(), [])

    def test_escalate_consumes_no_id_and_moves_nothing(self):
        rt = build_chain()
        mid = rt.send("worker", "fail")
        rt.run()
        before_next = rt._next_message_id
        rt.resolve_failure("worker", "mid", "escalate")
        self.assertEqual(rt._next_message_id, before_next)
        self.assertEqual(rt.pending_count("worker"), 1)
        self.assertEqual(rt.trace(), [])
        # New delivery numbers right after the failure id.
        self.assertEqual(rt.send("root", "x"), mid + 1)

    def test_escalate_keeps_position_in_failure_order(self):
        rt = ActorRuntime()
        rt.register("root", [], append_handler)
        rt.register("p", [], append_handler, supervisor="root")
        rt.register("a", [], fail_handler, supervisor="p")
        rt.register("b", [], fail_handler, supervisor="p")
        rt.send("a", "fail")
        rt.send("b", "fail")
        rt.run()
        rt.run()
        self.assertEqual([f.actor_name for f in rt.failures()], ["a", "b"])
        rt.resolve_failure("a", "p", "escalate")
        records = rt.failures()
        self.assertEqual([f.actor_name for f in records], ["a", "b"])
        self.assertEqual(records[0].supervisor, "root")
        self.assertEqual(records[1].supervisor, "p")
        # Dropping b leaves a's escalated record first.
        rt.resolve_failure("b", "p", "drop")
        self.assertEqual([f.actor_name for f in rt.failures()], ["a"])
        self.assertEqual(rt.failures()[0].supervisor, "root")

    def test_root_escalation_raises_and_changes_nothing(self):
        rt = ActorRuntime()
        rt.register("boss", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="boss")
        rt.send("c", "fail")
        rt.run()
        snapshot_before = rt.export_snapshot()
        with self.assertRaises(SupervisionError):
            rt.resolve_failure("c", "boss", "escalate")
        # Record, pause, mailbox, order -- everything untouched.
        self.assertEqual(len(rt.failures()), 1)
        self.assertTrue(rt._actors["c"].paused)
        self.assertEqual(rt.pending_count("c"), 1)
        self.assertEqual(rt.export_snapshot(), snapshot_before)

    def test_unknown_record_raises_lookup_error(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        # Never failed.
        with self.assertRaises(LookupError):
            rt.resolve_failure("c", "p", "retry")
        # Unknown actor entirely.
        with self.assertRaises(LookupError):
            rt.resolve_failure("ghost", "p", "drop")

    def test_supervisor_identity_mismatch_raises_lookup_error(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        # Only mid holds the decision at first.
        for wrong_supervisor in ("root", "worker", "nobody", None):
            with self.subTest(wrong=wrong_supervisor):
                with self.assertRaises(LookupError):
                    rt.resolve_failure("worker", wrong_supervisor, "retry")

    def test_invalid_action_raises_value_error(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        rt.send("c", "fail")
        rt.run()
        for bad in ("restart", "stop", "RESUME", "", "retry ", None, 3):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rt.resolve_failure("c", "p", bad)

    def test_failed_operation_changes_nothing(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        # Failed calls while mid holds the decision.
        snapshot_before = rt.export_snapshot()
        with self.assertRaises(LookupError):
            rt.resolve_failure("worker", "root", "retry")
        with self.assertRaises(LookupError):
            rt.resolve_failure("ghost", "mid", "drop")
        with self.assertRaises(ValueError):
            rt.resolve_failure("worker", "mid", "restart")
        self.assertEqual(rt.export_snapshot(), snapshot_before)

        # After a valid escalation, failed calls at root leave the new
        # state untouched too.
        rt.resolve_failure("worker", "mid", "escalate")
        snapshot_escalated = rt.export_snapshot()
        with self.assertRaises(SupervisionError):
            rt.resolve_failure("worker", "root", "escalate")
        with self.assertRaises(LookupError):
            rt.resolve_failure("worker", "mid", "drop")
        with self.assertRaises(LookupError):
            rt.resolve_failure("ghost", "root", "retry")
        with self.assertRaises(ValueError):
            rt.resolve_failure("worker", "root", "stop")
        self.assertEqual(rt.export_snapshot(), snapshot_escalated)


class SupervisionSnapshotTests(unittest.TestCase):
    HANDLERS = {"root": append_handler, "mid": append_handler,
                "worker": fail_handler, "p": append_handler,
                "c": fail_handler}

    def test_pending_failure_round_trip(self):
        rt = build_chain()
        rt.send("worker", "fail", priority=6)
        rt.send("root", "r")
        rt.run()  # root completes, worker fails
        rt.send("worker", "parked")
        data = rt.export_snapshot()
        restored = ActorRuntime.restore_snapshot(data, self.HANDLERS)
        self.assertEqual(restored.failures(), rt.failures())
        self.assertTrue(restored._actors["worker"].paused)
        self.assertEqual(restored._actors["worker"].supervisor_name, "mid")
        self.assertEqual(restored._actors["mid"].supervisor_name, "root")
        self.assertIsNone(restored._actors["root"].supervisor_name)
        self.assertEqual(restored.pending_count("worker"), 2)
        self.assertEqual(restored.get_state("root"), ["r"])
        # Paused on arrival: run skips it.
        self.assertEqual(restored.run(), 0)

    def test_supervision_queries_return_copies_after_restore(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(),
                                                 self.HANDLERS)
        failures = restored.failures()
        failures.clear()
        self.assertEqual(len(restored.failures()), 1)

    def test_resume_sequence_identical_after_restore(self):
        def scenario(rt):
            rt.send("worker", "fail")
            rt.send("root", "r1")
            rt.run()
            rt.resolve_failure("worker", "mid", "escalate")
            rt.resolve_failure("worker", "root", "drop")
            rt.send("worker", "after")
            rt.run()
            return rt

        live = scenario(build_chain())
        # Take the snapshot at the failure moment and replay the suffix.
        rt = build_chain()
        rt.send("worker", "fail")
        rt.send("root", "r1")
        rt.run()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(),
                                                 self.HANDLERS)
        restored.resolve_failure("worker", "mid", "escalate")
        restored.resolve_failure("worker", "root", "drop")
        restored.send("worker", "after")
        restored.run()
        self.assertEqual(restored.get_state("root"), live.get_state("root"))
        self.assertEqual(restored.get_state("worker"),
                         live.get_state("worker"))
        self.assertEqual(restored.trace(), live.trace())
        self.assertEqual(restored.failures(), live.failures())
        self.assertEqual(restored.export_snapshot(), live.export_snapshot())
        # Numbering continues identically too.
        probe = restored.send("root", "probe")
        self.assertEqual(probe, live.send("root", "probe"))

    def test_retry_snapshot_then_identical_replay(self):
        def build_and_decide():
            rt = build_chain()
            rt.send("worker", "fail", priority=3)
            rt.run()
            rt.resolve_failure("worker", "mid", "drop")
            rt.send("worker", "next")
            rt.run()
            return rt

        one = build_and_decide()

        rt = build_chain()
        rt.send("worker", "fail", priority=3)
        rt.run()
        data = rt.export_snapshot()
        two = ActorRuntime.restore_snapshot(data, self.HANDLERS)
        two.resolve_failure("worker", "mid", "drop")
        two.send("worker", "next")
        two.run()
        self.assertEqual(two.export_snapshot(), one.export_snapshot())

    def test_drop_then_dedup_survives_snapshot(self):
        rt = ActorRuntime()
        rt.register("p", [], append_handler)
        rt.register("c", [], fail_handler, supervisor="p")
        rt.send_once("c", "k", "fail")
        rt.run()
        rt.resolve_failure("c", "p", "drop")
        restored = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"p": append_handler, "c": fail_handler}
        )
        self.assertEqual(restored.pending_count("c"), 0)
        self.assertEqual(restored.failures(), [])
        self.assertEqual(
            restored.send_once("c", "k", "x"), DedupResult(1, False)
        )

    def test_unsupervised_snapshot_has_no_supervision_fields(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.register("b", None, noop_handler)
        rt.send("a", "m")
        data = rt.export_snapshot()
        self.assertNotIn(b"supervisor", data)
        self.assertNotIn(b"paused", data)
        self.assertNotIn(b"failures", data)

    def test_tampered_failure_message_rejected(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        data = reseal(
            rt.export_snapshot(),
            lambda env: env["failures"][0].__setitem__("message_id", 999),
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)

    def test_tampered_failure_path_rejected(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        data = reseal(
            rt.export_snapshot(),
            lambda env: env["failures"][0].__setitem__(
                "path", ["worker", "root"]
            ),
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)

    def test_tampered_failure_supervisor_rejected(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        # A supervisor that is not an ancestor on the captured path.
        data = reseal(
            rt.export_snapshot(),
            lambda env: env["failures"][0].__setitem__("supervisor", "ghost"),
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)

    def test_paused_flag_without_record_rejected(self):
        rt = build_chain()
        rt.send("worker", "work")
        data = reseal(
            rt.export_snapshot(),
            lambda env: next(
                a for a in env["actors"] if a["name"] == "worker"
            ).__setitem__("paused", True),
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)

    def test_record_without_paused_flag_rejected(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()
        data = reseal(
            rt.export_snapshot(),
            lambda env: next(
                a for a in env["actors"] if a["name"] == "worker"
            ).__setitem__("paused", False),
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)

    def test_unknown_supervisor_link_rejected(self):
        rt = build_chain()
        data = reseal(
            rt.export_snapshot(),
            lambda env: next(
                a for a in env["actors"] if a["name"] == "worker"
            ).__setitem__("supervisor", "ghost"),
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)

    def test_supervision_cycle_rejected(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.register("b", None, noop_handler, supervisor="a")

        def make_cycle(env):
            actors = {a["name"]: a for a in env["actors"]}
            actors["a"]["supervisor"] = "b"

        data = reseal(rt.export_snapshot(), make_cycle)
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(
                data, {"a": noop_handler, "b": noop_handler}
            )

    def test_failure_on_unsupervised_actor_rejected(self):
        rt = ActorRuntime()
        rt.register("c", [], fail_handler)
        rt.send("c", "fail")
        with self.assertRaises(ActorExecutionError):
            rt.run()
        # Hand-craft a failure record for a root actor: never produced by
        # the runtime, so restore must refuse it.
        envelope = json.loads(rt.export_snapshot().decode("ascii"))
        envelope["payload"]["failures"] = [{
            "actor": "c",
            "message_id": 1,
            "supervisor": "c",
            "error_type": "RuntimeError",
            "error_text": "boom:fail",
            "path": ["c"],
        }]
        canonical = json.dumps(
            envelope["payload"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
        envelope["digest"] = hashlib.sha256(canonical).hexdigest()
        data = json.dumps(envelope, sort_keys=True,
                          separators=(",", ":")).encode("ascii")
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, {"c": fail_handler})

    def test_duplicate_failure_for_same_actor_rejected(self):
        rt = build_chain()
        rt.send("worker", "fail")
        rt.run()

        def duplicate(env):
            env["failures"].append(dict(env["failures"][0]))

        data = reseal(rt.export_snapshot(), duplicate)
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, self.HANDLERS)


class SupervisionDeterminismTests(unittest.TestCase):
    def test_repeated_supervised_runs_identical(self):
        def build():
            rt = build_chain()
            rt.send("worker", "fail")
            rt.send("mid", "m1")
            rt.send("root", "r1")
            rt.run()
            rt.resolve_failure("worker", "mid", "escalate")
            rt.resolve_failure("worker", "root", "retry")
            rt.run()  # fails again, back with direct supervisor mid
            rt.resolve_failure("worker", "mid", "drop")
            rt.run()
            return rt

        first = build()
        signature = (
            copy.deepcopy(first.get_state("root")),
            copy.deepcopy(first.get_state("mid")),
            copy.deepcopy(first.get_state("worker")),
            [tuple(e) for e in first.trace()],
            [tuple(f) if isinstance(f, FailureRecord) else f
             for f in first.failures()],
            first.export_snapshot(),
        )
        for _ in range(3):
            rt = build()
            self.assertEqual(
                (
                    rt.get_state("root"),
                    rt.get_state("mid"),
                    rt.get_state("worker"),
                    [tuple(e) for e in rt.trace()],
                    [tuple(f) for f in rt.failures()],
                    rt.export_snapshot(),
                ),
                signature,
            )

    def test_same_snapshot_two_restores_evolve_identically(self):
        rt = build_chain()
        rt.send("worker", "fail", priority=4)
        rt.send("root", "seed")
        rt.run()
        data = rt.export_snapshot()
        handlers = {"root": append_handler, "mid": append_handler,
                    "worker": fail_handler}

        def drive(runtime):
            runtime.resolve_failure("worker", "mid", "escalate")
            runtime.resolve_failure("worker", "root", "drop")
            runtime.send("worker", "later")
            runtime.run()
            return runtime

        one = drive(ActorRuntime.restore_snapshot(data, handlers))
        two = drive(ActorRuntime.restore_snapshot(data, handlers))
        self.assertEqual(one.export_snapshot(), two.export_snapshot())
        self.assertEqual(one.trace(), two.trace())
        self.assertEqual(one.failures(), two.failures())


if __name__ == "__main__":
    unittest.main()
