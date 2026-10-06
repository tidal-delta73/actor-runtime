"""Tests for deterministic snapshot export and restore."""
import hashlib
import json
import unittest

from actor_runtime import (
    ActorExecutionError,
    ActorRuntime,
    DedupResult,
    SnapshotError,
)


def noop_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    state = list(state)
    state.append(message)
    return state


def counting_handler(state, message, ctx):
    state = dict(state)
    state["count"] = state.get("count", 0) + 1
    return state


def failing_handler(state, message, ctx):
    raise RuntimeError("boom")


def make_busy_runtime():
    """A runtime exercising every structure the snapshot must cover."""
    rt = ActorRuntime()
    rt.register("alpha", {"count": 0}, counting_handler)
    rt.register("beta", [], append_handler)
    rt.send("alpha", "first", priority=5)                 # id 1
    rt.send("beta", {"nested": [1, (2, 3), b"\x00\xff"]})  # id 2
    rt.send_once("alpha", "key-1", "idempotent")           # id 3
    rt.schedule("beta", "later", delay=10, ttl=20, priority=2)  # id 4
    rt.schedule("alpha", "forever", delay=30)              # id 5
    rt.advance(3)
    rt.run()            # completes ids 1, 3, 2 (registration order, priority)
    rt.send("beta", "pending", priority=1)                # id 6
    return rt


HANDLERS = {"alpha": counting_handler, "beta": append_handler}


class RoundTripTests(unittest.TestCase):
    def test_queries_reflect_snapshot_moment(self):
        rt = make_busy_runtime()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        self.assertEqual(restored.clock(), rt.clock())
        self.assertEqual(restored.get_state("alpha"), rt.get_state("alpha"))
        self.assertEqual(restored.get_state("beta"), rt.get_state("beta"))
        self.assertEqual(restored.pending_count("alpha"), rt.pending_count("alpha"))
        self.assertEqual(restored.pending_count("beta"), rt.pending_count("beta"))
        self.assertEqual(restored.scheduled_count("alpha"), 1)
        self.assertEqual(restored.scheduled_count("beta"), 1)
        self.assertEqual(restored.trace(), rt.trace())

    def test_restored_queries_return_independent_copies(self):
        rt = make_busy_runtime()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        state = restored.get_state("beta")
        state.append("mutated")
        self.assertNotEqual(restored.get_state("beta"), state)
        trace = restored.trace()
        trace[0].state_before["count"] = 999
        self.assertNotEqual(restored.trace()[0].state_before["count"], 999)

    def test_tuple_and_bytes_round_trip(self):
        # The message keeps its exact types through the snapshot.
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send("a", {"t": (1, "x", [b"\xde\xad"]), "l": [(True, None, 2.5)]})
        restored = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": append_handler}
        )
        restored.run()
        self.assertEqual(
            restored.get_state("a"),
            [{"t": (1, "x", [b"\xde\xad"]), "l": [(True, None, 2.5)]}],
        )

    def test_dedup_records_survive(self):
        rt = make_busy_runtime()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        # Old key still confirms the original id, accepted=False.
        self.assertEqual(
            restored.send_once("alpha", "key-1", "whatever"),
            DedupResult(3, False),
        )
        # New deliveries continue numbering from the saved next id.
        self.assertEqual(restored.send("alpha", "new"), 7)
        self.assertEqual(
            restored.send_once("alpha", "key-2", "fresh"),
            DedupResult(8, True),
        )

    def test_scheduled_deliveries_keep_deadlines(self):
        rt = make_busy_runtime()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        result = restored.advance(7)   # clock 3 -> 10: id 4 released
        self.assertEqual(result.time, 10)
        self.assertEqual(result.released, (4,))
        self.assertEqual(result.expired, ())
        result = restored.advance(20)  # clock 10 -> 30: id 5 released
        self.assertEqual(result.released, (5,))
        restored.run()
        # Priority 2 ("later") beats priority 1 ("pending") in beta's mailbox.
        self.assertEqual(restored.get_state("beta")[-2:], ["later", "pending"])

    def test_expiry_survives(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.schedule("a", "doomed", delay=5, ttl=5)  # deadline == expiry tick
        restored = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": append_handler}
        )
        result = restored.advance(5)
        self.assertEqual(result.released, ())
        self.assertEqual(result.expired, (1,))
        self.assertEqual(restored.get_state("a"), [])

    def test_run_semantics_after_restore(self):
        rt = make_busy_runtime()
        restored = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        # Same follow-up sequence on both runtimes must agree everywhere.
        for runtime in (rt, restored):
            runtime.send("beta", "x1", priority=9)
            runtime.send("alpha", "x2")
            advance = runtime.advance(7)
            runtime.run()
        self.assertEqual(restored.get_state("alpha"), rt.get_state("alpha"))
        self.assertEqual(restored.get_state("beta"), rt.get_state("beta"))
        self.assertEqual(restored.trace(), rt.trace())
        self.assertEqual(advance.released, (4,))

    def test_rollback_semantics_after_restore(self):
        rt = ActorRuntime()
        rt.register("ok", [], append_handler)
        rt.register("bad", [], failing_handler)
        rt.send("bad", "will-fail")
        rt.send("ok", "fine")
        restored = ActorRuntime.restore_snapshot(
            rt.export_snapshot(),
            {"ok": append_handler, "bad": failing_handler},
        )
        with self.assertRaises(ActorExecutionError):
            restored.run()
        # The failing message is still pending and can be retried.
        self.assertEqual(restored.pending_count("bad"), 1)
        self.assertEqual(restored.get_state("ok"), ["fine"])


class DeterminismTests(unittest.TestCase):
    def test_byte_identical_for_identical_observable_state(self):
        def build(key_order):
            rt = ActorRuntime()
            rt.register("a", dict(key_order[0]), counting_handler)
            rt.register("b", [], append_handler)
            rt.send("a", dict(key_order[1]), priority=3)
            rt.send_once("b", "k", dict(key_order[2]))
            rt.schedule("b", dict(key_order[3]), delay=4, ttl=9)
            rt.run()
            return rt

        orders = [
            ([("x", 1), ("y", 2)], [("p", 1), ("q", 2)],
             [("m", 1), ("n", 2)], [("u", 1), ("v", 2)]),
            ([("y", 2), ("x", 1)], [("q", 2), ("p", 1)],
             [("n", 2), ("m", 1)], [("v", 2), ("u", 1)]),
        ]
        first = build(orders[0])
        second = build(orders[1])
        self.assertEqual(first.export_snapshot(), second.export_snapshot())

    def test_same_snapshot_restored_twice_evolves_identically(self):
        rt = make_busy_runtime()
        data = rt.export_snapshot()
        one = ActorRuntime.restore_snapshot(data, HANDLERS)
        two = ActorRuntime.restore_snapshot(data, HANDLERS)
        results = []
        for runtime in (one, two):
            per_runtime = [
                runtime.send_once("alpha", "key-1", "dup"),
                runtime.send_once("beta", "key-9", "new"),
                runtime.advance(7),
            ]
            runtime.send("beta", "tail", priority=4)
            runtime.run()
            per_runtime.append(runtime.advance(23))
            results.append(per_runtime)
        self.assertEqual(results[0], results[1])
        self.assertEqual(one.get_state("alpha"), two.get_state("alpha"))
        self.assertEqual(one.get_state("beta"), two.get_state("beta"))
        self.assertEqual(one.trace(), two.trace())
        # And the re-exported snapshots are byte-identical again.
        self.assertEqual(one.export_snapshot(), two.export_snapshot())

    def test_snapshot_is_pure_data(self):
        rt = make_busy_runtime()
        data = rt.export_snapshot()
        # No module import or object execution is involved: the bytes are
        # canonical JSON and contain no handler reference.
        envelope = json.loads(data.decode("ascii"))
        self.assertEqual(envelope["payload"]["format"], "actor-runtime-snapshot")
        self.assertNotIn(b"handler", data)
        self.assertNotIn(b"counting_handler", data)


class ExportFailureTests(unittest.TestCase):
    def test_set_rejected(self):
        rt = ActorRuntime()
        rt.register("a", {"bad": {1, 2}}, noop_handler)
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()
        self.assertEqual(rt.get_state("a"), {"bad": {1, 2}})

    def test_custom_instance_rejected(self):
        class Thing:
            pass

        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.send("a", Thing())
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()
        self.assertEqual(rt.pending_count("a"), 1)

    def test_non_string_mapping_key_rejected(self):
        rt = ActorRuntime()
        rt.register("a", {1: "one"}, noop_handler)
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()

    def test_non_finite_float_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            rt = ActorRuntime()
            rt.register("a", None, noop_handler)
            rt.send("a", bad)
            with self.assertRaises(SnapshotError):
                rt.export_snapshot()
            # The failed export consumed no id and changed nothing.
            self.assertEqual(rt.send("a", "ok"), 2)

    def test_circular_reference_rejected(self):
        rt = ActorRuntime()
        circular = []
        circular.append(circular)
        rt.register("a", {"loop": circular}, noop_handler)
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()
        # The state itself is untouched and still readable.
        self.assertEqual(len(rt.get_state("a")["loop"]), 1)

    def test_circular_dict_rejected(self):
        rt = ActorRuntime()
        circular = {}
        circular["self"] = circular
        rt.register("a", None, noop_handler)
        rt.send("a", circular)
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()

    def test_failed_export_leaves_everything_untouched(self):
        rt = make_busy_runtime()
        before_clock = rt.clock()
        before_trace = rt.trace()
        rt.send("beta", {"bad": {1}})  # id 7: fine in memory, not snapshot-safe
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()
        # Clock, numbering, queues, dedup and trace are all untouched.
        self.assertEqual(rt.clock(), before_clock)
        self.assertEqual(rt.trace(), before_trace)
        self.assertEqual(rt.pending_count("beta"), 2)
        self.assertEqual(rt.send("beta", "after"), 8)
        self.assertEqual(
            rt.send_once("alpha", "key-1", "dup"), DedupResult(3, False)
        )

    def test_export_recovers_once_offending_value_is_gone(self):
        rt = ActorRuntime()
        rt.register("a", None, noop_handler)
        rt.send("a", {"bad": {1, 2}})
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()
        rt.run()  # consumes the message; the state never holds the set
        self.assertIsInstance(rt.export_snapshot(), bytes)


class RestoreFailureTests(unittest.TestCase):
    def setUp(self):
        self.rt = make_busy_runtime()
        self.data = self.rt.export_snapshot()

    def tampered(self, transform):
        """Modify the payload and reseal the envelope with a valid digest.

        This exercises the semantic validators; integrity failures are
        covered separately by test_integrity_check.
        """
        envelope = json.loads(self.data.decode("ascii"))
        transform(envelope["payload"])
        canonical = json.dumps(
            envelope["payload"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
        envelope["digest"] = hashlib.sha256(canonical).hexdigest()
        return json.dumps(
            envelope, sort_keys=True, separators=(",", ":")
        ).encode("ascii")

    def test_integrity_check(self):
        # Any modification of the sealed bytes -- even one that keeps the
        # JSON parseable and the payload internally consistent -- fails.
        envelope = json.loads(self.data.decode("ascii"))
        envelope["payload"]["clock"] = envelope["payload"]["clock"] + 1
        data = json.dumps(envelope).encode("ascii")
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)
        # A corrupted digest fails too.
        envelope = json.loads(self.data.decode("ascii"))
        envelope["digest"] = "0" * 64
        data = json.dumps(envelope).encode("ascii")
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_truncated_bytes(self):
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(self.data[: len(self.data) // 2], HANDLERS)

    def test_garbage_bytes(self):
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(b"\x00\xff not json", HANDLERS)

    def test_non_bytes_rejected(self):
        with self.assertRaises(TypeError):
            ActorRuntime.restore_snapshot("{}", HANDLERS)

    def test_unsupported_version(self):
        data = self.tampered(lambda env: env.update(version=2))
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_wrong_format(self):
        data = self.tampered(lambda env: env.update(format="other"))
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_missing_field(self):
        data = self.tampered(lambda env: env.pop("clock"))
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_tampered_message_id(self):
        def break_id(env):
            env["actors"][0]["mailbox"] = []
            env["next_message_id"] = 2  # smaller than ids still referenced
        data = self.tampered(break_id)
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_unknown_scheduled_actor(self):
        data = self.tampered(
            lambda env: env["scheduled"][0].update(actor="ghost")
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_duplicate_message_id(self):
        def dupe(env):
            # Reuse beta's pending mailbox id for a scheduled delivery.
            mailbox_id = env["actors"][1]["mailbox"][0][0]
            env["scheduled"][0]["id"] = mailbox_id
        data = self.tampered(dupe)
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_dangling_dedup_reference(self):
        def dangle(env):
            for actor in env["actors"]:
                for pair in actor["dedup"]:
                    pair[1] = 999
        data = self.tampered(dangle)
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_due_scheduled_delivery_rejected(self):
        data = self.tampered(
            lambda env: env["scheduled"][0].update(release_at=1)
        )
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(data, HANDLERS)

    def test_missing_handler_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            ActorRuntime.restore_snapshot(self.data, {"alpha": counting_handler})

    def test_extra_handlers_ignored(self):
        handlers = dict(HANDLERS)
        handlers["extra"] = noop_handler
        restored = ActorRuntime.restore_snapshot(self.data, handlers)
        with self.assertRaises(LookupError):
            restored.get_state("extra")

    def test_failed_restore_returns_no_runtime(self):
        data = self.tampered(lambda env: env.update(version=99))
        try:
            result = ActorRuntime.restore_snapshot(data, HANDLERS)
        except SnapshotError:
            pass
        else:
            self.fail(f"expected SnapshotError, got runtime {result!r}")


class NonInterferenceTests(unittest.TestCase):
    def test_in_memory_behaviour_unchanged_without_snapshots(self):
        # Arbitrary deep-copyable objects still work fine in memory; only
        # exporting them is rejected.
        class Thing:
            def __init__(self, value):
                self.value = value

        rt = ActorRuntime()
        rt.register("a", Thing(0), lambda s, m, c: Thing(s.value + m))
        rt.send("a", 5)
        rt.run()
        self.assertEqual(rt.get_state("a").value, 5)
        self.assertEqual(rt.clock(), 0)


if __name__ == "__main__":
    unittest.main()
