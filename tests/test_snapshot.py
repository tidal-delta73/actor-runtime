"""Tests for deterministic snapshot export and recovery.

Covers:
* complete capture of registration order, state, mailboxes, timers, dedup
  records, clock, id counter and trace;
* the supported data language and canonical (byte-identical) encoding;
* SnapshotError for unsupported data with the runtime left untouched;
* SnapshotError for truncated/tampered/unsupported/inconsistent bytes;
* LookupError for a missing handler, extras ignored;
* restored semantics: dedup reuse, id continuity, timer deadlines, run
  ordering/rollback/derived mail and independent read-only copies;
* deterministic replay of two restores driven by the same call sequence.
"""
from __future__ import annotations

import copy
import json
import unittest

from actor_runtime import (
    ActorContext,
    ActorExecutionError,
    ActorRuntime,
    AdvanceResult,
    DedupResult,
    SnapshotError,
    TraceEntry,
)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def keep_handler(state, message, ctx):
    return state


def append_handler(state, message, ctx):
    state = list(state)
    state.append(message)
    return state


def count_handler(state, message, ctx):
    return (state or 0) + 1


def collect_handler(state, message, ctx):
    new = dict(state)
    new["log"] = list(new.get("log", []))
    new["log"].append(message)
    if isinstance(message, dict) and message.get("derive"):
        ctx.send(message["derive"], "derived")
    return new


class Boom(Exception):
    pass


def fail_handler(state, message, ctx):
    raise Boom("planned failure")


HANDLERS = {
    "keep": keep_handler,
    "append": append_handler,
    "count": count_handler,
    "collect": collect_handler,
}


def build_busy_runtime():
    """A runtime with live mail, timers, dedup records, trace and clock."""
    rt = ActorRuntime()
    rt.register("keep", 0, keep_handler)
    rt.register("count", 0, count_handler)
    rt.register("collect", {"log": []}, collect_handler)
    rt.register("append", [], append_handler)
    rt.send("count", "hi", priority=10)                       # id 1
    rt.send_once("collect", "key-1", {"n": 1}, priority=2)   # id 2
    rt.schedule("append", "future", delay=10, ttl=20)        # id 3
    rt.schedule("collect", "dying", delay=8, ttl=5)          # id 4 -> expires
    rt.schedule("append", "same-tick", delay=6, ttl=6)       # id 5 -> expiry wins
    rt.run(limit=2)
    rt.advance(3)
    rt.send("count", "late")                                 # id 6
    rt.send_once("collect", "key-2", {"n": 7})               # id 7
    return rt


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


class RoundTripTests(unittest.TestCase):
    def test_empty_runtime_round_trip(self):
        rt = ActorRuntime()
        blob = rt.export_snapshot()
        self.assertIsInstance(blob, bytes)
        rt2 = ActorRuntime.restore_snapshot(blob, {})
        self.assertEqual(rt2.clock(), 0)
        self.assertEqual(rt2.trace(), [])

    def test_full_state_round_trip(self):
        rt = build_busy_runtime()
        blob = rt.export_snapshot()
        rt2 = ActorRuntime.restore_snapshot(blob, HANDLERS)

        # Registration order survives.
        self.assertEqual(
            list(rt2._actors), ["keep", "count", "collect", "append"]
        )
        self.assertEqual(rt2.clock(), 3)
        self.assertEqual(rt2.pending_count("count"), 1)
        self.assertEqual(rt2.pending_count("collect"), 1)
        self.assertEqual(rt2.pending_count("append"), 0)
        self.assertEqual(rt2.scheduled_count("append"), 2)
        self.assertEqual(rt2.scheduled_count("collect"), 1)
        self.assertEqual(rt2.get_state("count"), 1)
        self.assertEqual(rt2.get_state("collect"), {"log": [{"n": 1}]})
        self.assertEqual(rt2.get_state("append"), [])

        tr1, tr2 = rt.trace(), rt2.trace()
        self.assertEqual(tr1, tr2)
        self.assertEqual(
            [(e.message_id, e.actor_name, e.priority) for e in tr2],
            [(1, "count", 10), (2, "collect", 2)],
        )

    def test_re_export_is_byte_identical(self):
        rt = build_busy_runtime()
        blob = rt.export_snapshot()
        rt2 = ActorRuntime.restore_snapshot(blob, HANDLERS)
        self.assertEqual(rt2.export_snapshot(), blob)
        # Exporting is a read-only operation.
        self.assertEqual(rt.export_snapshot(), blob)

    def test_handlers_are_not_serialized(self):
        def uniquely_named_secret_handler(state, message, ctx):  # noqa: F841
            return state

        rt = ActorRuntime()
        rt.register("s", None, uniquely_named_secret_handler)
        blob = rt.export_snapshot()
        self.assertNotIn(b"uniquely_named_secret_handler", blob)
        # Recovery only uses the freshly supplied callable.
        rt2 = ActorRuntime.restore_snapshot(blob, {"s": keep_handler})
        self.assertIs(rt2._actors["s"].handler, keep_handler)

    def test_extra_handlers_ignored(self):
        rt = ActorRuntime()
        rt.register("a", 0, keep_handler)
        blob = rt.export_snapshot()
        rt2 = ActorRuntime.restore_snapshot(
            blob, {"a": keep_handler, "ghost": count_handler}
        )
        self.assertEqual(list(rt2._actors), ["a"])

    def test_missing_handler_raises_lookup_error(self):
        rt = build_busy_runtime()
        blob = rt.export_snapshot()
        with self.assertRaises(LookupError):
            ActorRuntime.restore_snapshot(blob, {"keep": keep_handler})
        # Failure is atomic: no runtime is returned and nothing is imported.
        with self.assertRaises(LookupError):
            ActorRuntime.restore_snapshot(blob, {"a": keep_handler})

    def test_non_mapping_handlers_raises_type_error(self):
        rt = ActorRuntime()
        blob = rt.export_snapshot()
        with self.assertRaises(TypeError):
            ActorRuntime.restore_snapshot(blob, [])  # type: ignore[arg-type]

    def test_non_bytes_blob_raises_snapshot_error(self):
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot("not bytes", {})  # type: ignore[arg-type]
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(None, {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Data language
# ---------------------------------------------------------------------------


class ValueLanguageTests(unittest.TestCase):
    def _round_trip_value(self, value):
        rt = ActorRuntime()
        rt.register("a", value, keep_handler)
        rt.send("a", value)
        rt.run()
        blob = rt.export_snapshot()
        rt2 = ActorRuntime.restore_snapshot(blob, {"a": keep_handler})
        self.assertEqual(rt2.get_state("a"), value)
        after = rt2.trace()[0]
        self.assertEqual(after.state_after, value)
        self.assertEqual(after.state_before, value)
        return blob, rt2

    def test_supported_scalars(self):
        for value in (None, True, False, 0, -17, 2 ** 70, 0.0, 3.5,
                      -2.25e10, "", "hello", b"", b"\x00\xffbinary"):
            with self.subTest(value=value):
                self._round_trip_value(value)

    def test_supported_containers(self):
        value = {
            "list": [1, [2, 3], (4, 5)],
            "tuple": (None, True, b"x", {"k": [1, (2,)]}),
            "nested": {"a": {"b": {"c": []}}},
            "empty": {"l": [], "t": (), "d": {}},
        }
        self._round_trip_value(value)

    def test_tuple_distinct_from_list_after_restore(self):
        _blob, rt2 = self._round_trip_value({"t": (1, 2), "l": [1, 2]})
        state = rt2.get_state("a")
        self.assertIsInstance(state["t"], tuple)
        self.assertIsInstance(state["l"], list)
        self.assertEqual(state["t"], (1, 2))

    def test_bytes_type_preserved(self):
        _blob, rt2 = self._round_trip_value([b"abc"])
        self.assertIsInstance(rt2.get_state("a")[0], bytes)

    def test_dict_key_order_does_not_change_bytes(self):
        rt1 = ActorRuntime()
        rt1.register("a", {"x": 1, "y": [1, 2], "z": {"p": 0, "q": 1}},
                     keep_handler)
        rt2 = ActorRuntime()
        rt2.register("a", {"z": {"q": 1, "p": 0}, "y": [1, 2], "x": 1},
                     keep_handler)
        self.assertEqual(rt1.export_snapshot(), rt2.export_snapshot())

    def test_signed_zero_is_canonicalised(self):
        rt1 = ActorRuntime()
        rt1.register("a", 0.0, keep_handler)
        rt2 = ActorRuntime()
        rt2.register("a", -0.0, keep_handler)
        self.assertEqual(rt1.export_snapshot(), rt2.export_snapshot())
        rt3 = ActorRuntime.restore_snapshot(
            rt1.export_snapshot(), {"a": keep_handler}
        )
        # Equivalent either way; the stored spelling is the canonical one.
        self.assertEqual(rt3.get_state("a"), 0.0)

    def test_identical_observable_states_identical_bytes(self):
        # The same observable state reached through different call
        # interleavings: batched drain versus one-at-a-time drain.
        a = ActorRuntime()
        a.register("c", 0, count_handler)
        a.send("c", "x")
        a.send("c", "y")
        a.run()
        b = ActorRuntime()
        b.register("c", 0, count_handler)
        b.send("c", "x")
        b.run()
        b.send("c", "y")
        b.run()
        self.assertEqual(a.get_state("c"), b.get_state("c"))
        self.assertEqual(a.trace(), b.trace())
        self.assertEqual(a.export_snapshot(), b.export_snapshot())

    # -- rejected values --------------------------------------------------

    def _assert_export_rejected_and_untouched(self, make_bad_value):
        """Export fails on the bad value and leaves every structure intact."""
        rt = ActorRuntime()
        rt.register("c", {"ok": True}, collect_handler)
        rt.send_once("c", "dup-key", make_bad_value(), priority=3)
        rt.schedule("c", "t", delay=5)
        before = {
            "clock": rt.clock(),
            "pending": rt.pending_count("c"),
            "scheduled": rt.scheduled_count("c"),
            "state": rt.get_state("c"),
            "trace": rt.trace(),
        }
        with self.assertRaises(SnapshotError):
            rt.export_snapshot()
        # Nothing moved: the failed export is a pure read.
        self.assertEqual(rt.clock(), before["clock"])
        self.assertEqual(rt.pending_count("c"), before["pending"])
        self.assertEqual(rt.scheduled_count("c"), before["scheduled"])
        self.assertEqual(rt.get_state("c"), before["state"])
        self.assertEqual(rt.trace(), before["trace"])
        # The id counter is unchanged: the next delivery continues exactly
        # where the pre-snapshot sequence left off (ids 1 and 2 were used).
        self.assertEqual(rt.send("c", "after"), 3)
        # The dedup reservation survives the failed export.
        dup = rt.send_once("c", "dup-key", "whatever")
        self.assertEqual(dup, DedupResult(message_id=1, accepted=False))
        # A corrected value can be snapshotted on a fresh runtime; the
        # original one can be drained normally too.
        rt.run()

    def test_reject_set(self):
        self._assert_export_rejected_and_untouched(lambda: {1, 2, 3})
        self._assert_export_rejected_and_untouched(lambda: frozenset((1, 2)))

    def test_reject_custom_instance(self):
        class Custom:
            pass
        self._assert_export_rejected_and_untouched(Custom)

    def test_reject_callable(self):
        self._assert_export_rejected_and_untouched(lambda: lambda s, m, c: s)

    def test_reject_non_string_mapping_key(self):
        self._assert_export_rejected_and_untouched(lambda: {1: "x"})
        self._assert_export_rejected_and_untouched(lambda: {("a",): "x"})
        self._assert_export_rejected_and_untouched(lambda: {None: "x"})

    def test_reject_non_finite_floats(self):
        self._assert_export_rejected_and_untouched(lambda: float("nan"))
        self._assert_export_rejected_and_untouched(lambda: float("inf"))
        self._assert_export_rejected_and_untouched(lambda: float("-inf"))
        self._assert_export_rejected_and_untouched(
            lambda: {"deep": [float("nan")]}
        )

    def test_reject_cycles(self):
        def list_cycle():
            x = []
            x.append(x)
            return x

        def dict_cycle():
            x = {}
            x["self"] = x
            return x

        def tuple_via_list_cycle():
            x = []
            x.append((x,))
            return x

        self._assert_export_rejected_and_untouched(list_cycle)
        self._assert_export_rejected_and_untouched(dict_cycle)
        self._assert_export_rejected_and_untouched(tuple_via_list_cycle)

    def test_shared_references_are_not_cycles(self):
        shared = {"v": 1}
        value = {"a": shared, "b": [shared, shared]}
        rt = ActorRuntime()
        rt.register("a", value, keep_handler)
        blob = rt.export_snapshot()  # must not raise
        rt2 = ActorRuntime.restore_snapshot(blob, {"a": keep_handler})
        restored = rt2.get_state("a")
        self.assertEqual(restored["a"], {"v": 1})
        self.assertEqual(restored["b"], [{"v": 1}, {"v": 1}])


# ---------------------------------------------------------------------------
# Corrupt / inconsistent bytes
# ---------------------------------------------------------------------------


def _canonical(document) -> bytes:
    return json.dumps(
        document, ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


class CorruptBytesTests(unittest.TestCase):
    def setUp(self):
        self.rt = build_busy_runtime()
        self.blob = self.rt.export_snapshot()

    def _assert_rejected(self, bad: bytes):
        with self.assertRaises(SnapshotError):
            ActorRuntime.restore_snapshot(bad, HANDLERS)

    def test_truncated_and_garbage(self):
        self._assert_rejected(self.blob[:-1])
        self._assert_rejected(self.blob[: len(self.blob) // 2])
        self._assert_rejected(b"")
        self._assert_rejected(b"{not json")
        self._assert_rejected(b"\x00\x01\x02")

    def test_trailing_bytes_rejected(self):
        self._assert_rejected(self.blob + b"  ")
        self._assert_rejected(self.blob + b"{}")

    def test_unsupported_version(self):
        d = json.loads(self.blob)
        for bad_version in (0, 2, -1, 999):
            d["v"] = bad_version
            self._assert_rejected(_canonical(d))
        d["v"] = True
        self._assert_rejected(_canonical(d))

    def test_missing_and_extra_fields(self):
        d = json.loads(self.blob)
        for field in ("v", "clock", "next_id", "actors", "scheduled", "trace"):
            removed = d.pop(field)
            self._assert_rejected(_canonical(d))
            d[field] = removed
        d["extra"] = 1
        self._assert_rejected(_canonical(d))

    def test_reordered_top_level_keys_rejected(self):
        d = json.loads(self.blob)
        reordered = json.dumps(
            {"v": d["v"], "trace": d["trace"], "scheduled": d["scheduled"],
             "next_id": d["next_id"], "clock": d["clock"], "actors": d["actors"]},
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertNotEqual(reordered, self.blob)
        self._assert_rejected(reordered)

    def test_duplicate_json_key_rejected(self):
        self._assert_rejected(
            b'{"v":1,"v":1,"clock":0,"next_id":1,'
            b'"actors":[],"scheduled":[],"trace":[]}'
        )

    def test_negative_clock_and_next_id(self):
        d = json.loads(self.blob)
        d["clock"] = -1
        self._assert_rejected(_canonical(d))
        d = json.loads(self.blob)
        d["next_id"] = 0
        self._assert_rejected(_canonical(d))
        d["next_id"] = True
        self._assert_rejected(_canonical(d))

    def test_bad_actor_structure(self):
        d = json.loads(self.blob)
        # duplicate name
        d["actors"][1]["name"] = d["actors"][0]["name"]
        self._assert_rejected(_canonical(d))
        # non-contiguous order
        d = json.loads(self.blob)
        d["actors"][0]["order"] = 5
        self._assert_rejected(_canonical(d))
        # unknown actor field
        d = json.loads(self.blob)
        d["actors"][0]["bogus"] = 1
        self._assert_rejected(_canonical(d))

    def test_mailbox_order_violation(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send("a", "low", priority=0)    # id 1
        rt.send("a", "high", priority=5)   # id 2, earlier in heap order
        d = json.loads(rt.export_snapshot())
        mails = d["actors"][0]["mailbox"]
        self.assertEqual([m[1] for m in mails], [2, 1])
        mails[0], mails[1] = mails[1], mails[0]
        self._assert_rejected(_canonical(d))

    def test_mailbox_id_in_two_actors(self):
        d = json.loads(self.blob)
        mails_a = d["actors"][1]["mailbox"]
        mails_b = d["actors"][2]["mailbox"]
        mails_b.append(copy.deepcopy(mails_a[0]))
        self._assert_rejected(_canonical(d))

    def test_dangling_dedup_reference(self):
        d = json.loads(self.blob)
        d["actors"][2]["dedup"]["ghost"] = 999
        self._assert_rejected(_canonical(d))

    def test_dedup_reference_to_other_actor_delivery(self):
        d = json.loads(self.blob)
        # id 1 completed for count; bind it as a dedup original for collect.
        d["actors"][2]["dedup"]["stolen"] = 1
        self._assert_rejected(_canonical(d))

    def test_timer_targets_unknown_actor(self):
        d = json.loads(self.blob)
        d["scheduled"][0]["actor"] = "ghost"
        self._assert_rejected(_canonical(d))

    def test_timer_past_its_deadline(self):
        d = json.loads(self.blob)
        d["scheduled"][0]["release_at"] = 2  # clock is 3
        self._assert_rejected(_canonical(d))

    def test_timer_already_expired(self):
        d = json.loads(self.blob)
        entry = d["scheduled"][1]  # delay 8, ttl 5 -> expire at 5
        self.assertEqual(entry["expire_at"], 5)
        d["clock"] = 5
        # release 8 > 5 but expiry 5 <= 5 means it should be gone
        self._assert_rejected(_canonical(d))

    def test_trace_names_unknown_actor(self):
        d = json.loads(self.blob)
        d["trace"][0]["actor"] = "ghost"
        self._assert_rejected(_canonical(d))

    def test_trace_duplicate_id(self):
        d = json.loads(self.blob)
        d["trace"].append(copy.deepcopy(d["trace"][0]))
        self._assert_rejected(_canonical(d))

    def test_id_both_pending_and_traced(self):
        d = json.loads(self.blob)
        # Forge a mailbox entry reusing completed id 1.
        d["actors"][1]["mailbox"] = [[0, 1, "forged"]]
        self._assert_rejected(_canonical(d))

    def test_id_outside_sequence(self):
        d = json.loads(self.blob)
        d["next_id"] = 3  # ids 1..7 exist
        self._assert_rejected(_canonical(d))

    def test_untagged_business_values(self):
        # A bare float/list where a tagged value is expected.
        self._assert_rejected(
            _canonical({"v": 1, "clock": 0, "next_id": 1,
                        "actors": [{
                            "name": "a", "order": 0, "state": 1.5,
                            "mailbox": [], "dedup": {},
                        }],
                        "scheduled": [], "trace": []})
        )
        self._assert_rejected(
            _canonical({"v": 1, "clock": 0, "next_id": 1,
                        "actors": [{
                            "name": "a", "order": 0, "state": [1, 2],
                            "mailbox": [], "dedup": {},
                        }],
                        "scheduled": [], "trace": []})
        )

    def test_unknown_value_tag(self):
        self._assert_rejected(
            _canonical({"v": 1, "clock": 0, "next_id": 1,
                        "actors": [{
                            "name": "a", "order": 0, "state": {"z": 1},
                            "mailbox": [], "dedup": {},
                        }],
                        "scheduled": [], "trace": []})
        )

    def test_bad_float_tag(self):
        def blob_with_state(tag):
            return _canonical({"v": 1, "clock": 0, "next_id": 1,
                               "actors": [{
                                   "name": "a", "order": 0, "state": tag,
                                   "mailbox": [], "dedup": {},
                               }],
                               "scheduled": [], "trace": []})
        self._assert_rejected(blob_with_state({"f": "not-a-number"}))
        self._assert_rejected(blob_with_state({"f": "NaN"}))
        self._assert_rejected(blob_with_state({"f": 3}))

    def test_bad_bytes_tag(self):
        def blob_with_state(tag):
            return _canonical({"v": 1, "clock": 0, "next_id": 1,
                               "actors": [{
                                   "name": "a", "order": 0, "state": tag,
                                   "mailbox": [], "dedup": {},
                               }],
                               "scheduled": [], "trace": []})
        self._assert_rejected(blob_with_state({"b": "not base64!!"}))
        self._assert_rejected(blob_with_state({"b": 5}))

    def test_dict_tag_keys_not_sorted(self):
        # ["b",...] before ["a",...] violates the sorted-key invariant.
        tag = {"d": [["b", 1], ["a", 2]]}
        self._assert_rejected(
            _canonical({"v": 1, "clock": 0, "next_id": 1,
                        "actors": [{
                            "name": "a", "order": 0, "state": tag,
                            "mailbox": [], "dedup": {},
                        }],
                        "scheduled": [], "trace": []})
        )

    def test_non_finite_json_tokens(self):
        self._assert_rejected(
            b'{"actors":[],"clock":NaN,"next_id":1,'
            b'"scheduled":[],"trace":[],"v":1}'
        )

    def test_restored_runtime_is_independent_of_blob(self):
        rt2 = ActorRuntime.restore_snapshot(self.blob, HANDLERS)
        # Mutating a second restore never touches the first.
        rt3 = ActorRuntime.restore_snapshot(self.blob, HANDLERS)
        rt3.send("count", "x")
        rt3.run()
        rt3.advance(20)
        self.assertEqual(rt2.export_snapshot(), self.blob)


# ---------------------------------------------------------------------------
# Restored runtime semantics
# ---------------------------------------------------------------------------


class RestoredSemanticsTests(unittest.TestCase):
    def test_read_only_queries_return_independent_copies(self):
        rt = build_busy_runtime()
        rt2 = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)

        state = rt2.get_state("collect")
        state["log"].append("mutated")
        state["new"] = 9
        self.assertEqual(rt2.get_state("collect"), {"log": [{"n": 1}]})

        trace = rt2.trace()
        trace.clear()
        fresh = rt2.trace()
        self.assertEqual(len(fresh), 2)
        self.assertEqual(
            [e.state_after for e in fresh],
            [1, {"log": [{"n": 1}]}],
        )

    def test_old_dedup_key_returns_original_id_and_false(self):
        rt = build_busy_runtime()
        rt2 = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        # key-1's first delivery (id 2) has already completed...
        again = rt2.send_once("collect", "key-1", {"n": 999})
        self.assertEqual(again, DedupResult(message_id=2, accepted=False))
        # ...key-2's first delivery (id 7) is still pending.
        again = rt2.send_once("collect", "key-2", "ignored")
        self.assertEqual(again, DedupResult(message_id=7, accepted=False))

    def test_new_deliveries_continue_id_sequence(self):
        rt = build_busy_runtime()
        rt2 = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        self.assertEqual(rt2.send("count", "fresh"), 8)
        self.assertEqual(
            rt2.send_once("collect", "key-3", "m"),
            DedupResult(message_id=9, accepted=True),
        )
        self.assertEqual(rt2.schedule("count", "tick", delay=4), 10)
        self.assertEqual(rt2.pending_count("collect"), 2)

    def test_timers_keep_original_deadlines_and_id_order(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        id_b = rt.schedule("a", "b", delay=5, priority=0)
        id_a = rt.schedule("a", "a", delay=5, priority=0)
        id_c = rt.schedule("a", "c", delay=2, ttl=2)  # expiry == release
        rt.advance(1)
        rt2 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": append_handler}
        )
        # Jump past all deadlines at once: events stably sorted by tick, id.
        result = rt2.advance(10)
        self.assertIsInstance(result, AdvanceResult)
        self.assertEqual(result.time, 11)
        self.assertEqual(result.released, (id_b, id_a))
        self.assertEqual(result.expired, (id_c,))
        # Released mail retains its scheduled priority/id; run drains it.
        self.assertEqual(rt2.run(), 2)
        self.assertEqual(rt2.get_state("a"), ["b", "a"])

    def test_same_tick_release_sorted_by_id_with_existing_mail(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send("a", "queued", priority=0)              # id 1
        t1 = rt.schedule("a", "t1", delay=3, priority=5)  # id 2
        t2 = rt.schedule("a", "t2", delay=3, priority=5)  # id 3
        rt2 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": append_handler}
        )
        result = rt2.advance(3)
        self.assertEqual(result.released, (t1, t2))
        rt2.run()
        # Higher priority first; within it id order; queued default last.
        self.assertEqual(rt2.get_state("a"), ["t1", "t2", "queued"])

    def test_run_respects_registration_order_after_restore(self):
        rt = ActorRuntime()
        rt.register("first", [], append_handler)
        rt.register("second", [], append_handler)
        rt.send("second", "s")
        rt.send("first", "f")
        rt2 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(),
            {"first": append_handler, "second": append_handler},
        )
        rt2.run(limit=1)
        self.assertEqual(rt2.get_state("first"), ["f"])
        self.assertEqual(rt2.get_state("second"), [])

    def test_handler_failure_rolls_back_after_restore(self):
        rt = ActorRuntime()
        rt.register("a", {"log": []}, collect_handler)
        rt.send_once("a", "fail-key", {"boom": True})
        rt2 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": fail_handler}
        )
        before_state = rt2.get_state("a")
        with self.assertRaises(ActorExecutionError) as caught:
            rt2.run()
        self.assertEqual(caught.exception.message_id, 1)
        # Message stays unacknowledged; nothing committed; trace unchanged.
        self.assertEqual(rt2.pending_count("a"), 1)
        self.assertEqual(rt2.get_state("a"), before_state)
        self.assertEqual(rt2.trace(), [])
        # The dedup reservation survives the failure.
        self.assertEqual(
            rt2.send_once("a", "fail-key", "again"),
            DedupResult(message_id=1, accepted=False),
        )
        # A second restore with a working handler retries the same mail.
        rt3 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": collect_handler}
        )
        rt3.run()
        self.assertEqual(rt3.get_state("a"), {"log": [{"boom": True}]})
        self.assertEqual(len(rt3.trace()), 1)
        self.assertEqual(
            rt3.send_once("a", "fail-key", "x"),
            DedupResult(message_id=1, accepted=False),
        )

    def test_derived_messages_commit_after_restore(self):
        rt = ActorRuntime()
        rt.register("collect", {"log": []}, collect_handler)
        rt.register("append", [], append_handler)
        rt.send("collect", {"derive": "append"})
        rt2 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(),
            {"collect": collect_handler, "append": append_handler},
        )
        self.assertEqual(rt2.run(), 2)
        self.assertEqual(rt2.get_state("append"), ["derived"])

    def test_priority_and_id_ordering_after_restore(self):
        rt = ActorRuntime()
        rt.register("a", [], append_handler)
        rt.send("a", "id-low", priority=0)
        rt.send("a", "id-high", priority=10)
        rt.send("a", "id-mid", priority=10)
        rt2 = ActorRuntime.restore_snapshot(
            rt.export_snapshot(), {"a": append_handler}
        )
        rt2.run()
        self.assertEqual(rt2.get_state("a"),
                         ["id-high", "id-mid", "id-low"])

    def test_trace_entries_fully_restored(self):
        rt = build_busy_runtime()
        rt2 = ActorRuntime.restore_snapshot(rt.export_snapshot(), HANDLERS)
        original = rt.trace()
        restored = rt2.trace()
        self.assertEqual(len(original), len(restored))
        for a, b in zip(original, restored):
            self.assertEqual(a, b)
            self.assertIsInstance(b, TraceEntry)
            self.assertEqual(a.message_id, b.message_id)
            self.assertEqual(a.actor_name, b.actor_name)
            self.assertEqual(a.priority, b.priority)
            self.assertEqual(a.state_before, b.state_before)
            self.assertEqual(a.state_after, b.state_after)


# ---------------------------------------------------------------------------
# Determinism of recovery
# ---------------------------------------------------------------------------


def _drive(rt: ActorRuntime):
    """Apply a fixed mixed call sequence; return observations."""
    out = {}
    out["dup1"] = rt.send_once("collect", "key-1", "x")
    out["new_id"] = rt.send("count", "m", priority=4)
    out["advance1"] = rt.advance(5)
    out["run1"] = rt.run()
    out["advance2"] = rt.advance(10)
    out["run2"] = rt.run()
    out["dup2"] = rt.send_once("collect", "key-2", "y")
    out["sched"] = rt.schedule("append", "z", delay=2)
    out["advance3"] = rt.advance(2)
    out["run3"] = rt.run()
    out["states"] = {name: rt.get_state(name) for name in HANDLERS}
    out["trace"] = [
        (e.message_id, e.actor_name, e.priority,
         e.state_before, e.state_after)
        for e in rt.trace()
    ]
    out["blob"] = rt.export_snapshot()
    return out


class ReplayDeterminismTests(unittest.TestCase):
    def test_two_restores_same_sequence_identical(self):
        base = build_busy_runtime()
        blob = base.export_snapshot()
        rt1 = ActorRuntime.restore_snapshot(blob, HANDLERS)
        rt2 = ActorRuntime.restore_snapshot(blob, HANDLERS)
        o1 = _drive(rt1)
        o2 = _drive(rt2)
        for key in o1:
            self.assertEqual(o1[key], o2[key], f"difference at {key}")
        self.assertEqual(o1["blob"], o2["blob"])

    def test_restored_matches_live_runtime_driven_the_same_way(self):
        base = build_busy_runtime()
        blob = base.export_snapshot()
        live = build_busy_runtime()
        # The two start byte-identical; the same calls from the public API
        # keep them so.
        self.assertEqual(live.export_snapshot(), blob)
        restored = ActorRuntime.restore_snapshot(blob, HANDLERS)
        o_live = _drive(live)
        o_restored = _drive(restored)
        for key in o_live:
            self.assertEqual(o_live[key], o_restored[key],
                             f"difference at {key}")

    def test_snapshot_bytes_are_stable(self):
        rt = build_busy_runtime()
        first = rt.export_snapshot()
        for _ in range(3):
            self.assertEqual(rt.export_snapshot(), first)


if __name__ == "__main__":
    unittest.main()
