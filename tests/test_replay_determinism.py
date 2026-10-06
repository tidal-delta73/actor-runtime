"""Deterministic replay-equivalence and malformed-input regression tests.

These tests drive only the existing public surface. The first run and the
replay both go through the project's existing replay entry points
(``tests.replay_harness.execute`` / ``replay_journal`` /
``recover_snapshot``), which themselves only use the public runtime API;
this module adds neither a product interface nor a new error protocol.

What is asserted:

* The first run records a public, replay-independent trajectory -- event
  kind, actor id, message id, logical time and order -- plus the completion
  trace (runtime message id, actor, priority) and the final queryable states.
* Replaying that recorded log through a *fresh* runtime reproduces all of it
  exactly. Replay takes only the durable bytes: no new external message can
  be delivered during it, and time advances only through recorded integer
  ``clock`` events (there is no real timer wait and no wall-clock value).
* Corrupting the recorded input -- truncating it, swapping two events,
  dropping/duplicating a frame, or referencing an actor that does not exist
  -- fails with the *existing*, specific error type
  (``CorruptRecordError`` for an unauthentic log, ``LookupError`` for an
  unknown actor) rather than an arbitrary exception, and leaves no partially
  recovered state that could still be queried: the call returns nothing, and
  a subsequent replay of the intact log still reconstructs everything.

A trajectory comparison reports the index of the first disagreeing event, so
a middle-of-the-run scheduling drift cannot be hidden by an equal end state.
"""
from __future__ import annotations

import io
import struct
import unittest

from tests.replay_harness import (
    A_CHILD,
    A_COUNTER,
    A_EVENTS,
    A_SUPERVISOR,
    CorruptRecordError,
    execute,
    recover_snapshot,
    replay_journal,
)

# ---------------------------------------------------------------------------
# Core scene: two interacting actors, normal+high priority, derived mail,
# several same-deadline timers, and one same-tick expiry.
# ---------------------------------------------------------------------------

CORE_SCRIPT = [
    ("cap", 2),
    ("send", A_COUNTER, {"kind": "inc", "n": 100}, 0),
    ("send", A_EVENTS,
     {"kind": "record", "event": "kick",
      "then": {"to": A_COUNTER, "msg": {"kind": "inc", "n": 7}}}, 0),
    ("send", A_EVENTS, {"kind": "record", "event": "lo"}, 0),
    ("send", A_EVENTS, {"kind": "record", "event": "hi"}, 9),
    ("cap", None),
    ("schedule", A_EVENTS, "t-lo", 3, 100, 0, True),
    ("schedule", A_EVENTS, "t-hi", 3, 100, 9, True),
    ("schedule", A_CHILD, "t-c", 3, 100, 0, True),
    ("schedule", A_EVENTS, "stale", 3, 0, 0, True),
    ("clock", 3),
    ("snapshot",),
]

# Runtime-wide completion order: (message_id, actor, priority). The derived
# inc (id 3) follows kick (id 2); high-priority hi (id 5) overtakes lo's
# completion slot ordering across the capped drain; among same-deadline
# timers t-hi (runtime id 7, priority 9) completes ahead of t-lo (id 6) even
# though 6 < 7; stale (id 9) never appears because it expires at its deadline.
EXPECTED_TRACE_TRIPLES = [
    (1, A_COUNTER, 0),
    (2, A_EVENTS, 0),
    (3, A_COUNTER, 0),
    (4, A_EVENTS, 0),
    (5, A_EVENTS, 9),
    (7, A_EVENTS, 9),
    (6, A_EVENTS, 0),
    (8, A_CHILD, 0),
]

EXPECTED_STATES = {
    A_EVENTS: {
        # Only the two events timers that actually fire reserve dedup ids;
        # t0003 targets child and t0004 (stale) expires, so neither is "seen"
        # by events.
        "seen": ["t0001", "t0002"],
        "arrivals": [
            {"kind": "record", "duplicate": False},
            {"kind": "record", "duplicate": False},
            {"kind": "record", "duplicate": False},
            {"kind": "timer", "duplicate": False},
            {"kind": "timer", "duplicate": False},
            # stale expired before a mailbox, so no timer arrival for t0004
        ],
        "value": [
            "kick", "lo", "hi",
            {"timer": "t-hi"}, {"timer": "t-lo"},
        ],
    },
    A_COUNTER: {
        "seen": [],
        "arrivals": [
            {"kind": "inc", "duplicate": False},
            {"kind": "inc", "duplicate": False},
        ],
        "value": 107,
    },
    A_CHILD: {
        "seen": ["t0003"],
        "arrivals": [{"kind": "timer", "duplicate": False}],
        "value": [{"timer": "t-c"}],
    },
    A_SUPERVISOR: {"observed": [], "escalations": []},
}

# The replay-independent public trajectory, with logical time carried on each
# event. Timer ids here are the durable timer ids (t0001..), not runtime ids.
EXPECTED_TRAJECTORY = [
    ("cap", 2, 0),
    ("deliver", A_COUNTER, "inc", 0, None, 0),
    ("deliver", A_EVENTS, "record", 0, None, 0),
    ("deliver", A_EVENTS, "record", 0, None, 0),
    ("deliver", A_EVENTS, "record", 9, None, 0),
    ("cap", None, 0),
    ("schedule", A_EVENTS, "t0001", 3),
    ("schedule", A_EVENTS, "t0002", 3),
    ("schedule", A_CHILD, "t0003", 3),
    ("schedule", A_EVENTS, "t0004", 3),
    ("clock", 3),
    ("timer_fired", A_EVENTS, "t0001", 3),
    ("timer_fired", A_EVENTS, "t0002", 3),
    ("timer_fired", A_CHILD, "t0003", 3),
    ("timer_expired", A_EVENTS, "t0004", 3),
    ("snapshot", 1, 3),
]


# ---------------------------------------------------------------------------
# Trajectory extraction and first-divergence comparison
# ---------------------------------------------------------------------------


def public_trajectory(result) -> list:
    """Project a scenario result onto replay-independent public events.

    Each element is a tuple of event kind, actor/message identity and logical
    time -- never object identity, thread ids or wall-clock timestamps. The
    logical clock (an integer) is threaded through the recorded events so a
    timer firing carries the tick it became due at.
    """
    clock = 0
    out: list = []
    for event in result.events:
        at = event["at"]
        if at == "cap":
            out.append(("cap", event["n"], clock))
        elif at == "deliver":
            out.append(("deliver", event["target"], event["kind"],
                        event["priority"], event.get("dedup"), clock))
        elif at == "schedule":
            out.append(("schedule", event["target"], event["id"],
                        event["fire_at"]))
        elif at == "clock":
            clock = event["new_clock"]
            assert isinstance(clock, int) and not isinstance(clock, bool)
            out.append(("clock", clock))
        elif at in ("timer_fired", "timer_expired"):
            out.append((at, event["target"], event["id"], clock))
        elif at == "snapshot":
            out.append(("snapshot", event["seq"], clock))
        else:  # fault / supervision (unused here, kept projection-total)
            out.append((at, tuple(sorted(
                (k, v) for k, v in event.items() if k != "at"))))
    return out


def first_divergence(expected, actual):
    """Index of the first unequal pair, else None -- pinpoints a schedule drift."""
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else "<missing>"
        right = actual[index] if index < len(actual) else "<missing>"
        if left != right:
            return index, left, right
    return None


def assert_trajectories_equal(testcase, expected, actual):
    drift = first_divergence(expected, actual)
    if drift is not None:
        index, left, right = drift
        testcase.fail(
            f"first trajectory divergence at event #{index}: "
            f"expected {left!r}, got {right!r}"
        )


def trace_triples(result):
    return [(entry[0], entry[1], entry[2]) for entry in result.traces]


# ---------------------------------------------------------------------------
# Raw-frame tampering (each frame keeps its own authenticity stamp, exactly as
# on-disk truncation/corruption would)
# ---------------------------------------------------------------------------


def split_frames(blob: bytes):
    stream = io.BytesIO(blob)
    frames = []
    while True:
        header = stream.read(4)
        if not header:
            break
        (length,) = struct.unpack("<I", header)
        body = stream.read(length)
        frames.append(header + body)
    return frames[0], frames[1:]          # header frame, data frames


def join_frames(header: bytes, frames) -> bytes:
    return bytes(header + b"".join(frames))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class FirstRunDeterminismTests(unittest.TestCase):
    def test_first_run_repeated_is_byte_and_event_identical(self):
        first = execute(CORE_SCRIPT)
        second = execute(CORE_SCRIPT)
        # Same inputs -> byte-identical durable log and public observations.
        self.assertEqual(first.journal, second.journal)
        self.assertEqual(first.signature(), second.signature())
        self.assertEqual(first.traces, second.traces)
        self.assertEqual(first.states, second.states)
        assert_trajectories_equal(
            self, public_trajectory(first), public_trajectory(second)
        )

    def test_first_run_matches_the_locked_public_contract(self):
        result = execute(CORE_SCRIPT)
        self.assertEqual(trace_triples(result), EXPECTED_TRACE_TRIPLES)
        self.assertEqual(result.states, EXPECTED_STATES)
        assert_trajectories_equal(
            self, EXPECTED_TRAJECTORY, public_trajectory(result)
        )


class ReplayEquivalenceTests(unittest.TestCase):
    def test_replay_into_fresh_runtime_reproduces_state_and_events(self):
        first = execute(CORE_SCRIPT)
        # replay_journal takes only durable bytes: there is no parameter by
        # which a new external message could enter during replay, and timers
        # are reconstructed integer clock events, not real waits.
        replayed = replay_journal(first.journal)
        # Final queryable state is reproduced.
        self.assertEqual(replayed.states, first.states)
        self.assertEqual(replayed.states, EXPECTED_STATES)
        # Comparable public event sequence: kind, actor, message id, logical
        # time and order, item for item starting at the first event.
        assert_trajectories_equal(
            self, public_trajectory(first), public_trajectory(replayed)
        )
        assert_trajectories_equal(
            self, EXPECTED_TRAJECTORY, public_trajectory(replayed)
        )
        # Runtime message identity and completion order reproduce exactly.
        self.assertEqual(replayed.traces, first.traces)
        self.assertEqual(trace_triples(replayed), EXPECTED_TRACE_TRIPLES)
        # The replayed log is canonically re-encodable to the same bytes.
        from tests.replay_harness import encode_records, parse_records
        self.assertEqual(
            encode_records(parse_records(replayed.journal)), first.journal
        )

    def test_replay_takes_no_live_messages_and_uses_only_logical_time(self):
        first = execute(CORE_SCRIPT)
        replayed = replay_journal(first.journal)
        # Every clock observation is an integer logical tick; nowhere does a
        # wall-clock value appear in the public trajectory.
        clocks = [event[-1] for event in public_trajectory(replayed)
                  if event[0] in ("cap", "deliver", "timer_fired",
                                  "timer_expired", "snapshot")]
        self.assertTrue(clocks)
        self.assertTrue(all(isinstance(c, int) and not isinstance(c, bool)
                            for c in clocks))
        self.assertEqual(max(clocks), 3)

    def test_snapshot_recovery_agrees_with_first_run_and_full_replay(self):
        first = execute(CORE_SCRIPT)
        seq, marker_offset, snapshot = first.snapshots[-1]
        recovered = recover_snapshot(
            snapshot, first.journal, marker_offset=marker_offset
        )
        self.assertEqual(recovered.states, first.states)
        assert_trajectories_equal(
            self, public_trajectory(first), public_trajectory(recovered)
        )
        self.assertEqual(recovered.trace_quads(), first.trace_quads())


class MalformedReplayAtomicityTests(unittest.TestCase):
    """A bad input fails with the existing error type and no partial state."""

    def setUp(self):
        self.first = execute(CORE_SCRIPT)
        self.header, self.frames = split_frames(self.first.journal)

    def _assert_fails_atomically(self, blob, expected_type):
        # The call must raise a specific existing type and hand back no
        # (partially recoverable) result at all.
        sentinel = object()
        result = sentinel
        with self.assertRaises(expected_type):
            result = replay_journal(blob)
        self.assertIs(result, sentinel,
                      "failed replay must not return a partial result")
        # Failure is atomic and leaves no residue: a subsequent replay of the
        # intact log into a fresh runtime reconstructs the whole first run.
        rebuilt = replay_journal(self.first.journal)
        self.assertEqual(rebuilt.states, self.first.states)
        self.assertEqual(rebuilt.traces, self.first.traces)
        assert_trajectories_equal(
            self, public_trajectory(self.first), public_trajectory(rebuilt)
        )

    def test_truncated_log_raises_corrupt_record_error(self):
        # Whole tail frame dropped: the header's declared frame count no
        # longer matches the stream.
        dropped_last = join_frames(self.header, self.frames[:-1])
        self._assert_fails_atomically(dropped_last, CorruptRecordError)
        # Last frame cut mid-body.
        cut_mid = (join_frames(self.header, self.frames[:-1])
                   + self.frames[-1][:3])
        self._assert_fails_atomically(cut_mid, CorruptRecordError)

    def test_swapped_events_raise_corrupt_record_error(self):
        swapped = list(self.frames)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self._assert_fails_atomically(
            join_frames(self.header, swapped), CorruptRecordError
        )
        # Swap the schedule/clock pair too (a semantic reordering).
        kinds = []
        from tests.replay_harness import parse_records
        for record in parse_records(self.first.journal):
            kinds.append(record.get("t"))
        schedule_i = kinds.index("schedule")
        clock_i = kinds.index("clock")
        swapped = list(self.frames)
        swapped[schedule_i], swapped[clock_i] = (
            swapped[clock_i], swapped[schedule_i])
        self._assert_fails_atomically(
            join_frames(self.header, swapped), CorruptRecordError
        )

    def test_dropped_or_duplicated_frame_raises_corrupt_record_error(self):
        dropped = list(self.frames)
        del dropped[2]
        self._assert_fails_atomically(
            join_frames(self.header, dropped), CorruptRecordError
        )
        duplicated = list(self.frames)
        duplicated.insert(3, duplicated[3])
        self._assert_fails_atomically(
            join_frames(self.header, duplicated), CorruptRecordError
        )

    def test_garbled_frame_raises_corrupt_record_error(self):
        from tests.replay_harness import corrupt_midframe
        self._assert_fails_atomically(
            corrupt_midframe(self.first.journal), CorruptRecordError
        )

    def test_unknown_actor_raises_lookup_error_atomically(self):
        # Structurally authentic log, but a delivery names an actor the
        # registry does not know: the runtime's existing LookupError surfaces.
        from tests.replay_harness import encode_records, parse_records
        records = [dict(record) for record in parse_records(self.first.journal)]
        target_i = next(i for i, record in enumerate(records)
                        if record.get("t") == "deliver")
        records[target_i]["target"] = "ghost"
        self._assert_fails_atomically(
            encode_records(records), LookupError
        )

    def test_error_type_is_the_specific_existing_one_not_generic(self):
        # Guard against the suite accepting "any exception": name the types.
        dropped = join_frames(self.header, self.frames[:-1])
        with self.assertRaises(CorruptRecordError):
            replay_journal(dropped)
        from tests.replay_harness import encode_records, parse_records
        records = [dict(r) for r in parse_records(self.first.journal)]
        target_i = next(i for i, r in enumerate(records)
                        if r.get("t") == "deliver")
        records[target_i]["target"] = "ghost"
        with self.assertRaises(LookupError):
            replay_journal(encode_records(records))


class FirstDivergenceReportingTests(unittest.TestCase):
    def test_helper_locates_the_first_drift_not_just_the_end_state(self):
        base = [("a", 0), ("b", 0), ("c", 0)]
        # Same final element, different middle element: the first mismatch is
        # at index 1, proving the comparison does not collapse to end-state.
        drifted = [("a", 0), ("b", 9), ("c", 0)]
        self.assertEqual(first_divergence(base, drifted),
                         (1, ("b", 0), ("b", 9)))
        self.assertIsNone(first_divergence(base, list(base)))
        longer = [("a", 0), ("b", 0), ("c", 0), ("d", 0)]
        self.assertEqual(first_divergence(base, longer)[0], 3)


if __name__ == "__main__":
    unittest.main()
