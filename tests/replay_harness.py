"""Test-only deterministic replay harness, built strictly on the public API.

Nothing in this module is a product surface: it only exercises
``ActorRuntime.register/send/run/get_state/pending_count/trace`` and the
documented exceptions (``ValueError``/``TypeError``/``LookupError``/
``ActorExecutionError``).  Facilities the baseline deliberately does not ship
(timed delivery, durable recovery, supervision, deduplication, throughput
limits) are emulated *around* the public runtime, inside the test package:

* a virtual clock replaces wall-clock time; timer expiry is an explicit
  ``clock`` action, never ``time.sleep``;
* an append-only, length-framed JSON journal stands in for a persistence log,
  and snapshot blobs stand in for checkpoints;
* supervision (resume / restart / stop / escalate) is expressed by replaying
  the durable record prefix through a fresh public runtime -- the same act a
  journal-based recovery performs;
* at-least-once duplicate delivery is a recorded delivery carrying the same
  dedup id;
* an in-flight cap is a scheduling configuration: after one external input
  at most ``cap`` messages complete before the next external input, each
  completion being a public ``run(limit=1)`` step.  Ordering still depends
  only on registration order, priority and message id.

First execution, journal replay and snapshot recovery all drive the same
interpreter over the same record stream, so equivalence is structural.

Script invariants (enforced by :func:`validate_script`):

1. a fault injection delivery is always immediately followed by ``resolve``,
   and faults are only injected while no in-flight cap is active;
2. every ``resolve`` precedes every ``snapshot`` (snapshot recovery only ever
   covers a suffix that contains no supervision rebuild);
3. no derived delivery (``then``) may appear in the fault prefix -- the
   prefix is a sequence of plain deliveries, so a restart rebuild has no
   cross-actor derived effects to account for, and a derived ``then`` never
   targets the restartable actor;
4. a snapshot is only requested after the cap has been restored to ``None``,
   which drains every live mailbox to quiescence first.

Durable bytes use ``json`` (ensure_ascii, allow_nan=False, sort_keys, compact
separators) with a 4-byte length frame per record, so equivalent journals are
byte-for-byte identical.  Corrupt frames raise :class:`CorruptRecordError`;
an incompatible schema/codec raises :class:`IncompatibleSnapshotError`.
Both are fixture-local error types describing the test persistence layer,
not new product error protocols.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from actor_runtime import ActorContext, ActorExecutionError, ActorRuntime

# ---------------------------------------------------------------------------
# Fixture-local error protocol (test persistence layer only)
# ---------------------------------------------------------------------------


class HarnessError(Exception):
    """Base class for errors raised by the test persistence fixtures."""


class CorruptRecordError(HarnessError):
    """A persisted journal frame cannot be decoded or authenticated."""


class IncompatibleSnapshotError(HarnessError):
    """A snapshot or journal header has an incompatible schema or codec."""


class InvalidScript(HarnessError):
    """A generated/shrunk script violates the replay invariants."""


# ---------------------------------------------------------------------------
# Deterministic pseudo-random generator (fixed, never OS-seeded)
# ---------------------------------------------------------------------------


class DeterministicRng:
    """xorshift64*: a tiny fixed PRNG with stable cross-run behaviour.

    ``random.Random`` is avoided on purpose: its seeding/compact-state
    semantics are a CPython implementation detail replay equivalence should
    not depend on.
    """

    MULT = 0x2545F4914F6CDD1D
    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        state = int(seed) & self.MASK
        if state == 0:
            state = 0x9E3779B97F4A7C15  # xorshift never allows a zero state
        self._state = state

    def next_u64(self) -> int:
        x = self._state
        x ^= (x >> 12) & self.MASK
        x ^= (x << 25) & self.MASK
        x ^= (x >> 27) & self.MASK
        x &= self.MASK
        self._state = x
        return (x * self.MULT) & self.MASK

    def below(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        return self.next_u64() % n

    def choose(self, seq):
        return seq[self.below(len(seq))]

    def chance(self, pct: int) -> bool:
        """Return True with probability ``pct`` percent."""
        return self.below(100) < pct

    def randint(self, lo: int, hi: int) -> int:
        return lo + self.below(hi - lo + 1)


# ---------------------------------------------------------------------------
# Actor names and behaviour (all handler state is JSON-canonicalisable)
# ---------------------------------------------------------------------------

A_EVENTS = "events"
A_COUNTER = "counter"
A_SUPERVISOR = "supervisor"
A_CHILD = "child"
ACTOR_NAMES = (A_EVENTS, A_COUNTER, A_SUPERVISOR, A_CHILD)
# Actors that may be restarted; no handler-derived delivery may target them,
# so that a restart rebuild is self-contained.
RESTARTABLE = (A_CHILD,)

# Message envelopes (plain JSON-able dicts):
#   events:     {"kind":"record","event":E,"dedup"?:id,"then"?:derived}
#               {"kind":"timer","token":str,"dedup"?:id}
#               {"kind":"fail","key":str}
#   counter:    {"kind":"inc","n":int,"dedup"?:id} / {"kind":"fail","key"}
#   supervisor: {"kind":"record","event":E} / {"kind":"escalated","key"}
#   child:      {"kind":"record","event":E,"dedup"?:id}
#               {"kind":"timer",...} / {"kind":"fail","key"}


def _dedup_then(state, message, mut, after):
    """Apply ``mut`` once per dedup id; ``after`` runs only on first apply.

    Every delivery reaches the handler and is recorded as an arrival; a
    duplicate contributes no state change and no derived delivery.  State is
    replaced (never mutated in place), matching the atomic commit contract.
    """
    dedup_id = message.get("dedup")
    if dedup_id is not None and dedup_id in state["seen"]:
        return {
            "seen": state["seen"],
            "arrivals": state["arrivals"] + [
                {"kind": message["kind"], "duplicate": True}
            ],
            "value": state["value"],
        }
    new_state = {
        "seen": set(state["seen"]),
        "arrivals": list(state["arrivals"]),
        "value": list(state["value"]) if isinstance(state["value"], list)
        else state["value"],
    }
    mut(new_state)
    new_state["arrivals"].append(
        {"kind": message["kind"], "duplicate": False}
    )
    if dedup_id is not None:
        new_state["seen"].add(dedup_id)
    if after is not None:
        after(new_state)
    return new_state


def events_handler(state, message, ctx: ActorContext):
    kind = message["kind"]
    if kind == "record":
        def mut(ns):
            ns["value"].append(message["event"])

        def after(ns):
            derived = message.get("then")
            if derived is not None:
                # Async derived delivery: the runtime enqueues it only after
                # this handler commits.
                ctx.send(derived["to"], derived["msg"])

        return _dedup_then(state, message, mut, after)
    if kind == "timer":
        return _dedup_then(
            state, message,
            lambda ns: ns["value"].append({"timer": message["token"]}),
            None,
        )
    if kind == "fail":
        raise RuntimeError(f"events fault: {message['key']}")
    raise AssertionError(f"events actor got unexpected message: {message!r}")


def counter_handler(state, message, ctx: ActorContext):
    kind = message["kind"]
    if kind == "inc":
        return _dedup_then(
            state, message,
            lambda ns: ns.__setitem__("value", ns["value"] + message["n"]),
            None,
        )
    if kind == "fail":
        raise ValueError(f"counter fault: {message['key']}")
    raise AssertionError(f"counter actor got unexpected message: {message!r}")


def supervisor_handler(state, message, ctx: ActorContext):
    # The supervisor deliberately emits no derived messages: its child is
    # restartable, and keeping this handler side-effect free makes a restart
    # rebuild a pure function of the durable record prefix.  Escalations
    # reach it as ordinary external ("escalated") deliveries.
    kind = message["kind"]
    if kind == "record":
        return {
            "observed": list(state["observed"]) + [message["event"]],
            "escalations": list(state["escalations"]),
        }
    if kind == "escalated":
        return {
            "observed": list(state["observed"]),
            "escalations": list(state["escalations"]) + [message["key"]],
        }
    if kind == "fail":
        raise RuntimeError(f"supervisor fault: {message['key']}")
    raise AssertionError(f"supervisor got unexpected message: {message!r}")


def child_handler(state, message, ctx: ActorContext):
    kind = message["kind"]
    if kind == "record":
        return _dedup_then(
            state, message,
            lambda ns: ns["value"].append(message["event"]),
            None,
        )
    if kind == "timer":
        return _dedup_then(
            state, message,
            lambda ns: ns["value"].append({"timer": message["token"]}),
            None,
        )
    if kind == "fail":
        raise RuntimeError(f"child fault: {message['key']}")
    raise AssertionError(f"child actor got unexpected message: {message!r}")


HANDLERS = {
    A_EVENTS: events_handler,
    A_COUNTER: counter_handler,
    A_SUPERVISOR: supervisor_handler,
    A_CHILD: child_handler,
}


def initial_state_for(name: str) -> Any:
    if name == A_SUPERVISOR:
        return {"observed": [], "escalations": []}
    value = 0 if name == A_COUNTER else []
    return {"seen": set(), "arrivals": [], "value": value}


def register_all(rt: ActorRuntime) -> None:
    for name in ACTOR_NAMES:
        rt.register(name, initial_state_for(name), HANDLERS[name])


def canonical_state(raw: Any) -> Any:
    """Normalise handler state: sets become sorted lists."""
    if isinstance(raw, dict) and "seen" in raw and "arrivals" in raw:
        return {
            "seen": sorted(raw["seen"]),
            "arrivals": raw["arrivals"],
            "value": raw["value"],
        }
    return raw


# ---------------------------------------------------------------------------
# Durable record codec
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
CODEC = "json-v1"


def _json_dumps(obj: Any) -> bytes:
    return (
        json.dumps(
            obj, ensure_ascii=True, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")


def _json_loads(blob: bytes) -> Any:
    return json.loads(blob.decode("utf-8"))


def _frame(rec: dict) -> bytes:
    body = _json_dumps(rec)
    return struct.pack("<I", len(body)) + body


def encode_records(records: list[dict], *,
                   schema: int = SCHEMA_VERSION,
                   codec: str = CODEC) -> bytes:
    out = bytearray()
    out += _frame({"t": "header", "schema": schema, "codec": codec})
    for rec in records:
        out += _frame(rec)
    return bytes(out)


def _read_frame(stream: io.BytesIO, *, require_header: bool) -> Optional[dict]:
    header = stream.read(4)
    if not header:
        return None
    if len(header) != 4:
        raise CorruptRecordError("truncated record header")
    (length,) = struct.unpack("<I", header)
    body = stream.read(length)
    if len(body) != length:
        raise CorruptRecordError("truncated record body")
    try:
        rec = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptRecordError(f"undecodable record: {exc}") from exc
    if not isinstance(rec, dict) or "t" not in rec:
        raise CorruptRecordError("record missing type tag")
    if require_header and rec.get("t") != "header":
        raise CorruptRecordError("journal missing header frame")
    if rec.get("t") == "header":
        if rec.get("schema") != SCHEMA_VERSION:
            raise IncompatibleSnapshotError(
                f"journal schema {rec.get('schema')!r} incompatible "
                f"with {SCHEMA_VERSION}"
            )
        if rec.get("codec") != CODEC:
            raise IncompatibleSnapshotError(
                f"journal codec {rec.get('codec')!r} unsupported"
            )
    return rec


def parse_records(blob: bytes, *, start_offset: int = 0,
                  require_header: bool = True) -> list[dict]:
    stream = io.BytesIO(blob)
    stream.seek(start_offset)
    records = []
    while True:
        rec = _read_frame(stream, require_header=require_header)
        require_header = False
        if rec is None:
            break
        if rec.get("t") == "header":
            continue
        records.append(rec)
    if start_offset == 0 and not records:
        raise CorruptRecordError("journal contains only a header")
    return records


def frame_offsets(blob: bytes) -> list[int]:
    """Byte offset of every frame (header first) in ``blob``."""
    stream = io.BytesIO(blob)
    offsets = []
    while True:
        pos = stream.tell()
        header = stream.read(4)
        if not header:
            break
        if len(header) != 4:
            raise CorruptRecordError("truncated record header")
        (length,) = struct.unpack("<I", header)
        offsets.append(pos)
        if len(stream.read(length)) != length:
            raise CorruptRecordError("truncated record body")
    return offsets


def corrupt_midframe(blob: bytes) -> bytes:
    """Return a deterministically damaged copy of a journal blob."""
    if len(blob) < 12:
        return blob + b"\x00"
    damaged = bytearray(blob)
    damaged[len(damaged) // 2] ^= 0xFF
    return bytes(damaged)


def truncate_midframe(blob: bytes) -> bytes:
    """Return a copy cut a few bytes short of its last frame."""
    return blob[: max(5, len(blob) - 3)]


def tamper_snapshot(blob: bytes, **changes) -> bytes:
    """Decode a snapshot blob, apply changes, re-encode it."""
    outer = _json_loads(blob)
    outer["snapshot"].update(changes)
    return _json_dumps(outer)


# ---------------------------------------------------------------------------
# Script normalisation: commands -> durable records
# ---------------------------------------------------------------------------
#
# Command tuple alphabet:
#   ("send", target, message, priority[, want_dedup])
#   ("redeliver", ordinal)           ordinal = nth dedup send, 0-based
#   ("schedule", target, token, delay, ttl, priority, want_dedup)
#   ("clock", ticks)
#   ("cap", None|positive_int)
#   ("resolve",)
#   ("snapshot",)

SEND = "send"
REDELIVER = "redeliver"
SCHEDULE = "schedule"
CLOCK = "clock"
CAP = "cap"
RESOLVE = "resolve"
SNAPSHOT = "snapshot"


@dataclass
class PendingTimer:
    fire_at: int
    expire_at: int
    target: str
    message: dict
    priority: int
    timer_id: str


def validate_script(script: list) -> None:
    """Check the structural replay invariants (raises InvalidScript)."""
    dedup_sends = 0
    snapshot_positions: list[int] = []
    resolve_positions: list[int] = []
    current_cap: Optional[int] = None
    for i, cmd in enumerate(script):
        op = cmd[0]
        if op == SEND:
            message = cmd[2]
            if len(cmd) > 4 and cmd[4]:
                dedup_sends += 1
            if message.get("kind") == "fail":
                if i + 1 >= len(script) or script[i + 1][0] != RESOLVE:
                    raise InvalidScript("fail delivery must adjoin resolve")
                if current_cap is not None:
                    raise InvalidScript(
                        "fault injection requires an uncapped drain"
                    )
            if message.get("then") is not None:
                target = message["then"].get("to")
                if target in RESTARTABLE:
                    raise InvalidScript(
                        "derived delivery must not target a restartable actor"
                    )
        elif op == REDELIVER:
            if cmd[1] >= dedup_sends:
                raise InvalidScript("redeliver ordinal has no original")
        elif op == CAP:
            n = cmd[1]
            if n is not None and (not isinstance(n, int)
                                  or isinstance(n, bool) or n <= 0):
                raise InvalidScript("cap must be None or a positive int")
            current_cap = n
        elif op == RESOLVE:
            resolve_positions.append(i)
        elif op == SNAPSHOT:
            snapshot_positions.append(i)
            if current_cap is not None:
                raise InvalidScript(
                    "snapshot requires cap restored to None (drained)"
                )
    if snapshot_positions and resolve_positions and \
            max(resolve_positions) > min(snapshot_positions):
        raise InvalidScript("all resolves must precede the first snapshot")
    # No derived deliveries may share a prefix with a fault: they complicate a
    # restart rebuild with cross-actor effects. Generator-issued scripts put
    # all faults in one early fault-free-of-deriveds window; hand-written
    # scenarios using supervision simply avoid ``then`` before resolution.
    last_resolve = max(resolve_positions, default=-1)
    for i, cmd in enumerate(script[: last_resolve + 1]):
        if cmd[0] == SEND and cmd[2].get("then") is not None:
            raise InvalidScript(
                "no derived (then) delivery allowed in the fault prefix"
            )


def normalize_script(script: list) -> list[dict]:
    validate_script(script)
    records: list[dict] = []
    originals: list[dict] = []  # durable deliveries keyed by dedup ordinal
    d_seq = 0
    t_seq = 0
    snap_seq = 0

    for cmd in script:
        op = cmd[0]
        if op == SEND:
            _, target, message, priority = cmd[:4]
            want_dedup = cmd[4] if len(cmd) > 4 else False
            message = dict(message)
            dedup_id = None
            if want_dedup:
                d_seq += 1
                dedup_id = f"d{d_seq:04d}"
                message["dedup"] = dedup_id
            else:
                message.pop("dedup", None)
            rec = {
                "t": "deliver",
                "target": target,
                "message": message,
                "priority": int(priority),
                "dedup": dedup_id,
                "redelivery": False,
            }
            records.append(rec)
            if dedup_id is not None:
                originals.append(rec)
        elif op == REDELIVER:
            original = originals[cmd[1]]
            records.append({
                "t": "deliver",
                "target": original["target"],
                "message": dict(original["message"]),
                "priority": original["priority"],
                "dedup": original["dedup"],
                "redelivery": True,
            })
        elif op == SCHEDULE:
            _, target, token, delay, ttl, priority, want_dedup = cmd
            t_seq += 1
            timer_id = f"t{t_seq:04d}"
            records.append({
                "t": "schedule",
                "id": timer_id,
                "target": target,
                "token": token,
                "delay": int(delay),
                "ttl": int(ttl),
                "priority": int(priority),
                "dedup": timer_id if want_dedup else None,
            })
        elif op == CLOCK:
            records.append({"t": "clock", "ticks": int(cmd[1])})
        elif op == CAP:
            n = cmd[1]
            records.append({"t": "cap", "n": None if n is None else int(n)})
        elif op == RESOLVE:
            records.append({"t": "resolve"})
        elif op == SNAPSHOT:
            snap_seq += 1
            records.append({"t": "snapshot", "seq": snap_seq})
        else:
            raise InvalidScript(f"unknown command: {op!r}")
    return records


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    states: dict
    traces: list
    events: list
    supervision: list
    delivered_dedup: list
    restored_len: int = 0
    journal: bytes = b""
    snapshots: list = field(default_factory=list)  # (seq, offset, blob)

    def signature(self) -> bytes:
        """Byte signature of contract-visible state/event observations.

        Runtime-assigned message ids are compared separately (full trace for
        journal replay, generation-relative ordering for snapshot recovery).
        """
        return _json_dumps({
            "states": self.states,
            "events": self.events,
            "supervision": self.supervision,
            "delivered_dedup": self.delivered_dedup,
        })

    def trace_quads(self) -> list:
        """Completion-order trace without runtime-assigned ids."""
        return [(e[1], e[2], e[3], e[4]) for e in self.traces]

    def trace_ids(self) -> list:
        return [e[0] for e in self.traces]


# ---------------------------------------------------------------------------
# Supervision policy
# ---------------------------------------------------------------------------
#
# Deterministic from public failure facts only:
#   child      : first fault -> restart, further faults -> escalate
#   counter    : always resume
#   events     : always stop
#   supervisor : always escalate

DEFAULT_POLICY_OUTCOMES = {
    A_CHILD: ("restart", "escalate"),
    A_COUNTER: ("resume", "resume"),
    A_EVENTS: ("stop", "stop"),
    A_SUPERVISOR: ("escalate", "escalate"),
}


def default_policy(actor_name: str, attempts: int) -> str:
    outcomes = DEFAULT_POLICY_OUTCOMES[actor_name]
    return outcomes[min(attempts - 1, len(outcomes) - 1)]


# ---------------------------------------------------------------------------
# Interpreter (shared by first run, journal replay and snapshot recovery)
# ---------------------------------------------------------------------------


class Interpreter:
    def __init__(self, *, policy: Callable[[str, int], str] = default_policy):
        self.policy = policy
        self.rt = ActorRuntime()
        register_all(self.rt)
        self.clock = 0
        self.cap: Optional[int] = None
        self.stopped: set[str] = set()
        self.attempts: dict[str, int] = {}
        self.timers: list[PendingTimer] = []
        self.events: list[dict] = []
        self.supervision: list[dict] = []
        self.delivered_dedup: list[dict] = []
        self.pending_fault: Optional[dict] = None
        self.snapshots: list[tuple] = []

    # -- snapshots --------------------------------------------------------

    def snapshot_blob(self) -> bytes:
        states = {}
        for name in ACTOR_NAMES:
            states[name] = {"stopped": True} if name in self.stopped \
                else canonical_state(self.rt.get_state(name))
        payload = {
            "snapshot": {
                "schema": SCHEMA_VERSION,
                "codec": CODEC,
                "cap": self.cap,
                "clock": self.clock,
                "stopped": sorted(self.stopped),
                "attempts": dict(self.attempts),
                "states": states,
                "timers": [
                    {
                        "fire_at": t.fire_at,
                        "expire_at": t.expire_at,
                        "target": t.target,
                        "message": t.message,
                        "priority": t.priority,
                        "id": t.timer_id,
                    }
                    for t in self.timers
                ],
                "trace": [
                    [
                        e.message_id, e.actor_name, e.priority,
                        canonical_state(e.state_before),
                        canonical_state(e.state_after),
                    ]
                    for e in self.rt.trace()
                ],
                "events": list(self.events),
                "supervision": list(self.supervision),
                "delivered_dedup": list(self.delivered_dedup),
            }
        }
        return _json_dumps(payload)

    def load_snapshot(self, blob: bytes) -> int:
        try:
            outer = _json_loads(blob)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptRecordError(f"undecodable snapshot: {exc}") from exc
        snap = outer.get("snapshot") if isinstance(outer, dict) else None
        if not isinstance(snap, dict):
            raise CorruptRecordError("snapshot missing payload")
        if snap.get("schema") != SCHEMA_VERSION:
            raise IncompatibleSnapshotError(
                f"snapshot schema {snap.get('schema')!r} incompatible "
                f"with {SCHEMA_VERSION}"
            )
        if snap.get("codec") != CODEC:
            raise IncompatibleSnapshotError(
                f"snapshot codec {snap.get('codec')!r} unsupported"
            )
        self.cap = snap["cap"]
        self.clock = int(snap["clock"])
        self.stopped = set(snap.get("stopped", []))
        self.attempts = dict(snap.get("attempts", {}))
        self.timers = [
            PendingTimer(
                fire_at=int(t["fire_at"]),
                expire_at=int(t["expire_at"]),
                target=t["target"],
                message=t["message"],
                priority=int(t["priority"]),
                timer_id=t["id"],
            )
            for t in snap.get("timers", [])
        ]
        self.events = list(snap["events"])
        self.supervision = list(snap["supervision"])
        self.delivered_dedup = list(snap["delivered_dedup"])
        # Restore through the public registration entry point: the snapshot
        # state becomes the new runtime generation's initial state.
        self.rt = ActorRuntime()
        for name in ACTOR_NAMES:
            raw = snap["states"][name]
            if isinstance(raw, dict) and raw.get("stopped"):
                state = initial_state_for(name)
            elif name == A_SUPERVISOR:
                state = {
                    "observed": list(raw["observed"]),
                    "escalations": list(raw["escalations"]),
                }
            else:
                state = {
                    "seen": set(raw["seen"]),
                    "arrivals": list(raw["arrivals"]),
                    "value": raw["value"],
                }
            self.rt.register(name, state, HANDLERS[name])
        self._stored_trace = [tuple(e) for e in snap["trace"]]
        return len(self._stored_trace)

    # -- record processing ------------------------------------------------

    @staticmethod
    def _faulting_positions(records: list[dict]) -> set[int]:
        """Deliver positions immediately followed by their fault resolution.

        Adjacency is guaranteed by validate_script: the faulting delivery is
        the last durable input before the resolve (first run) or supervision
        (replay) record the failure produced.
        """
        out = set()
        for i in range(len(records) - 1):
            rec, nxt = records[i], records[i + 1]
            if (rec.get("t") == "deliver"
                    and nxt.get("t") in ("resolve", "supervision")
                    and (nxt.get("t") == "resolve"
                         or nxt.get("actor") == rec.get("target"))):
                out.add(i)
        return out

    @staticmethod
    def _restart_positions(records: list[dict]) -> dict[str, int]:
        return {
            rec["actor"]: i
            for i, rec in enumerate(records)
            if rec.get("t") == "supervision" and rec.get("action") == "restart"
        }

    def run_records(self, records: list[dict], *, rebuild: bool = False):
        # Faulting positions are structural (fail adjoins resolve), so they
        # are known on the first run too; restart positions only exist once
        # resolves have been concretised, i.e. during rebuild.
        faulting = self._faulting_positions(records)
        restart_positions = (
            self._restart_positions(records) if rebuild else {}
        )
        for i, rec in enumerate(records):
            self._step(rec, i, rebuild, faulting, restart_positions)
        return self._result()

    def _step(self, rec, i, rebuild, faulting, restart_positions) -> None:
        kind = rec["t"]
        if kind == "cap":
            self.cap = rec["n"]
            self.events.append({"at": "cap", "n": rec["n"]})
            # Restoring the cap to None drains to quiescence; tightening the
            # cap only affects future external inputs (nothing to re-drain).
            if rec["n"] is None:
                self._drain()
        elif kind == "deliver":
            self._step_deliver(rec, i, rebuild, faulting, restart_positions)
            self._drain()
        elif kind == "schedule":
            self._step_schedule(rec)
        elif kind == "clock":
            self._step_clock(rec, i, rebuild, restart_positions)
            self._drain()
        elif kind == "resolve":
            raise InvalidScript("resolve must be concretised before replay")
        elif kind == "supervision":
            self._step_supervision(rec)
            self._drain()
        elif kind == "snapshot":
            # Marker event is logged *before* the blob is captured, so a
            # recovery that skips this frame still observes the marker.
            self.events.append({"at": "snapshot", "seq": rec["seq"]})
            self.snapshots.append((rec["seq"], self.snapshot_blob()))
        else:
            raise CorruptRecordError(f"unknown record type: {kind!r}")

    def _step_deliver(self, rec, i, rebuild, faulting,
                      restart_positions) -> None:
        target = rec["target"]
        message = dict(rec["message"])
        if rec.get("dedup") is not None:
            message["dedup"] = rec["dedup"]
        is_faulting = rebuild and i in faulting
        erased_by_restart = (
            rebuild and not is_faulting
            and target in RESTARTABLE
            and i < restart_positions.get(target, -1)
        )
        discarded = is_faulting or erased_by_restart
        # The delivery did enter a live actor mailbox unless the actor was
        # already stopped; a faulting/restart-erased delivery is separately
        # marked discarded by supervision rather than refused up front.
        accepted = target not in self.stopped
        if accepted and not discarded:
            self.rt.send(target, message, priority=int(rec["priority"]))
        self.events.append({
            "at": "deliver",
            "target": target,
            "kind": message["kind"],
            "priority": int(rec["priority"]),
            "dedup": rec.get("dedup"),
            "redelivery": bool(rec.get("redelivery")),
            "accepted": accepted,
            "discarded": discarded,
        })
        if rec.get("dedup") is not None:
            self.delivered_dedup.append({
                "dedup": rec["dedup"],
                "duplicate": bool(rec.get("redelivery")),
            })

    def _step_schedule(self, rec) -> None:
        message = {"kind": "timer", "token": rec["token"]}
        if rec.get("dedup") is not None:
            message["dedup"] = rec["dedup"]
        fire_at = self.clock + int(rec["delay"])
        self.timers.append(PendingTimer(
            fire_at=fire_at,
            expire_at=fire_at + int(rec["ttl"]),
            target=rec["target"],
            message=message,
            priority=int(rec["priority"]),
            timer_id=rec["id"],
        ))
        self.events.append({
            "at": "schedule",
            "id": rec["id"],
            "target": rec["target"],
            "fire_at": fire_at,
        })

    def _step_clock(self, rec, i, rebuild, restart_positions) -> None:
        self.clock += int(rec["ticks"])
        self.events.append({
            "at": "clock",
            "ticks": int(rec["ticks"]),
            "new_clock": self.clock,
        })
        due = [t for t in self.timers if t.fire_at <= self.clock]
        due.sort(key=lambda t: (t.fire_at, t.timer_id))
        self.timers = [t for t in self.timers if t.fire_at > self.clock]
        for timer in due:
            if self.clock >= timer.expire_at:
                # Expired at/before it became eligible: never enqueued, never
                # reaches a handler. ttl == 0 expires at the deadline itself.
                self.events.append({
                    "at": "timer_expired",
                    "id": timer.timer_id,
                    "target": timer.target,
                })
                continue
            erased_by_restart = (
                rebuild and timer.target in RESTARTABLE
                and i < restart_positions.get(timer.target, -1)
            )
            delivered = (
                timer.target not in self.stopped and not erased_by_restart
            )
            if delivered:
                self.rt.send(
                    timer.target, timer.message, priority=timer.priority
                )
            self.events.append({
                "at": "timer_fired",
                "id": timer.timer_id,
                "target": timer.target,
                "delivered": delivered,
            })
            self.delivered_dedup.append({
                "dedup": timer.timer_id,
                "duplicate": False,
            })

    def _step_supervision(self, rec) -> None:
        actor = rec["actor"]
        action = rec["action"]
        self.attempts[actor] = self.attempts.get(actor, 0) + 1
        self.events.append({
            "at": "fault",
            "actor": actor,
            "message_id": rec["message_id"],
            "error_type": rec["error_type"],
        })
        self.events.append({
            "at": "supervision",
            "actor": actor,
            "message_id": rec["message_id"],
            "action": action,
        })
        self.supervision.append({
            "actor": actor,
            "message_id": rec["message_id"],
            "error_type": rec["error_type"],
            "action": action,
        })
        if action in ("stop", "escalate"):
            self.stopped.add(actor)
        if action == "escalate" and actor == A_CHILD:
            self.rt.send(A_SUPERVISOR, {"kind": "escalated", "key": actor})

    def supervision_record_for_pending(self) -> dict:
        assert self.pending_fault is not None
        fault = self.pending_fault
        actor = fault["actor"]
        attempts = self.attempts.get(actor, 0) + 1
        return {
            "t": "supervision",
            "actor": actor,
            "message_id": fault["message_id"],
            "error_type": fault["error_type"],
            "action": self.policy(actor, attempts),
        }

    # -- scheduling -------------------------------------------------------

    def _live_actors_with_mail(self) -> bool:
        return any(
            name not in self.stopped and self.rt.pending_count(name) > 0
            for name in ACTOR_NAMES
        )

    def _drain(self) -> None:
        """Process pending messages under the configured cap.

        No cap: run to idle (the runtime's documented registration / priority
        / id order).  Cap N: at most N completions per external input; each
        step is one public ``run(limit=1)``.  No wall-clock waits, no threads.
        """
        if self.pending_fault is not None:
            return
        budget = self.cap
        while self._live_actors_with_mail():
            try:
                if budget is None:
                    self.rt.run()
                    return
                self.rt.run(limit=1)
            except ActorExecutionError as exc:
                self.pending_fault = {
                    "actor": exc.actor_name,
                    "message_id": exc.message_id,
                    "error_type": type(exc.original).__name__,
                }
                return
            budget -= 1
            if budget == 0:
                return

    # -- result -----------------------------------------------------------

    def _states(self) -> dict:
        return {
            name: ({"stopped": True} if name in self.stopped
                   else canonical_state(self.rt.get_state(name)))
            for name in ACTOR_NAMES
        }

    def _live_trace(self) -> list:
        return [
            (
                e.message_id, e.actor_name, e.priority,
                canonical_state(e.state_before),
                canonical_state(e.state_after),
            )
            for e in self.rt.trace()
        ]

    def _result(self, stored_trace: Optional[list] = None) -> ScenarioResult:
        live = self._live_trace()
        traces = (list(stored_trace) + live) if stored_trace else live
        return ScenarioResult(
            states=self._states(),
            traces=traces,
            events=list(self.events),
            supervision=list(self.supervision),
            delivered_dedup=list(self.delivered_dedup),
            restored_len=len(stored_trace) if stored_trace else 0,
            snapshots=list(self.snapshots),
        )


# ---------------------------------------------------------------------------
# Facade: first run / journal replay / snapshot recovery
# ---------------------------------------------------------------------------


def _process(records, *, rebuild_trigger: str, policy):
    """Shared driver: feed records one at a time, rebuilding at boundaries.

    ``rebuild_trigger`` is ``"resolve"`` (first run; record is concretised
    using the live fault) or ``"supervision"`` (replay; record already
    concrete).
    """
    interp = Interpreter(policy=policy)
    i = 0
    while i < len(records):
        rec = records[i]
        if rec.get("t") != rebuild_trigger:
            interp.run_records([rec])
            i += 1
            continue
        if rebuild_trigger == "resolve":
            if interp.pending_fault is None:
                raise InvalidScript("resolve without a pending fault")
            records[i] = interp.supervision_record_for_pending()
        # Rebuild the whole durable prefix through a fresh public runtime,
        # honouring every supervision decision so far.
        rebuilt = Interpreter(policy=policy)
        rebuilt.run_records(records[: i + 1], rebuild=True)
        assert rebuilt.pending_fault is None
        interp = rebuilt
        i += 1
    return interp


def execute(script: list, *,
            policy: Callable[[str, int], str] = default_policy
            ) -> ScenarioResult:
    """First execution: normalise, interpret, and return journal + result."""
    records = normalize_script(script)
    interp = _process(records, rebuild_trigger=RESOLVE, policy=policy)
    journal = encode_records(records)
    result = interp._result()
    result.journal = journal

    # Replace the interpreter's (seq, blob) pairs with (seq, offset, blob),
    # the offset being the snapshot marker frame's position in the journal.
    offsets = frame_offsets(journal)  # [0] = header; record k at index k+1
    located = []
    for seq, blob in interp.snapshots:
        for k, rec in enumerate(records):
            if rec.get("t") == "snapshot" and rec["seq"] == seq:
                located.append((seq, offsets[k + 1], blob))
                break
    result.snapshots = located
    return result


def replay_journal(journal: bytes, *,
                   policy: Callable[[str, int], str] = default_policy
                   ) -> ScenarioResult:
    """Reconstruct a result from durable bytes alone."""
    records = parse_records(journal)
    interp = _process(records, rebuild_trigger="supervision", policy=policy)
    result = interp._result()
    result.journal = journal
    return result


def recover_snapshot(snapshot_blob: bytes, journal: bytes, *,
                     marker_offset: int,
                     policy: Callable[[str, int], str] = default_policy
                     ) -> ScenarioResult:
    """Restore from the snapshot, then apply the journal suffix."""
    interp = Interpreter(policy=policy)
    restored_len = interp.load_snapshot(snapshot_blob)
    stored_trace = list(interp._stored_trace)

    offsets = frame_offsets(journal)
    end_offsets = offsets[1:] + [len(journal)]
    suffix_start = None
    for off, end in zip(offsets, end_offsets):
        if off == marker_offset:
            suffix_start = end
            break
    if suffix_start is None:
        raise CorruptRecordError("snapshot marker not found in journal")
    suffix = (
        parse_records(
            journal, start_offset=suffix_start, require_header=False
        )
        if suffix_start < len(journal) else []
    )
    for rec in suffix:
        if rec.get("t") in ("supervision", "resolve"):
            # Script invariant: supervision always precedes snapshots.
            raise InvalidScript(
                "recovery suffix must not contain supervision records"
            )
        interp.run_records([rec])
    result = interp._result(stored_trace=stored_trace)
    result.journal = journal
    return result
