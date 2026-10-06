"""Deterministic snapshot codec for :class:`ActorRuntime`.

A snapshot is a single canonical JSON document (UTF-8 bytes). It carries
*data* only -- states, queued messages, dedup records, trace entries, timer
deadlines and the runtime counters -- never handlers or any other executable
object. Recovery rebuilds an :class:`ActorRuntime` purely from supplied
callables, so loading a snapshot imports nothing and runs no snapshot-carried
code.

Business values use a small tagged form:

``None``, booleans, integers and strings encode as plain JSON values;
finite floats are tagged ``{"f": "3.5"}`` (with signed zero canonicalised to
``0.0``); ``bytes`` are tagged ``{"b": "base64"}``; lists are ``{"l": [...]}``
and tuples the distinct ``{"t": [...]}``; string-keyed dicts are
``{"d": [[key, value], ...]}`` with entries sorted by key. The container
tags are explicit because lists and tuples must round-trip distinctly even
inside other containers, and the dict tagging guarantees byte-identical
snapshots regardless of insertion order.

Every malformed, truncated, tampered, unsupported-version or internally
inconsistent blob -- and every value outside the supported data language --
is reported as :class:`SnapshotError`. Encoding never mutates runtime
structures, and a raised :class:`SnapshotError` commits nothing.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any

__all__ = [
    "SnapshotError",
    "SNAPSHOT_VERSION",
    "encode_value",
    "decode_value",
    "canonical_dumps",
    "decode_snapshot",
]

SNAPSHOT_VERSION = 1


class SnapshotError(Exception):
    """Raised when a snapshot cannot be exported or restored.

    Export raises it for business data outside the supported value language
    (sets, custom instances, non-string mapping keys, NaN/infinite floats or
    a container cycle); the runtime is left exactly as it was. Restore raises
    it for truncated, tampered, unsupported, incomplete or internally
    inconsistent bytes; no partially restored runtime is ever returned.
    """


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class _CycleDetector:
    """Identity tracking for the containers currently being encoded."""

    __slots__ = ("_active",)

    def __init__(self) -> None:
        self._active: set[int] = set()

    def push(self, obj: Any) -> None:
        marker = id(obj)
        if marker in self._active:
            raise SnapshotError("cyclic reference is not snapshotable")
        self._active.add(marker)

    def pop(self, obj: Any) -> None:
        self._active.discard(id(obj))


def encode_value(value: Any, guards: _CycleDetector | None = None) -> Any:
    """Encode one supported business value into canonical JSON data.

    Raises :class:`SnapshotError` for any value outside the supported
    language; never mutates ``value``.
    """
    if guards is None:
        guards = _CycleDetector()
    # bool is a subclass of int: it must be tested first and kept distinct.
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SnapshotError(
                "only finite floats are snapshotable; got NaN or infinity"
            )
        if value == 0.0:
            # 0.0 and -0.0 compare equal but render differently; one zero
            # keeps the byte-for-byte identity contract.
            value = 0.0
        return {"f": repr(value)}
    if isinstance(value, bytes):
        return {"b": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        guards.push(value)
        try:
            return {"t": [encode_value(item, guards) for item in value]}
        finally:
            guards.pop(value)
    if isinstance(value, list):
        guards.push(value)
        try:
            return {"l": [encode_value(item, guards) for item in value]}
        finally:
            guards.pop(value)
    if isinstance(value, dict):
        guards.push(value)
        try:
            entries = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise SnapshotError(
                        "only strings are snapshotable mapping keys"
                    )
                entries.append((key, encode_value(item, guards)))
        finally:
            guards.pop(value)
        # Sorting keys makes two equal mappings with different insertion
        # orders produce identical bytes. Keys are unique strings, so a plain
        # code-point order is unambiguous.
        entries.sort(key=lambda entry: entry[0])
        return {"d": entries}
    raise SnapshotError(
        f"value of type {type(value).__name__} is not snapshotable"
    )


def canonical_dumps(document: Any) -> bytes:
    """Serialise an already-encoded envelope deterministically."""
    # allow_nan=False is defence in depth: finite floats are already tagged
    # as strings, so no NaN/Infinity token can reach the encoder anyway.
    try:
        text = json.dumps(
            document, ensure_ascii=True, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotError(f"failed to encode snapshot: {exc}") from exc
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _fail(reason: str) -> None:
    raise SnapshotError(reason)


def _reject_constant(token: str) -> Any:
    # json calls parse_constant for NaN/Infinity/-Infinity tokens.
    _fail(f"snapshot contains a non-finite token: {token}")


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> Any:
    # object_pairs_hook: refuse any JSON object with repeated keys so a
    # tampered document can never smuggle a second shadowed field.
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(f"snapshot object contains a duplicate key: {key!r}")
        result[key] = value
    return result


def decode_value(value: Any, what: str) -> Any:
    """Decode one tagged business value, rejecting any unknown shape.

    Cycles cannot occur in freshly parsed JSON, so no guard is needed.
    """
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        # The encoder never emits a bare float; one here means a forged doc.
        _fail(f"{what} contains an untagged float")
    if isinstance(value, list):
        # Lists are always tagged {"l": ...}, so a bare list is a forgery.
        _fail(f"{what} contains an untagged list")
    if isinstance(value, dict):
        keys = value.keys()
        if len(keys) == 1:
            if "f" in value:
                text = value["f"]
                if not isinstance(text, str):
                    _fail(f"{what} float tag must hold a string")
                try:
                    number = float(text)
                except ValueError:
                    _fail(f"{what} float tag is not a number: {text!r}")
                if not math.isfinite(number):
                    _fail(f"{what} float tag is not finite")
                if number == 0.0:
                    number = 0.0  # normalise any forged -0.0 spelling
                return number
            if "b" in value:
                text = value["b"]
                if not isinstance(text, str):
                    _fail(f"{what} bytes tag must hold a string")
                try:
                    return base64.b64decode(text.encode("ascii"), validate=True)
                except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
                    _fail(f"{what} bytes tag is not valid base64: {exc}")
            if "l" in value:
                items = value["l"]
                if not isinstance(items, list):
                    _fail(f"{what} list tag must hold a list")
                return [
                    decode_value(item, f"{what} item") for item in items
                ]
            if "t" in value:
                items = value["t"]
                if not isinstance(items, list):
                    _fail(f"{what} tuple tag must hold a list")
                return tuple(
                    decode_value(item, f"{what} item") for item in items
                )
            if "d" in value:
                pairs = value["d"]
                if not isinstance(pairs, list):
                    _fail(f"{what} dict tag must hold a list")
                result: dict[str, Any] = {}
                previous: str | None = None
                for index, pair in enumerate(pairs):
                    if not isinstance(pair, list) or len(pair) != 2:
                        _fail(
                            f"{what} dict entry {index} must be a "
                            f"[key, value] pair"
                        )
                    key = pair[0]
                    if not isinstance(key, str):
                        _fail(f"{what} dict entry {index} key must be a string")
                    if previous is not None and not previous < key:
                        _fail(
                            f"{what} dict keys must be unique and sorted: "
                            f"{previous!r} before {key!r}"
                        )
                    result[key] = decode_value(
                        pair[1], f"{what} dict entry {key!r}"
                    )
                    previous = key
                return result
        _fail(f"{what} uses an unknown tagged object")
    _fail(f"{what} contains a value of unsupported JSON type")


def _as_int(value: Any, what: str) -> int:
    # bool must not be accepted where a counter/deadline is expected.
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{what} must be a non-boolean integer")
    return value


def _as_envelope(value: Any, what: str, exact: set[str]) -> dict[str, Any]:
    """Check a raw (structurally walked, not tagged) envelope object."""
    if not isinstance(value, dict):
        _fail(f"{what} must be an object")
    if set(value.keys()) != exact:
        _fail(f"{what} has wrong fields: expected {sorted(exact)}")
    return value


def _check(condition: bool, reason: str) -> None:
    if not condition:
        _fail(reason)


_ROOT_FIELDS = {"v", "clock", "next_id", "actors", "scheduled", "trace"}
_ACTOR_FIELDS = {"name", "order", "state", "mailbox", "dedup"}
_TIMER_FIELDS = {
    "id", "actor", "priority", "message", "release_at", "expire_at",
}
_TRACE_FIELDS = {"id", "actor", "priority", "state_before", "state_after"}


def decode_snapshot(data: bytes) -> dict[str, Any]:
    """Parse and strictly validate a snapshot, returning its plain payload.

    The envelope is walked as raw JSON; only business values (states and
    messages) go through the tagged :func:`decode_value`. The payload is the
    runtime's private contract with :meth:`ActorRuntime.export_snapshot`:
    this function authenticates framing, version, field presence and internal
    referential consistency before any runtime is rebuilt. All returned
    values are ordinary Python objects freshly built from the bytes -- never
    shared with the caller's objects.
    """
    if not isinstance(data, (bytes, bytearray)):
        _fail("snapshot must be bytes")
    raw = bytes(data)
    try:
        document = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate,
        )
    except SnapshotError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SnapshotError(f"snapshot is not valid JSON: {exc}") from exc

    # Authenticate the exact canonical form, not just an equivalent JSON
    # value: appended whitespace, reordered keys or altered number spelling
    # are tampering even though they parse to the same document. Genuine
    # snapshots come from canonical_dumps and survive this round trip
    # byte-for-byte.
    if canonical_dumps(document) != raw:
        _fail("snapshot is not in canonical form")

    root = _as_envelope(document, "snapshot root", _ROOT_FIELDS)

    version = _as_int(root["v"], "snapshot version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"unsupported snapshot version: {version} "
            f"(supported: {SNAPSHOT_VERSION})"
        )

    clock = _as_int(root["clock"], "logical clock")
    _check(clock >= 0, "logical clock must be non-negative")
    next_id = _as_int(root["next_id"], "next message id")
    _check(next_id >= 1, "next message id must be at least 1")

    def id_in_sequence(value: int, what: str) -> None:
        _check(1 <= value < next_id, f"{what} falls outside the id sequence")

    # -- actors ------------------------------------------------------------
    actors_field = root["actors"]
    _check(isinstance(actors_field, list), "actors field must be a list")
    actors: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, raw_entry in enumerate(actors_field):
        where = f"actor entry {index}"
        entry = _as_envelope(raw_entry, where, _ACTOR_FIELDS)
        name = entry["name"]
        _check(isinstance(name, str) and name != "",
               f"{where} name must be a non-empty string")
        _check(name not in seen_names, f"duplicate actor name: {name!r}")
        seen_names.add(name)
        order = _as_int(entry["order"], f"{where} registration order")
        _check(order == index, f"{where} registration order is not contiguous")

        mailbox_field = entry["mailbox"]
        _check(isinstance(mailbox_field, list), f"{where} mailbox must be a list")
        mailbox: list[tuple[int, int, Any]] = []
        for m_index, mail in enumerate(mailbox_field):
            m_where = f"{where} mailbox entry {m_index}"
            _check(isinstance(mail, list) and len(mail) == 3,
                   f"{m_where} must be [priority, message_id, message]")
            priority = _as_int(mail[0], f"{m_where} priority")
            message_id = _as_int(mail[1], f"{m_where} message id")
            message = decode_value(mail[2], f"{m_where} message")
            if m_index > 0:
                # Entries are serialised in canonical heap order:
                # (-priority, id), strictly increasing.
                prev_priority, prev_id = mailbox[-1][0], mailbox[-1][1]
                _check(
                    (-prev_priority, prev_id) < (-priority, message_id),
                    f"{m_where} is out of the canonical mailbox order"
                )
            mailbox.append((priority, message_id, message))

        dedup_field = entry["dedup"]
        _check(isinstance(dedup_field, dict),
               f"{where} dedup table must be an object")
        dedup: dict[str, int] = {}
        for key, mapped in dedup_field.items():
            _check(isinstance(key, str) and key != "",
                   f"{where} dedup key must be a non-empty string")
            dedup[key] = _as_int(mapped, f"{where} dedup mapping for {key!r}")

        actors.append({
            "name": name,
            "order": order,
            "state": decode_value(entry["state"], f"{where} state"),
            "mailbox": mailbox,
            "dedup": dedup,
        })

    # -- scheduled deliveries ---------------------------------------------
    scheduled_field = root["scheduled"]
    _check(isinstance(scheduled_field, list), "scheduled field must be a list")
    scheduled: list[dict[str, Any]] = []
    scheduled_ids: set[int] = set()
    previous_timer_id = 0
    for index, raw_entry in enumerate(scheduled_field):
        where = f"scheduled entry {index}"
        entry = _as_envelope(raw_entry, where, _TIMER_FIELDS)
        message_id = _as_int(entry["id"], f"{where} message id")
        id_in_sequence(message_id, f"{where} message id")
        _check(message_id not in scheduled_ids,
               f"{where} repeats scheduled message id {message_id}")
        _check(message_id > previous_timer_id,
               f"{where} scheduled entries must be sorted by id")
        previous_timer_id = message_id
        scheduled_ids.add(message_id)

        target = entry["actor"]
        _check(isinstance(target, str) and target in seen_names,
               f"{where} targets an unknown actor")
        priority = _as_int(entry["priority"], f"{where} priority")
        release_at = _as_int(entry["release_at"], f"{where} release_at")
        expire_raw = entry["expire_at"]
        expire_at = None if expire_raw is None else _as_int(
            expire_raw, f"{where} expire_at"
        )
        # Only observable deadlines are stored: the original scheduling tick
        # is invisible through the public API. A pending delivery is one
        # advance has not yet settled: the deadline is still in the future,
        # and -- when the ttl runs out no later than the deadline -- so is
        # the expiry tick. Equality release_at == expire_at is a legitimate
        # not-yet-settled state (expiry wins once the clock reaches it).
        _check(release_at > clock,
               f"{where} is past its deadline but still pending")
        if expire_at is not None:
            if expire_at <= release_at:
                _check(expire_at > clock,
                       f"{where} has expired but is still pending")

        scheduled.append({
            "id": message_id,
            "actor": target,
            "priority": priority,
            "message": decode_value(entry["message"], f"{where} message"),
            "release_at": release_at,
            "expire_at": expire_at,
        })

    # -- trace (kept in completion order, never reordered) ----------------
    trace_field = root["trace"]
    _check(isinstance(trace_field, list), "trace field must be a list")
    trace: list[dict[str, Any]] = []
    trace_ids: set[int] = set()
    trace_actor_by_id: dict[int, str] = {}
    for index, raw_entry in enumerate(trace_field):
        where = f"trace entry {index}"
        entry = _as_envelope(raw_entry, where, _TRACE_FIELDS)
        message_id = _as_int(entry["id"], f"{where} message id")
        id_in_sequence(message_id, f"{where} message id")
        _check(message_id not in trace_ids,
               f"{where} repeats message id {message_id}")
        trace_ids.add(message_id)
        actor_name = entry["actor"]
        _check(isinstance(actor_name, str) and actor_name in seen_names,
               f"{where} names an unknown actor")
        trace_actor_by_id[message_id] = actor_name
        trace.append({
            "id": message_id,
            "actor": actor_name,
            "priority": _as_int(entry["priority"], f"{where} priority"),
            "state_before": decode_value(
                entry["state_before"], f"{where} state_before"
            ),
            "state_after": decode_value(
                entry["state_after"], f"{where} state_after"
            ),
        })

    # -- cross-structure consistency --------------------------------------
    mailbox_ids: set[int] = set()
    dedup_original_ids: set[int] = set()
    for actor in actors:
        own: set[int] = set()
        for _priority, message_id, _message in actor["mailbox"]:
            id_in_sequence(message_id, f"mailbox for {actor['name']!r}")
            _check(message_id not in mailbox_ids,
                   f"message id {message_id} appears in two mailboxes")
            mailbox_ids.add(message_id)
            own.add(message_id)
        for key, original_id in actor["dedup"].items():
            id_in_sequence(original_id, f"dedup key {key!r}")
            _check(original_id not in dedup_original_ids,
                   f"message id {original_id} backs dedup keys for "
                   f"more than one record")
            dedup_original_ids.add(original_id)
            _check(original_id not in scheduled_ids,
                   f"dedup key {key!r} for actor {actor['name']!r} "
                   f"references a still-scheduled delivery")
            # The first delivery for a key is either still waiting in this
            # actor's mailbox (also after a handler failure) or has already
            # completed in this actor's trace.
            if original_id in trace_ids:
                _check(trace_actor_by_id[original_id] == actor["name"],
                       f"dedup key {key!r} for actor {actor['name']!r} "
                       f"references a delivery completed for another actor")
            else:
                _check(original_id in own,
                       f"dedup key {key!r} for actor {actor['name']!r} "
                       f"references id {original_id} with no matching delivery")

    # An id admitted to a mailbox never simultaneously waits on its deadline,
    # and neither kind of pending delivery can already be in the trace.
    _check(mailbox_ids.isdisjoint(scheduled_ids),
           "a message id is both scheduled and present in a mailbox")
    pending_ids = mailbox_ids | scheduled_ids
    _check(pending_ids.isdisjoint(trace_ids),
           "a message id is both pending and present in the trace")

    # Ids whose timer expired before release leave no residue, so issued ids
    # may legitimately be absent from every structure; that is
    # indistinguishable from real expiry and harmless. What must never happen
    # is duplication or a dangling reference, both rejected above.

    return {
        "version": version,
        "clock": clock,
        "next_id": next_id,
        "actors": actors,
        "scheduled": scheduled,
        "trace": trace,
    }
