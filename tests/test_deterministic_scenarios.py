"""Deterministic replay scenario tests for the baseline actor runtime.

Scope mapping
-------------
The baseline ``actor_runtime`` deliberately exposes only an in-memory,
single-process, sequential runtime: ``register / send(priority=...) /
run(limit=...) / get_state / pending_count / trace``.  Its README states that
there is intentionally *no persistence, supervision, timed messages or
parallel scheduling*.  This test module therefore does **not** add product
APIs; the richer concepts from the scenario requirements are expressed as
test-only scaffolding that drives the runtime exclusively through its public
entry points:

* *persisted record* ...... the generated/hand-written finite action log;
  every action is re-issued verbatim against a fresh runtime to "replay".
* *snapshot* .............. a checkpoint of public observables (all actor
  states, all pending counts, the full trace) plus the journal length.
* *recovery / restart* .... rebuild a fresh runtime from the action-log
  prefix, assert it equals the checkpoint, then continue with the suffix.
* *scheduler concurrency
  cap* ..................... the public ``run(limit=...)`` knob; the action
  log records the exact limits, identical on every replay.
* *timed messages* ........ a deterministic virtual clock advanced only by
  explicit ``("tick", n)`` actions; due messages are released through the
  public ``send``; no wall-clock waiting is ever used.
* *message dedup id* ...... a payload-level ``key`` convention; the handler
  applies a keyed state change once, while every delivery still completes
  and therefore still leaves a public ``TraceEntry``.
* *expired messages* ...... a guard in front of the (test) user logic checks
  the virtual deadline; expired deliveries complete (trace + decision are
  observable via ``state_before == state_after``) but never run the user
  body, which is witnessed by the ``touched`` accumulator.
* *supervision* ........... the baseline's only failure rule is the public
  one: a failing handler commits nothing, keeps the message pending and
  escalates (``ActorExecutionError`` for ordinary exceptions, the original
  ``BaseException`` unwrapped otherwise).  Resume = retry the pending
  message; restart = rebuild from the recorded prefix; stop = the poisoned
  mailbox stays halted; escalation = the raised error reaches the caller.

The baseline exposes no persistence entry point, so the conditional
requirement to assert exceptions for *corrupt persisted records* or
*incompatible snapshots* does not apply; the existing contract exceptions
that do exist (``TypeError`` for illegal priority, ``LookupError`` for a
missing target, ``ValueError`` for bad limits/names) are asserted
unchanged.

Observable contract
-------------------
Two executions of the same action log are compared on public data only:
per-run result journal, every actor state, every pending count and every
:class:`TraceEntry` field (message id, actor name, priority, state before,
state after).  Comparisons are both field-level (structured equality) and
byte-level on a canonical JSON encoding.  Wall-clock time, thread ids,
object identity and log text are never consulted, and internal queues are
never touched to fabricate state.
"""
from __future__ import annotations

import copy
import json
import unittest

from actor_runtime import ActorExecutionError, ActorRuntime

ACTORS = ("acc", "relay", "sink")

INITIAL_ACC_STATE = {
    "items": [],   # committed payloads, in processing order
    "once": {},    # dedup key -> first value (exactly-once state change)
    "touched": [],  # values that actually entered the user logic
    "n": 0,
}
INITIAL_SINK_STATE = []


class _HardStop(BaseException):
    """Deterministic non-Exception BaseException used by failure scenarios."""


# ---------------------------------------------------------------------------
# Deterministic pseudo-random source (fixed, dependency free, cross-version)
# ---------------------------------------------------------------------------

class _Lcg:
    """Numerical Recipes style LCG; generation never uses the global RNG."""

    def __init__(self, seed: int) -> None:
        state = (int(seed) & 0xFFFFFFFF) ^ 0x9E3779B9
        self.state = state or 1

    def below(self, n: int) -> int:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state % n

    def integer(self, low: int, high: int) -> int:
        return low + self.below(high - low + 1)

    def choose(self, seq):
        return seq[self.below(len(seq))]

    def weighted(self, weighted_pairs):
        total = sum(weight for _, weight in weighted_pairs)
        roll = self.below(total)
        upto = 0
        for value, weight in weighted_pairs:
            upto += weight
            if roll < upto:
                return value
        return weighted_pairs[-1][0]


# ---------------------------------------------------------------------------
# Scenario execution harness (test-only; talks to public API exclusively)
# ---------------------------------------------------------------------------

class _Env:
    def __init__(self) -> None:
        self.rt = ActorRuntime()
        self.now = 0
        # pending timers: (due, seq, target, payload)
        self.timers: list[tuple[int, int, str, object]] = []
        self.timer_seq = 0
        # transient-failure attempt counts, keyed by message key
        self.attempts: dict[str, int] = {}
        # ("ok", processed) | ("err", actor, msg_id, original_type_name)
        # | ("base", actor, msg_id, exception_type_name)
        self.journal: list[tuple] = []
        # label -> (observables snapshot, journal length)
        self.checkpoints: dict[str, tuple[dict, int]] = {}


def _make_env() -> _Env:
    env = _Env()

    def acc_handler(state, message, ctx):
        # Expiry guard runs *before* any user logic.
        if isinstance(message, dict) and message.get("t") == "exp":
            if env.now > message["deadline"]:
                # Delivery completes and is traced, but the user body is
                # never entered: no item, no touched marker, no counter.
                return state
            value = message["v"]
            state = _clone(state)
            state["items"].append(["exp", value])
            state["touched"].append(value)
            return state

        if isinstance(message, int):
            state = _clone(state)
            state["n"] += message
            return state

        if not isinstance(message, dict):
            return state

        kind = message.get("t")
        if kind == "add":
            key = message["key"]
            if key in state["once"]:
                # Duplicate dedup id: delivery still completes (a trace
                # entry is appended by the runtime) but contributes no
                # state change.
                return state
            state = _clone(state)
            state["once"][key] = message["v"]
            state["items"].append(["add", key, message["v"]])
            state["touched"].append(key)
            return state

        if kind == "fail":
            key = message["key"]
            seen = env.attempts.get(key, 0) + 1
            env.attempts[key] = seen
            if seen == 1:
                raise RuntimeError(f"transient failure {key!r}")
            state = _clone(state)
            state["items"].append(["recovered", key])
            return state

        if kind == "poison":
            # Permanent failure: the message can never succeed in place.
            raise RuntimeError("permanent poison")

        if kind == "basefail":
            raise _HardStop("hard stop")

        return state

    def relay_handler(state, message, ctx):
        # Synchronous call during a handler: the derived send is buffered by
        # the public context and committed only after this handler returns.
        if isinstance(message, dict) and "to" in message:
            ctx.send(message["to"], message["payload"])
        else:
            ctx.send("sink", message)
        return state

    def sink_handler(state, message, ctx):
        state = list(state)
        state.append(copy.deepcopy(message))
        return state

    # Fixed topology and registration order so replays are identical.
    env.rt.register("acc", copy.deepcopy(INITIAL_ACC_STATE), acc_handler)
    env.rt.register("relay", None, relay_handler)
    env.rt.register("sink", copy.deepcopy(INITIAL_SINK_STATE), sink_handler)
    return env


def _clone(value):
    return copy.deepcopy(value)


def _release_due_timers(env: _Env) -> None:
    due = [t for t in env.timers if t[0] <= env.now]
    env.timers = [t for t in env.timers if t[0] > env.now]
    # Deterministic release order: due time first, then timer scheduling id.
    due.sort(key=lambda t: (t[0], t[1]))
    for _, _, target, payload in due:
        env.rt.send(target, payload)


def run_actions(env: _Env, actions: list[tuple]) -> _Env:
    """Apply a finite action log through public runtime calls only."""
    for action in actions:
        tag = action[0]
        if tag == "send":
            _, target, payload, priority = action
            env.rt.send(target, payload, priority=priority)
        elif tag == "timer":
            _, target, payload, delay = action
            env.timers.append(
                (env.now + delay, env.timer_seq, target, copy.deepcopy(payload))
            )
            env.timer_seq += 1
        elif tag == "tick":
            env.now += action[1]
            _release_due_timers(env)
        elif tag == "run":
            limit = action[1]
            try:
                processed = env.rt.run(limit)
            except ActorExecutionError as exc:
                env.journal.append(
                    (
                        "err",
                        exc.actor_name,
                        exc.message_id,
                        type(exc.original).__name__,
                    )
                )
            except BaseException as exc:  # only the deliberate _HardStop
                # A raw BaseException carries no public message id, so only
                # its type is observable contract data.
                env.journal.append(("base", type(exc).__name__))
            else:
                env.journal.append(("ok", processed))
        elif tag == "cp":
            env.checkpoints[action[1]] = (_observe(env), len(env.journal))
        else:  # pragma: no cover - malformed action is a test bug
            raise AssertionError(f"unknown action tag: {tag!r}")
    return env


def execute(actions: list[tuple]) -> _Env:
    return run_actions(_make_env(), actions)


def _observe(env: _Env) -> dict:
    return {
        "states": {name: env.rt.get_state(name) for name in ACTORS},
        "pending": {name: env.rt.pending_count(name) for name in ACTORS},
        "trace": [list(entry) for entry in env.rt.trace()],
    }


def _canonical(observables: dict) -> bytes:
    return json.dumps(
        observables, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _checkpoint_indices(actions: list[tuple]) -> dict[str, int]:
    return {
        action[1]: index
        for index, action in enumerate(actions)
        if action[0] == "cp"
    }


def assert_replay_equivalent(
    actions: list[tuple], *, msg: str = "", must_drain: bool = False
) -> None:
    """Core property: first run, replay, snapshot and restart agree.

    * executing the same action log twice independently yields byte-equal
      canonical observables and equal run-result journals;
    * replaying only the prefix up to each checkpoint reproduces the stored
      snapshot (state, pending counts, trace) and journal prefix;
    * rebuilding from the earliest snapshot and then continuing the suffix
      reproduces the full first run ("restart from record + latest
      snapshot, then keep processing").

    ``must_drain`` additionally requires the final state to be fully idle
    (generated scenarios are always finalised to drain; poison/stop
    scenarios pass the default and assert the blocked state explicitly).
    """
    first = execute(actions)
    second = execute(actions)

    first_obs = _observe(first)
    second_obs = _observe(second)
    first_bytes = _canonical(first_obs)
    second_bytes = _canonical(second_obs)

    prefix = msg or "scenario"
    if first_bytes != second_bytes:
        raise AssertionError(
            f"{prefix}: non-deterministic observables\n"
            f"first : {first_bytes!r}\n"
            f"second: {second_bytes!r}"
        )
    if first.journal != second.journal:
        raise AssertionError(
            f"{prefix}: run-result journals differ\n"
            f"first : {first.journal!r}\n"
            f"second: {second.journal!r}"
        )
    if must_drain:
        leftovers = {
            name: first.rt.pending_count(name)
            for name in ACTORS
            if first.rt.pending_count(name)
        }
        undelivered = len(first.timers)
        if leftovers or undelivered:
            raise AssertionError(
                f"{prefix}: scenario did not drain "
                f"(pending={leftovers}, unreleased_timers={undelivered})"
            )
        if first.journal and first.journal[-1][0] != "ok":
            raise AssertionError(
                f"{prefix}: scenario ended on an unhandled escalation: "
                f"{first.journal[-1]!r}"
            )

    cps = _checkpoint_indices(actions)
    for label, index in cps.items():
        prefix_env = execute(actions[: index + 1])
        saved_obs, saved_journal_len = first.checkpoints[label]
        if _canonical(_observe(prefix_env)) != _canonical(saved_obs):
            raise AssertionError(
                f"{prefix}: replay prefix at checkpoint {label!r} does not "
                f"match the recorded snapshot"
            )
        if prefix_env.journal != first.journal[:saved_journal_len]:
            raise AssertionError(
                f"{prefix}: replayed journal at checkpoint {label!r} "
                f"differs: {prefix_env.journal!r} != "
                f"{first.journal[:saved_journal_len]!r}"
            )

    if cps:
        label = min(cps, key=cps.get)
        index = cps[label]
        rebuilt = execute(actions[: index + 1])          # restart + snapshot
        run_actions(rebuilt, actions[index + 1 :])      # continue suffix
        if _canonical(_observe(rebuilt)) != first_bytes:
            raise AssertionError(
                f"{prefix}: restart from snapshot {label!r} followed by the "
                f"remaining actions diverged from the first execution"
            )
        if rebuilt.journal != first.journal:
            raise AssertionError(
                f"{prefix}: post-restart journal differs: "
                f"{rebuilt.journal!r} != {first.journal!r}"
            )


def _format_failure(seed: int, shrunk: list[tuple]) -> str:
    rendered = "\n".join(f"    {action!r}," for action in shrunk)
    return (
        f"deterministic replay mismatch for seed {seed}; "
        f"generate_actions({seed!r}) reproduces the full scenario, and the "
        f"following minimal action sequence ({len(shrunk)} actions, replayed "
        f"verbatim with execute(...)) reproduces the failure:\n"
        f"[\n{rendered}\n]"
    )


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_actions(seed: int) -> list[tuple]:
    """Generate a finite, replayable action log from a fixed seed.

    The log mixes asynchronous external deliveries (mixed priorities),
    synchronous handler-derived sends (via the relay actor), virtual-time
    timer scheduling and ticks, transient fault injection, duplicate
    dedup-key deliveries, bounded ``run(limit=...)`` scheduling steps and
    snapshot checkpoints.
    """
    rng = _Lcg(seed)
    length = rng.integer(12, 22)

    actions: list[tuple] = []
    keys: list[str] = []
    fail_count = 0

    for _ in range(length):
        kind = rng.weighted(
            [
                ("send", 38),
                ("run", 24),
                ("tick", 13),
                ("timer", 13),
                ("fail", 8),
                ("checkpoint", 4),
            ]
        )

        if kind == "send":
            target = rng.weighted(
                [("acc", 60), ("sink", 20), ("relay", 20)]
            )
            payload = _generate_payload(rng, target, keys)
            priority = rng.integer(-2, 6)
            actions.append(("send", target, payload, priority))

        elif kind == "run":
            limit = rng.weighted([(None, 50), (1, 20), (2, 15), (3, 15)])
            actions.append(("run", limit))

        elif kind == "tick":
            actions.append(("tick", rng.integer(1, 2)))

        elif kind == "timer":
            payload = _generate_timer_payload(rng)
            actions.append(
                ("timer", "acc", payload, rng.integer(1, 3))
            )

        elif kind == "fail":
            fail_count += 1
            actions.append(
                ("send", "acc", {"t": "fail", "key": f"f{fail_count}"}, 0)
            )

        else:
            actions.append(("cp", f"g{len(actions)}"))

    # Deterministic checkpoint positions (at least one split point): insert
    # from the end so earlier indices do not shift.
    body = list(actions)
    actions = list(body)
    for pos in sorted((p for p in (5, 12) if p < len(body)), reverse=True):
        actions.insert(pos, ("cp", f"fixed{pos}"))

    # Finalisation: release every outstanding timer (max delay is 3) and
    # give every surfaced transient failure its retry run.  In the worst
    # case each full run surfaces a single failure and the next one retries
    # it, so 2*fail_count + 2 trailing runs always reach a fully drained,
    # idle runtime (generated scenarios contain no permanent poison).
    actions.append(("tick", 3))
    for _ in range(2 * fail_count + 2):
        actions.append(("run", None))
    return actions


def _generate_payload(rng: _Lcg, target: str, keys: list[str]):
    if target == "sink":
        return rng.choose(
            [rng.integer(0, 999), f"s{rng.integer(0, 9)}",
             {"q": rng.integer(0, 4)}]
        )

    if target == "relay":
        inner = rng.choose(
            [rng.integer(1, 50),
             {"t": "add", "key": f"r{rng.integer(0, 9)}",
              "v": rng.integer(0, 99)}]
        )
        return {"to": rng.choose(["acc", "sink"]), "payload": inner}

    choice = rng.weighted([("int", 30), ("add", 45), ("exp", 25)])
    if choice == "int":
        return rng.integer(-9, 9)
    if choice == "add":
        if keys and rng.below(100) < 45:
            key = rng.choose(keys)  # duplicate dedup id
        else:
            key = f"k{len(keys) + 1}"
            keys.append(key)
        return {"t": "add", "key": key, "v": rng.integer(0, 99)}
    return {
        "t": "exp",
        "deadline": rng.integer(0, 5),
        "v": f"e{rng.integer(0, 99)}",
    }


def _generate_timer_payload(rng: _Lcg) -> dict:
    if rng.below(100) < 50:
        return {
            "t": "exp",
            "deadline": rng.integer(0, 4),
            "v": f"t{rng.integer(0, 99)}",
        }
    return {"t": "add", "key": f"timer{rng.integer(0, 9)}",
            "v": rng.integer(0, 99)}


# ---------------------------------------------------------------------------
# Shrinking helper (test-only): removal-based minimiser
# ---------------------------------------------------------------------------

def _shrink(actions: list[tuple], fails) -> list[tuple]:
    """Greedily remove actions while the property still fails.

    Alternates chunk-level delta debugging and single-action removal until
    no single action can be dropped; returns a locally minimal prefix-style
    reproducer together with the seed reported by the caller.
    """
    current = list(actions)
    changed = True
    while changed:
        changed = False
        n = 2
        while n <= len(current):
            step = max(1, len(current) // n)
            reduced = False
            for start in range(0, len(current), step):
                candidate = current[:start] + current[start + step :]
                if fails(candidate):
                    current = candidate
                    changed = True
                    reduced = True
                    break
            if reduced:
                break
            if n >= len(current):
                break
            n = min(len(current), n * 2)
        if not changed:
            for index in range(len(current)):
                candidate = current[:index] + current[index + 1 :]
                if fails(candidate):
                    current = candidate
                    changed = True
                    break
    return current


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class ReplayDeterminismTests(unittest.TestCase):
    def test_same_seed_produces_byte_identical_trace(self):
        for seed in (1, 2, 17, 4242):
            with self.subTest(seed=seed):
                actions = generate_actions(seed)
                first = execute(actions)
                second = execute(actions)
                self.assertEqual(
                    _canonical(_observe(first)),
                    _canonical(_observe(second)),
                )
                self.assertEqual(first.journal, second.journal)

    def test_generated_corpus_covers_every_scenario_feature(self):
        """The fixed seed corpus must actually exercise every feature; this
        fails loudly if generator weights ever collapse onto one path."""
        seeds = range(1, 201)
        scenarios = [generate_actions(seed) for seed in seeds]
        flat = [action for actions in scenarios for action in actions]

        def sends(actions):
            return [a for a in actions if a[0] == "send"]

        def acc_adds(actions):
            return [
                a[2] for a in sends(actions)
                if isinstance(a[2], dict) and a[2].get("t") == "add"
            ]

        def has_duplicate_dedup_key(actions):
            keys = [m["key"] for m in acc_adds(actions)]
            return len(keys) != len(set(keys))

        def has_fault(actions):
            return any(
                isinstance(a[2], dict) and a[2].get("t") == "fail"
                for a in sends(actions)
            )

        def has_expired_delivery(actions):
            env = execute(actions)
            delivered_exp = 0
            for action in actions:
                if action[0] in ("send", "timer"):
                    payload = action[2]
                    if isinstance(payload, dict) and payload.get("t") == "exp":
                        delivered_exp += 1
            applied = sum(
                1 for item in env.rt.get_state("acc")["items"]
                if item[0] == "exp"
            )
            return applied < delivered_exp

        self.assertGreaterEqual(
            sum(has_duplicate_dedup_key(a) for a in scenarios), 20
        )
        self.assertGreaterEqual(sum(has_fault(a) for a in scenarios), 20)
        self.assertGreaterEqual(
            sum(has_expired_delivery(a) for a in scenarios), 5
        )
        self.assertGreaterEqual(
            sum(any(a[0] == "timer" for a in actions)
                for actions in scenarios), 20
        )
        self.assertGreaterEqual(
            sum(any(a == ("run", 1) for a in actions)
                for actions in scenarios), 10
        )
        self.assertGreaterEqual(
            sum(any(a[0] == "send" and a[1] == "relay" for a in actions)
                for actions in scenarios), 10
        )
        self.assertTrue(all(
            any(a[0] == "cp" for a in actions) for actions in scenarios
        ))

    def test_shrinker_keeps_a_minimal_failing_sequence(self):
        actions = [("tick", 1)] * 10
        # Synthetic failing property: sequences of length >= 4 fail.
        shrunk = _shrink(actions, lambda candidate: len(candidate) >= 4)
        self.assertEqual(len(shrunk), 4)
        # Any single further removal makes the property pass.
        for index in range(len(shrunk)):
            candidate = shrunk[:index] + shrunk[index + 1 :]
            self.assertFalse(len(candidate) >= 4)

    def test_failure_report_is_reproducible_from_seed(self):
        actions = generate_actions(7)
        report = _format_failure(7, actions[:3])
        self.assertIn("seed 7", report)
        self.assertIn("generate_actions(7)", report)
        for action in actions[:3]:
            self.assertIn(repr(action), report)

    def test_generated_scenarios_first_run_matches_replay(self):
        seeds = list(range(200))
        for seed in seeds:
            actions = generate_actions(seed + 1)
            try:
                assert_replay_equivalent(
                    actions, msg=f"seed {seed + 1}", must_drain=True
                )
            except AssertionError:
                def fails(candidate):
                    # Re-finalise every candidate with the idempotent
                    # standard drain suffix so shrinking isolates the
                    # replay divergence itself, never a truncated tail.
                    finalised = (
                        list(candidate)
                        + [("tick", 3)]
                        + [("run", None)] * 32
                    )
                    try:
                        assert_replay_equivalent(
                            finalised, must_drain=True
                        )
                    except AssertionError:
                        return True
                    return False

                shrunk = _shrink(actions, fails)
                self.fail(_format_failure(seed + 1, shrunk))


class MixedMailboxAndSchedulingTests(unittest.TestCase):
    def test_priority_and_registration_order_mix_is_replayable(self):
        actions = [
            ("cp", "start"),
            ("send", "sink", "low", 0),
            ("send", "sink", "high", 10),
            ("send", "sink", "neg", -3),
            ("send", "sink", "mid", 5),
            ("send", "acc", 1, 0),
            ("run", None),
        ]
        env = execute(actions)
        # acc is registered first: its single message drains before sink.
        self.assertEqual(
            [e.actor_name for e in env.rt.trace()],
            ["acc", "sink", "sink", "sink", "sink"],
        )
        # Higher priority first, FIFO by id on ties, negatives last.
        self.assertEqual(env.rt.get_state("sink"),
                         ["high", "mid", "low", "neg"])
        self.assertEqual(env.rt.get_state("acc")["n"], 1)
        assert_replay_equivalent(actions, msg="mixed mailbox")

    def test_synchronous_derived_send_interleaves_with_external_delivery(self):
        actions = [
            ("cp", "start"),
            # id 1: relay will synchronously ctx.send acc=100 after return
            ("send", "relay", {"to": "acc", "payload": 100}, 0),
            # id 2: external async delivery straight to acc
            ("send", "acc", 1, 0),
            ("run", None),
        ]
        env = execute(actions)
        trace = env.rt.trace()
        # acc is selected first (id 2); relay then completes and only after
        # its return is the derived id-3 message enqueued and processed.
        self.assertEqual(
            [(e.message_id, e.actor_name) for e in trace],
            [(2, "acc"), (1, "relay"), (3, "acc")],
        )
        self.assertEqual(env.rt.get_state("acc")["n"], 101)
        assert_replay_equivalent(actions, msg="sync/async interleave")

    def test_run_limit_chunks_are_stable_across_replay(self):
        actions = [("cp", "start")]
        for i in range(6):
            actions.append(("send", "acc", i, 0))
        actions += [
            ("run", 1),
            ("cp", "one"),
            ("run", 1),
            ("run", 3),
            ("cp", "five"),
            ("run", None),
        ]
        first = execute(actions)
        second = execute(actions)
        self.assertEqual(first.journal, second.journal)
        self.assertEqual(
            first.journal,
            [("ok", 1), ("ok", 1), ("ok", 3), ("ok", 1)],
        )
        self.assertEqual(
            [e.message_id for e in first.rt.trace()], [1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(first.rt.get_state("acc")["n"], sum(range(6)))
        assert_replay_equivalent(actions, msg="bounded scheduling")


class TimerAndExpiryTests(unittest.TestCase):
    def test_due_timer_processed_and_expired_one_skips_user_logic(self):
        actions = [
            ("cp", "start"),
            # Released at virtual time 3, deadline 1 -> already expired.
            ("timer", "acc",
             {"t": "exp", "deadline": 1, "v": "late"}, 2),
            # Released at virtual time 1, deadline 5 -> still valid.
            ("timer", "acc",
             {"t": "exp", "deadline": 5, "v": "ontime"}, 1),
            ("run", None),  # nothing released yet
            ("tick", 1),
            ("run", None),  # "ontime" delivered and applied
            ("cp", "after-ontime"),
            ("tick", 2),    # now == 3: "late" released
            ("run", None),  # expired: traced, decision visible, no body
        ]
        env = execute(actions)
        acc = env.rt.get_state("acc")
        self.assertEqual(acc["touched"], ["ontime"])
        self.assertEqual(acc["items"], [["exp", "ontime"]])
        entries = env.rt.trace()
        # Both deliveries completed and left a trace entry ...
        self.assertEqual(len(entries), 2)
        expired_entry = entries[-1]
        # ... but the expired one shows an empty state delta.
        self.assertEqual(expired_entry.state_before, expired_entry.state_after)
        self.assertEqual(env.rt.pending_count("acc"), 0)
        # The pre-expiry run had no pending messages.
        self.assertEqual(env.journal[0], ("ok", 0))
        assert_replay_equivalent(actions, msg="timer expiry")

    def test_timers_release_in_due_then_schedule_order(self):
        actions = [
            ("cp", "start"),
            ("timer", "sink", "c", 2),
            ("timer", "sink", "a", 1),
            ("timer", "sink", "b", 1),
            ("tick", 2),
            ("run", None),
        ]
        env = execute(actions)
        self.assertEqual(env.rt.get_state("sink"), ["a", "b", "c"])
        assert_replay_equivalent(actions, msg="timer order")


class DedupTests(unittest.TestCase):
    def test_duplicate_dedup_id_changes_state_once_but_each_delivery_traced(self):
        payload = {"t": "add", "key": "k1", "v": 7}
        actions = [
            ("cp", "start"),
            ("send", "acc", copy.deepcopy(payload), 0),
            ("send", "acc", copy.deepcopy(payload), 2),
            ("send", "acc", copy.deepcopy(payload), 0),
            ("run", None),
        ]
        env = execute(actions)
        entries = env.rt.trace()
        self.assertEqual(len(entries), 3)
        changed = [e.state_before != e.state_after for e in entries]
        # Priority does not change dedup semantics: first completion wins.
        self.assertEqual(changed, [True, False, False])
        acc = env.rt.get_state("acc")
        self.assertEqual(acc["once"], {"k1": 7})
        self.assertEqual(acc["items"], [["add", "k1", 7]])
        self.assertEqual(acc["touched"], ["k1"])
        assert_replay_equivalent(actions, msg="dedup")


class FailureSupervisionTests(unittest.TestCase):
    def test_transient_failure_resume_and_restart_share_one_outcome(self):
        actions = [
            ("send", "acc", 5, 0),
            ("run", None),                # id 1 commits
            ("cp", "healthy"),
            ("send", "acc", {"t": "fail", "key": "f1"}, 0),  # id 2
            ("send", "sink", "after", 0),
            ("run", None),                # escalates; id 2 stays pending
            ("cp", "failed"),
            ("run", None),                # resume: retry id 2, then sink drains
        ]
        env = execute(actions)
        self.assertEqual(
            env.journal,
            [("ok", 1), ("err", "acc", 2, "RuntimeError"), ("ok", 2)],
        )
        failed_cp = env.checkpoints["failed"][0]
        self.assertEqual(failed_cp["states"]["acc"]["n"], 5)
        self.assertEqual(failed_cp["pending"], {"acc": 1, "relay": 0,
                                                "sink": 1})
        self.assertEqual(
            [e[0] for e in failed_cp["trace"]], [1]
        )

        acc = env.rt.get_state("acc")
        self.assertEqual(acc["n"], 5)
        self.assertEqual(acc["items"], [["recovered", "f1"]])
        self.assertEqual(env.rt.get_state("sink"), ["after"])
        # The retried message kept its original id; nothing was committed
        # twice.
        self.assertEqual(
            [e.message_id for e in env.rt.trace()], [1, 2, 3]
        )
        assert_replay_equivalent(actions, msg="transient failure")

    def test_permanent_poison_stops_mailbox_and_restart_replays_escalation(self):
        healthy_prefix = [
            ("send", "acc", 1, 0),
            ("run", None),
            ("cp", "healthy"),
        ]
        poisoned = healthy_prefix + [
            ("send", "acc", {"t": "poison"}, 0),
            # sink is registered after acc: while acc's mailbox is blocked,
            # it is never selected (documented baseline scheduling rule).
            ("send", "sink", "trapped", 0),
            ("run", None),
            ("cp", "stopped"),
            ("run", None),  # restart in place: deterministic re-escalation
        ]
        env = execute(poisoned)
        self.assertEqual(
            env.journal,
            [("ok", 1), ("err", "acc", 2, "RuntimeError"),
             ("err", "acc", 2, "RuntimeError")],
        )
        stopped = env.checkpoints["stopped"][0]
        self.assertEqual(stopped["states"]["acc"]["n"], 1)
        self.assertEqual(stopped["pending"], {"acc": 1, "relay": 0,
                                              "sink": 1})
        self.assertEqual(len(stopped["trace"]), 1)

        # Restart from the durable prefix reproduces the healthy snapshot.
        rebuilt = execute(healthy_prefix)
        self.assertEqual(
            _canonical(_observe(rebuilt)),
            _canonical(env.checkpoints["healthy"][0]),
        )
        # Replaying the record including the poison re-escalates the same
        # way after restart.
        replayed = execute(healthy_prefix + [
            ("send", "acc", {"t": "poison"}, 0),
            ("run", None),
        ])
        self.assertEqual(replayed.journal[-1],
                         ("err", "acc", 2, "RuntimeError"))

        # After supervisor stops the poisoned message, the still-valid
        # messages process normally on the rebuilt runtime.
        recovered = execute(healthy_prefix + [
            ("send", "acc", 2, 0),
            ("send", "sink", "after", 0),
            ("run", None),
        ])
        self.assertEqual(recovered.rt.get_state("acc")["n"], 3)
        self.assertEqual(recovered.rt.get_state("sink"), ["after"])
        self.assertEqual(
            [(e.message_id, e.actor_name) for e in recovered.rt.trace()],
            [(1, "acc"), (2, "acc"), (3, "sink")],
        )

    def test_base_exception_propagates_unwrapped_and_replays_identically(self):
        actions = [
            ("send", "acc", 1, 0),
            ("run", None),
            ("cp", "healthy"),
            ("send", "acc", {"t": "basefail"}, 0),
            ("run", None),
            ("cp", "aborted"),
        ]
        first = execute(actions)
        second = execute(actions)
        self.assertEqual(first.journal, second.journal)
        self.assertEqual(first.journal[-1], ("base", "_HardStop"))
        aborted = first.checkpoints["aborted"][0]
        self.assertEqual(aborted["pending"]["acc"], 1)
        self.assertEqual(len(aborted["trace"]), 1)
        # Rebuild from the healthy snapshot and continue with valid work:
        # the escalation left exactly one healthy trace entry behind.
        recovered = execute(actions[:3])
        run_actions(
            recovered,
            [("send", "acc", 9, 0), ("run", None)],
        )
        self.assertEqual(recovered.rt.get_state("acc")["n"], 10)

    def test_unknown_target_from_handler_escalates_and_commits_nothing(self):
        actions = [
            ("cp", "start"),
            ("send", "relay", {"to": "ghost", "payload": 1}, 0),
            ("run", None),
        ]
        env = execute(actions)
        self.assertEqual(env.journal[0][:2], ("err", "relay"))
        self.assertEqual(env.journal[0][3], "LookupError")
        self.assertEqual(env.rt.pending_count("relay"), 1)
        self.assertEqual(env.rt.trace(), [])
        assert_replay_equivalent(actions, msg="handler lookup failure")


class ContractErrorTests(unittest.TestCase):
    """Pre-existing public error protocol must stay exactly as defined."""

    def test_illegal_priority_type_error_consumes_no_id(self):
        for bad in (1.0, "1", None, [1], True):
            with self.subTest(bad=bad):
                rt = ActorRuntime()
                rt.register("acc", None, lambda s, m, c: s)
                with self.assertRaises(TypeError):
                    rt.send("acc", "x", priority=bad)
                self.assertEqual(rt.pending_count("acc"), 0)

        rt = ActorRuntime()
        rt.register("acc", None, lambda s, m, c: s)
        self.assertEqual(rt.send("acc", "ok"), 1)
        with self.assertRaises(TypeError):
            rt.send("acc", "bad", priority=1.5)
        with self.assertRaises(LookupError):
            rt.send("ghost", "bad")
        self.assertEqual(rt.send("acc", "ok2"), 2)

    def test_missing_target_and_bad_registration_raise_defined_types(self):
        rt = ActorRuntime()
        with self.assertRaises(LookupError):
            rt.send("ghost", 1)
        rt.register("acc", 0, lambda s, m, c: s)
        with self.assertRaises(ValueError):
            rt.register("", 0, lambda s, m, c: s)
        with self.assertRaises(ValueError):
            rt.register("acc", 0, lambda s, m, c: s)
        with self.assertRaises(LookupError):
            rt.get_state("ghost")
        with self.assertRaises(LookupError):
            rt.pending_count("ghost")

    def test_bad_limit_value_error_consumes_nothing(self):
        rt = ActorRuntime()
        rt.register("acc", [], lambda s, m, c: list(s) + [m])
        rt.send("acc", 1)
        for bad in (0, -1, 1.5, True, "2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rt.run(limit=bad)
        self.assertEqual(rt.pending_count("acc"), 1)
        self.assertEqual(rt.trace(), [])

    def test_contract_errors_replay_identically_as_an_action_log(self):
        # Invalid deliveries are rejected before entering the mailbox; they
        # are not actions of the log (the log only contains accepted
        # deliveries), and the accepted sequence around them still replays.
        actions = [
            ("cp", "start"),
            ("send", "acc", 1, 0),
            ("run", None),
            ("send", "sink", "x", 4),
            ("run", 2),
        ]
        assert_replay_equivalent(actions, msg="contract replay")


if __name__ == "__main__":
    unittest.main()
