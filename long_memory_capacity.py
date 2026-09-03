#!/usr/bin/env python3
"""Deterministic long-history cases and A/B/C prompt construction.

The benchmark owns its facts, rules, and state transitions.  V0 deliberately
does not claim free-form semantic extraction from arbitrary prose.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Protocol

import torch


Arm = Literal["A", "B", "C"]
Task = Literal["distant_recall", "causal_reasoning"]
SCHEMA = "helix.long-memory-capacity-case.v0"
EVALUATOR_SCHEMA = "helix.long-memory-abc-evaluator.v0"


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_root(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    kind: str
    subject: str
    predicate: str
    value: str
    parents: tuple[str, ...] = ()
    supersedes: str | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    schema: str
    case_id: str
    task: Task
    history_distance: int
    query: str
    answer: str
    filler_token_id: int
    prefix_token_ids: tuple[int, ...]
    events: tuple[MemoryEvent, ...]
    current_state: tuple[tuple[str, str], ...]
    causal_path: tuple[str, ...]

    def contract(self) -> dict[str, Any]:
        body = asdict(self)
        body["prefix_token_root"] = hashlib.sha256(
            torch.tensor(self.prefix_token_ids, dtype=torch.int32).numpy().tobytes()
        ).hexdigest()
        del body["prefix_token_ids"]
        return body


def _encode(tokenizer: Tokenizer, text: str) -> tuple[int, ...]:
    ids = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError(f"text produced no tokens: {text!r}")
    return ids


def build_cases(
    tokenizer: Tokenizer,
    *,
    distances: Iterable[int] = (10_000, 50_000, 250_000, 750_000),
    seed: int = 42,
) -> list[BenchmarkCase]:
    distances = tuple(int(value) for value in distances)
    if not distances or any(value <= 0 for value in distances):
        raise ValueError("history distances must be positive")
    filler_ids = _encode(tokenizer, " ordinary background detail")
    filler_token_id = filler_ids[(seed * 17) % len(filler_ids)]
    cases: list[BenchmarkCase] = []
    for distance in distances:
        recall_events = (
            MemoryEvent("r1", "assign", "vault-kestrel", "access-code", "amber-17"),
            MemoryEvent(
                "r2",
                "assign",
                "vault-kestrel",
                "access-code",
                "cobalt-29",
                supersedes="r1",
            ),
        )
        recall_prefix = _encode(
            tokenizer,
            "Record r1: vault-kestrel access-code is amber-17. "
            "Record r2 supersedes r1: vault-kestrel access-code is cobalt-29. ",
        )
        cases.append(
            BenchmarkCase(
                schema=SCHEMA,
                case_id=f"recall-{distance}",
                task="distant_recall",
                history_distance=distance,
                query="Question: what is the current access-code for vault-kestrel? Answer:",
                answer=" cobalt-29",
                filler_token_id=filler_token_id,
                prefix_token_ids=recall_prefix,
                events=recall_events,
                current_state=(("vault-kestrel.access-code", "cobalt-29"),),
                causal_path=("r2",),
            )
        )

        causal_events = (
            MemoryEvent("c1", "fact", "switch-iona", "state", "on"),
            MemoryEvent("c2", "rule", "switch-iona:on", "activates", "relay-pavo"),
            MemoryEvent(
                "c3",
                "rule",
                "relay-pavo:active",
                "illuminates",
                "beacon-orin",
                parents=("c2",),
            ),
            MemoryEvent(
                "c4",
                "derived",
                "beacon-orin",
                "state",
                "glowing",
                parents=("c1", "c2", "c3"),
            ),
        )
        causal_prefix = _encode(
            tokenizer,
            "Fact c1: switch-iona is on. Rule c2: when switch-iona is on, relay-pavo activates. "
            "Rule c3: when relay-pavo activates, beacon-orin glows. ",
        )
        cases.append(
            BenchmarkCase(
                schema=SCHEMA,
                case_id=f"causal-{distance}",
                task="causal_reasoning",
                history_distance=distance,
                query="Question: after applying the rules, what is beacon-orin doing? Answer:",
                answer=" glowing",
                filler_token_id=filler_token_id,
                prefix_token_ids=causal_prefix,
                events=causal_events,
                current_state=(
                    ("beacon-orin.state", "glowing"),
                    ("switch-iona.state", "on"),
                ),
                causal_path=("c1", "c2", "c3", "c4"),
            )
        )
    return cases


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9-]+", text.lower()) if len(term) > 2}


def raw_event_documents(case: BenchmarkCase) -> list[str]:
    documents = []
    for event in case.events:
        if event.kind == "derived":
            continue
        parents = f" parents={','.join(event.parents)}" if event.parents else ""
        supersedes = f" supersedes={event.supersedes}" if event.supersedes else ""
        documents.append(
            f"{event.event_id} {event.kind}: {event.subject} {event.predicate} {event.value}"
            f"{parents}{supersedes}"
        )
    return documents


def compile_automata_state(case: BenchmarkCase) -> tuple[tuple[str, str], ...]:
    """Replay the benchmark's typed transitions into one bounded live state."""
    state: dict[str, str] = {}
    observed: set[str] = set()
    for event in case.events:
        if event.event_id in observed:
            raise ValueError(f"duplicate event id: {event.event_id}")
        if event.supersedes is not None and event.supersedes not in observed:
            raise ValueError(
                f"event {event.event_id} supersedes unobserved event {event.supersedes}"
            )
        missing_parents = [parent for parent in event.parents if parent not in observed]
        if missing_parents:
            raise ValueError(f"event {event.event_id} has missing parents: {missing_parents}")
        if event.kind in {"assign", "fact", "derived"}:
            state[f"{event.subject}.{event.predicate}"] = event.value
        observed.add(event.event_id)
    compiled = tuple(sorted(state.items()))
    if compiled != tuple(sorted(case.current_state)):
        raise ValueError(f"compiled state disagrees with case contract: {compiled!r}")
    return compiled


def retrieve_raw_documents(case: BenchmarkCase, *, limit: int = 2) -> list[str]:
    if limit <= 0:
        raise ValueError("retrieval limit must be positive")
    query_terms = _terms(case.query)
    scored = []
    for index, document in enumerate(raw_event_documents(case)):
        score = len(query_terms & _terms(document))
        scored.append((-score, index, document))
    return [document for _, _, document in sorted(scored)[:limit]]


def retrieve_state_and_path(case: BenchmarkCase) -> list[str]:
    by_id = {event.event_id: event for event in case.events}
    missing = [event_id for event_id in case.causal_path if event_id not in by_id]
    if missing:
        raise ValueError(f"causal path refers to missing events: {missing}")
    state = [f"STATE {key}={value}" for key, value in compile_automata_state(case)]
    path = [
        f"EDGE {event.event_id}: {event.subject} {event.predicate} {event.value}"
        for event in (by_id[event_id] for event_id in case.causal_path)
    ]
    return state + path


def state_accounting(case: BenchmarkCase) -> dict[str, int]:
    live_state = canonical_json(dict(compile_automata_state(case)))
    transition_log = b"\n".join(canonical_json(asdict(event)) for event in case.events)
    return {
        "automata_state_bytes": len(live_state),
        "transition_log_bytes": len(transition_log),
    }


def render_arm(
    case: BenchmarkCase,
    tokenizer: Tokenizer,
    *,
    arm: Arm,
    max_prompt_tokens: int,
    retrieval_limit: int = 2,
) -> dict[str, Any]:
    if arm not in ("A", "B", "C"):
        raise ValueError(f"unknown arm: {arm!r}")
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    query_ids = list(_encode(tokenizer, "\n" + case.query))
    if arm == "A":
        evidence: list[str] = []
    elif arm == "B":
        evidence = retrieve_raw_documents(case, limit=retrieval_limit)
    else:
        evidence = retrieve_state_and_path(case)
    evidence_bytes = "\n".join(evidence).encode("utf-8")
    evidence_ids = list(_encode(tokenizer, "\n".join(evidence) + "\n")) if evidence else []
    fixed = evidence_ids + query_ids
    if len(fixed) >= max_prompt_tokens:
        raise ValueError("evidence and query leave no room for live context")
    live_budget = max_prompt_tokens - len(fixed)
    if live_budget <= case.history_distance:
        live_history = [case.filler_token_id] * live_budget
    else:
        prefix_budget = live_budget - case.history_distance
        live_history = list(case.prefix_token_ids[-prefix_budget:]) + [
            case.filler_token_id
        ] * case.history_distance
        live_history = live_history[-live_budget:]
    prompt_ids = live_history + fixed
    if len(prompt_ids) > max_prompt_tokens:
        raise AssertionError("rendered prompt exceeded its bound")
    accounting = state_accounting(case) if arm == "C" else {
        "automata_state_bytes": 0,
        "transition_log_bytes": 0,
    }
    return {
        "arm": arm,
        "prompt_ids": prompt_ids,
        "answer_ids": list(_encode(tokenizer, case.answer)),
        "live_context_token_count": len(live_history),
        "retrieved_bytes": len(evidence_bytes),
        "retrieved_documents": evidence,
        **accounting,
    }


def benchmark_contract(cases: list[BenchmarkCase]) -> dict[str, Any]:
    case_contracts = [case.contract() for case in cases]
    body = {
        "schema": EVALUATOR_SCHEMA,
        "arms": {
            "A": "bounded local context only",
            "B": "same checkpoint plus deterministic lexical raw-event retrieval",
            "C": "same checkpoint plus benchmark-owned typed state and causal-path retrieval",
        },
        "cases": case_contracts,
        "scoring": {
            "model_accuracy": "teacher-forced exact next-token accuracy over answer tokens",
            "model_nll": "mean negative log likelihood over answer tokens",
            "kv_cache_bytes": "zero only when the model returns no past_key_values",
            "flops": "explicit estimate with formula recorded by evaluator",
        },
    }
    return {**body, "contract_root": canonical_root(body)}
