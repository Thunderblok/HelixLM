#!/usr/bin/env python3
"""Preflight a real vertical-depth-3 Helix experiment without using the GPU."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM


REFERENCE_MLFLOW_RUN = "8ad467637b8b407aaa1c5ced49d6384f"
REFERENCE_RESOLVED_CONFIG_SHA256 = (
    "9ae8b80ee1eaabd587491a82c4a42e18d654a3d1c194a41722de7ad25a253ea0"
)


@dataclass(frozen=True)
class TopologyTreatment:
    name: str
    d_model: int
    n_heads: int
    n_columns: int
    nodes_per_column: tuple[int, ...]
    lateral_p: float
    vertical_p: float
    vertical_depth: int


def canonical_root(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def branch53_config(treatment: TopologyTreatment) -> HelixConfig:
    """Reproduce the admitted FFN3/LR2e-4 recipe with explicit topology."""
    return HelixConfig.small_v2(
        vocab_size=50_257,
        d_model=treatment.d_model,
        n_heads=treatment.n_heads,
        n_loops=3,
        seq_len=512,
        batch_size=12,
        n_columns=treatment.n_columns,
        nodes_per_column=treatment.nodes_per_column,
        lateral_p=treatment.lateral_p,
        vertical_p=treatment.vertical_p,
        vertical_depth=treatment.vertical_depth,
        attention_mode="multi_scale_windowed",
        local_window=64,
        coarse_window=128,
        compressed_windows=8,
        compressed_views=8,
        consensus_type="cosine",
        corrector_type="ffn",
        use_titans_memory=False,
        use_ssm=False,
        use_cca=False,
        strict_nan_check=True,
        dtype="float32",
        amp_dtype="bfloat16",
        dropout=0.05,
        attn_dropout=0.05,
        ffn_expansion=3.0,
        lr=2e-4,
        warmup_steps=2_000,
        weight_decay=0.05,
        grad_clip=1.0,
        tokenizer_name="gpt2",
        pad_token_id=50_256,
        eos_token_id=50_256,
        bos_token_id=50_256,
        tie_word_embeddings=True,
        grad_buffer_ratio=0.0,
        architectures=["HelixForCausalLM"],
        seed=42,
    )


def inspect_treatment(treatment: TopologyTreatment) -> dict[str, Any]:
    config = branch53_config(treatment)
    model = HelixForCausalLM(config)
    graph = model.model.recurrent.graph
    edges = []
    predecessor_distances = []
    for target, predecessors in sorted(graph.graph.items()):
        target_column = graph.node_meta[target][0]
        for predecessor in sorted(predecessors):
            predecessor_column = graph.node_meta[predecessor][0]
            edges.append([predecessor, target])
            predecessor_distances.append(target_column - predecessor_column)
    counts = model.count_parameters()
    result = {
        **asdict(treatment),
        "parameter_count_total": int(counts["total"]),
        "parameter_count_trainable": int(counts["trainable"]),
        "effective_hidden_layers": config.num_hidden_layers,
        "node_count": len(graph.nodes),
        "edge_count": len(edges),
        "maximum_observed_predecessor_column_distance": max(predecessor_distances, default=0),
        "maximum_possible_predecessor_column_distance": min(
            treatment.vertical_depth, treatment.n_columns - 1
        ),
        "graph_root": canonical_root(
            {
                "node_meta": {key: list(value) for key, value in sorted(graph.node_meta.items())},
                "edges": edges,
            }
        ),
    }
    del model
    gc.collect()
    return result


def run_preflight() -> dict[str, Any]:
    treatments = [
        TopologyTreatment("reference_3c_d2", 512, 8, 3, (2, 3, 2), 0.5, 0.7, 2),
        TopologyTreatment("vacuous_3c_d3", 512, 8, 3, (2, 3, 2), 0.5, 0.7, 3),
        TopologyTreatment("depth_control_4c_d2", 512, 8, 4, (2, 3, 2, 2), 0.5, 0.7, 2),
        TopologyTreatment("depth_candidate_4c_d3", 512, 8, 4, (2, 3, 2, 2), 0.5, 0.7, 3),
        # David's production priority: widen first. Twelve heads preserve the
        # reference 64-wide attention heads when d_model rises from 512 to 768.
        TopologyTreatment("width_priority_3c_d2", 768, 12, 3, (2, 3, 2), 0.5, 0.7, 2),
        TopologyTreatment("width_plus_depth_4c_d3", 768, 12, 4, (2, 3, 2, 2), 0.5, 0.7, 3),
    ]
    observed = {item.name: inspect_treatment(item) for item in treatments}
    if observed["reference_3c_d2"]["graph_root"] != observed["vacuous_3c_d3"]["graph_root"]:
        raise RuntimeError("three-column depth 3 unexpectedly changed the graph")
    if observed["depth_control_4c_d2"]["graph_root"] == observed["depth_candidate_4c_d3"]["graph_root"]:
        raise RuntimeError("four-column depth treatment failed to change the graph")
    parameter_delta = (
        observed["depth_candidate_4c_d3"]["parameter_count_total"]
        - observed["depth_control_4c_d2"]["parameter_count_total"]
    )
    width_priority_delta = (
        observed["width_priority_3c_d2"]["parameter_count_total"]
        - observed["reference_3c_d2"]["parameter_count_total"]
    )
    width_plus_depth_delta = (
        observed["width_plus_depth_4c_d3"]["parameter_count_total"]
        - observed["width_priority_3c_d2"]["parameter_count_total"]
    )
    return {
        "schema": "helix.topology-depth3-preflight.v0",
        "reference_mlflow_run": REFERENCE_MLFLOW_RUN,
        "reference_resolved_config_sha256": REFERENCE_RESOLVED_CONFIG_SHA256,
        "gpu_used": False,
        "training_started": False,
        "production_effect": "none",
        "treatments": observed,
        "findings": {
            "three_column_depth3": "vacuous_same_graph_as_depth2",
            "depth_ablation": "compare_four_columns_depth2_vs_depth3",
            "depth_pair_parameter_matching": "not_matched_variable_fanin_merge_layers",
            "depth3_parameter_delta": parameter_delta,
            "production_priority": "d_model_768_before_additional_columns_or_depth",
            "width_priority_parameter_delta_from_reference": width_priority_delta,
            "width_plus_depth_parameter_delta_from_width_priority": width_plus_depth_delta,
            "width_preflight_head_dimension": 64,
            "interpretation_constraint": (
                "vertical_depth changes graph fan-in and therefore learned merge width; "
                "report parameter and active-parameter deltas rather than claiming an "
                "identical-parameter ablation"
            ),
            "reference_role": "recipe_and_quality_anchor_not_depth_matched_checkpoint",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight()
    result["terminal_root"] = canonical_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
