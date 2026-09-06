import unittest

import torch

from helix_lm.config import HelixConfig
from helix_lm.graph import HelixGraph
from helix_lm.model import HelixLMCore


def topology_config(nodes_per_column, **overrides):
    values = {
        "vocab_size": 32,
        "seq_len": 8,
        "batch_size": 1,
        "d_model": 32,
        "n_columns": len(nodes_per_column),
        "nodes_per_column": nodes_per_column,
        "attention_mode": "linear",
        "n_heads": 4,
        "n_loops": 1,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "use_ssm": False,
        "use_titans_memory": False,
        "seed": 42,
    }
    values.update(overrides)
    return HelixConfig(**values)


class NodesPerColumnTopologyTest(unittest.TestCase):
    def test_nodes_per_column_controls_compute_node_count(self):
        graph = HelixGraph(topology_config((2, 3, 2)))

        compute_counts = [
            sum(node_type != "gate" for node_type, _ in column)
            for column in graph.node_spec
        ]
        node_types = [
            [node_type for node_type, _ in column]
            for column in graph.node_spec
        ]

        self.assertEqual(compute_counts, [2, 3, 2])
        self.assertEqual(
            graph.get_graph_info()["compute_nodes_per_column"],
            [2, 3, 2],
        )
        self.assertEqual(
            node_types,
            [
                ["linear_attn", "swiglu", "gate"],
                ["linear_attn", "swiglu", "linear_attn", "gate"],
                ["linear_attn", "swiglu", "gate"],
            ],
        )

    def test_topology_change_changes_graph_and_parameter_count(self):
        compact = HelixGraph(topology_config((2, 3, 2)))
        expanded = HelixGraph(topology_config((3, 3, 3)))

        self.assertNotEqual(compact.node_spec, expanded.node_spec)
        self.assertEqual(compact.get_graph_info()["n_nodes"], 10)
        self.assertEqual(expanded.get_graph_info()["n_nodes"], 12)
        self.assertNotEqual(
            sum(parameter.numel() for parameter in compact.parameters()),
            sum(parameter.numel() for parameter in expanded.parameters()),
        )

    def test_optional_nodes_consume_slots_before_base_pattern_repeats(self):
        graph = HelixGraph(
            topology_config(
                (5,),
                use_ssm=True,
                use_titans_memory=True,
                titans_always_select=True,
            )
        )

        self.assertEqual(
            [node_type for node_type, _ in graph.node_spec[0]],
            ["linear_attn", "swiglu", "mamba2", "titans", "linear_attn", "gate"],
        )

    def test_232_model_executes_all_constructed_nodes(self):
        model = HelixLMCore(topology_config((2, 3, 2)))
        visited = []
        hooks = [
            node.register_forward_hook(
                lambda _module, _inputs, _output, name=name: visited.append(name)
            )
            for name, node in model.recurrent.graph.nodes.items()
        ]
        try:
            logits = model(torch.randint(0, 32, (1, 8)))
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(logits.shape), (1, 8, 32))
        self.assertEqual(set(visited), set(model.recurrent.graph.nodes))

    def test_config_normalizes_json_list_and_rejects_invalid_counts(self):
        config = topology_config([2, 3])
        self.assertEqual(config.nodes_per_column, (2, 3))

        with self.assertRaisesRegex(ValueError, "positive integers"):
            topology_config((2, 0, 2))

        with self.assertRaisesRegex(ValueError, "at least one value"):
            HelixConfig(n_columns=3, nodes_per_column=())


if __name__ == "__main__":
    unittest.main()
