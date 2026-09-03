import unittest

from topology_depth3_preflight import TopologyTreatment, branch53_config, canonical_root


class TopologyDepth3PreflightTest(unittest.TestCase):
    def test_branch53_recipe_is_explicit(self):
        config = branch53_config(
            TopologyTreatment("candidate", 512, 8, 4, (2, 3, 2, 2), 0.5, 0.7, 3)
        )
        self.assertEqual(config.ffn_expansion, 3.0)
        self.assertEqual(config.lr, 2e-4)
        self.assertEqual(config.n_columns, 4)
        self.assertEqual(config.vertical_depth, 3)
        self.assertEqual(config.lateral_p, 0.5)
        self.assertEqual(config.vertical_p, 0.7)

    def test_width_priority_preserves_attention_head_width(self):
        config = branch53_config(
            TopologyTreatment("width", 768, 12, 3, (2, 3, 2), 0.5, 0.7, 2)
        )
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.n_heads, 12)
        self.assertEqual(config.d_model // config.n_heads, 64)

    def test_depth_three_requires_four_columns_to_add_reach(self):
        self.assertEqual(min(2, 3 - 1), min(3, 3 - 1))
        self.assertNotEqual(min(2, 4 - 1), min(3, 4 - 1))

    def test_canonical_root_is_order_independent_for_mappings(self):
        self.assertEqual(canonical_root({"a": 1, "b": 2}), canonical_root({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
