from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


LAUNCHER_PATH = Path(__file__).with_name("113M_param_train.py")
SPEC = importlib.util.spec_from_file_location("production_pretrain_launcher", LAUNCHER_PATH)
LAUNCHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LAUNCHER
SPEC.loader.exec_module(LAUNCHER)


class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 1

    def __len__(self):
        return 50_257


class ProductionPretrainLauncherTest(unittest.TestCase):
    def test_default_contract_is_the_documented_single_gpu_profile(self):
        settings = LAUNCHER.resolve_settings({})

        self.assertEqual(settings.profile_name, "single-gpu-16gb")
        self.assertEqual(settings.dataset, LAUNCHER.SUTRA_DATASET)
        self.assertEqual(settings.dataset_revision, LAUNCHER.SUTRA_REVISION)
        self.assertEqual(settings.compile_store_dir, Path("pretrain_store"))
        self.assertEqual(settings.d_model, 768)
        self.assertEqual(settings.n_heads, 12)
        self.assertEqual(settings.n_columns, 3)
        self.assertEqual(settings.nodes_per_column, (3, 3, 3))
        self.assertEqual(settings.n_loops, 4)
        self.assertEqual(settings.seq_len, 1024)
        self.assertEqual(settings.ffn_expansion, 3.0)
        self.assertEqual(settings.lateral_p, 0.8)
        self.assertEqual(settings.vertical_p, 0.9)
        self.assertEqual(settings.vertical_depth, 2)
        self.assertEqual(settings.batch_size, 3)
        self.assertEqual(settings.grad_accum, 28)
        self.assertEqual(settings.effective_batch, 84)
        self.assertEqual(settings.learning_rate, 2e-4)

        cfg = LAUNCHER.build_config(settings, TokenizerStub())
        self.assertEqual(cfg.d_model, settings.d_model)
        self.assertEqual(cfg.n_heads, settings.n_heads)
        self.assertEqual(cfg.n_columns, settings.n_columns)
        self.assertEqual(cfg.nodes_per_column, settings.nodes_per_column)
        self.assertEqual(cfg.n_loops, settings.n_loops)
        self.assertEqual(cfg.seq_len, settings.seq_len)
        self.assertEqual(cfg.ffn_expansion, settings.ffn_expansion)
        self.assertEqual(cfg.lateral_p, settings.lateral_p)
        self.assertEqual(cfg.vertical_p, settings.vertical_p)
        self.assertEqual(cfg.vertical_depth, settings.vertical_depth)

    def test_all_public_model_and_training_knobs_resolve_from_environment(self):
        settings = LAUNCHER.resolve_settings(
            {
                "HELIX_D_MODEL": "1024",
                "HELIX_N_HEADS": "16",
                "HELIX_N_COLUMNS": "4",
                "HELIX_NODES_PER_COLUMN": "2,3,3,2",
                "HELIX_N_LOOPS": "5",
                "HELIX_SEQUENCE_LENGTH": "2048",
                "HELIX_DROPOUT": "0.2",
                "HELIX_ATTENTION_DROPOUT": "0.1",
                "HELIX_FFN_EXPANSION": "3.5",
                "HELIX_WEIGHT_DECAY": "0.1",
                "HELIX_GRAD_CLIP": "0.5",
                "HELIX_GRAD_BUFFER_RATIO": "0.25",
                "HELIX_USE_CCA": "1",
                "HELIX_USE_SSM": "true",
                "HELIX_USE_TITANS_MEMORY": "yes",
                "HELIX_DTYPE": "float32",
                "HELIX_AMP_DTYPE": "bfloat16",
                "HELIX_LATERAL_P": "0.7",
                "HELIX_VERTICAL_P": "0.85",
                "HELIX_VERTICAL_DEPTH": "3",
                "HELIX_ATTENTION_MODE": "multi_scale_windowed",
                "HELIX_LOCAL_WINDOW": "128",
                "HELIX_COARSE_WINDOW": "256",
                "HELIX_COMPRESSED_WINDOWS": "16",
                "HELIX_COMPRESSED_VIEWS": "4",
                "HELIX_CONSENSUS_TYPE": "cosine",
                "HELIX_CORRECTOR_TYPE": "ffn",
                "HELIX_TIE_WORD_EMBEDDINGS": "false",
                "HELIX_STRICT_NAN_CHECK": "true",
                "HELIX_BATCH_SIZE": "4",
                "HELIX_GRAD_ACCUM": "8",
                "HELIX_EPOCHS": "2",
                "HELIX_LEARNING_RATE": "0.0003",
                "HELIX_WARMUP_MICROBATCHES": "200",
                "HELIX_MAX_OPTIMIZER_STEPS": "900",
                "HELIX_VALIDATION_SAMPLES": "128",
                "HELIX_VALIDATION_BATCHES": "4",
                "HELIX_CHECKPOINT_EVERY": "100",
                "HELIX_CHECKPOINT_SLOTS": "3",
                "HELIX_EVAL_EVERY": "50",
                "HELIX_NUM_WORKERS": "2",
                "HELIX_SEED": "7",
            }
        )

        self.assertEqual(settings.d_model, 1024)
        self.assertEqual(settings.n_heads, 16)
        self.assertEqual(settings.n_columns, 4)
        self.assertEqual(settings.nodes_per_column, (2, 3, 3, 2))
        self.assertEqual(settings.n_loops, 5)
        self.assertEqual(settings.seq_len, 2048)
        self.assertEqual(settings.ffn_expansion, 3.5)
        self.assertTrue(settings.use_cca)
        self.assertTrue(settings.use_ssm)
        self.assertTrue(settings.use_titans_memory)
        self.assertFalse(settings.tie_word_embeddings)
        self.assertEqual(settings.vertical_depth, 3)
        self.assertEqual(settings.batch_size, 4)
        self.assertEqual(settings.grad_accum, 8)
        self.assertEqual(settings.effective_batch, 32)
        self.assertEqual(settings.checkpoint_slots, 3)
        self.assertEqual(settings.seed, 7)

        cfg = LAUNCHER.build_config(settings, TokenizerStub())
        self.assertEqual(cfg.nodes_per_column, (2, 3, 3, 2))
        self.assertEqual(cfg.seq_len, 2048)
        self.assertEqual(cfg.local_window, 128)
        self.assertEqual(cfg.coarse_window, 256)
        self.assertEqual(cfg.compressed_windows, 16)
        self.assertEqual(cfg.compressed_views, 4)

    def test_invalid_topology_is_refused_before_model_construction(self):
        invalid_environments = (
            {"HELIX_N_COLUMNS": "3", "HELIX_NODES_PER_COLUMN": "3,3"},
            {"HELIX_NODES_PER_COLUMN": "3,zero,3"},
            {"HELIX_N_HEADS": "0"},
            {"HELIX_D_MODEL": "769", "HELIX_N_HEADS": "12"},
            {"HELIX_LATERAL_P": "1.1"},
            {"HELIX_VERTICAL_P": "-0.1"},
            {"HELIX_DROPOUT": "1.1"},
            {"HELIX_ATTENTION_DROPOUT": "-0.1"},
            {"HELIX_MAX_OPTIMIZER_STEPS": "-1"},
            {"HELIX_REQUIRE_MLFLOW": "1", "HELIX_MLFLOW_URI": ""},
        )

        for environ in invalid_environments:
            with self.subTest(environ=environ), self.assertRaises(ValueError):
                LAUNCHER.resolve_settings(environ)

    def test_instantiated_graph_must_match_the_declared_node_counts(self):
        settings = LAUNCHER.resolve_settings({})
        graph_info = {
            "configured_nodes_per_column": [3, 3, 3],
            "compute_nodes_per_column": [3, 3, 3],
        }

        self.assertEqual(
            LAUNCHER.admit_graph_topology(settings, graph_info),
            (3, 3, 3),
        )

        for observed in ([2, 2, 2], [3, 3], [3, 3, 4]):
            with self.subTest(observed=observed), self.assertRaisesRegex(
                RuntimeError, "instantiated graph"
            ):
                LAUNCHER.admit_graph_topology(
                    settings,
                    {
                        "configured_nodes_per_column": [3, 3, 3],
                        "compute_nodes_per_column": observed,
                    },
                )

        with self.assertRaisesRegex(RuntimeError, "instantiated graph"):
            LAUNCHER.admit_graph_topology(
                settings,
                {
                    "configured_nodes_per_column": [2, 3, 2],
                    "compute_nodes_per_column": [3, 3, 3],
                },
            )

    def test_existing_store_disables_the_default_compile_target(self):
        settings = LAUNCHER.resolve_settings(
            {"HELIX_PRETRAIN_STORE_DIR": "/data/sutra-store"}
        )

        self.assertEqual(settings.train_store_dir, Path("/data/sutra-store"))
        self.assertIsNone(settings.compile_store_dir)

    def test_explicit_dual_store_modes_are_refused(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            LAUNCHER.resolve_settings(
                {
                    "HELIX_PRETRAIN_STORE_DIR": "/data/existing",
                    "HELIX_PRETRAIN_COMPILE_DIR": "/data/new",
                }
            )

    def test_hub_publication_requires_an_explicit_account(self):
        with self.assertRaisesRegex(ValueError, "requires HF_USER"):
            LAUNCHER.resolve_settings({"HELIX_PUSH_TO_HUB": "1"})

        with self.assertRaisesRegex(ValueError, "requires HF_TOKEN"):
            LAUNCHER.resolve_settings(
                {"HELIX_PUSH_TO_HUB": "1", "HF_USER": "example"}
            )

    def test_public_launcher_contains_no_internal_campaign_identifiers(self):
        source = LAUNCHER_PATH.read_text().lower()

        for forbidden in (
            "branch60",
            "branch62",
            "rtx5080-relative",
            "reference_run_id",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_hugging_face_name_uses_the_resolved_configuration(self):
        settings = LAUNCHER.resolve_settings(
            {
                "HELIX_D_MODEL": "1024",
                "HELIX_N_HEADS": "16",
                "HELIX_N_COLUMNS": "4",
                "HELIX_NODES_PER_COLUMN": "2,3,3,2",
                "HELIX_N_LOOPS": "5",
                "HELIX_FFN_EXPANSION": "3.5",
                "HELIX_SEQUENCE_LENGTH": "2048",
                "HELIX_EPOCHS": "3",
            }
        )
        name = LAUNCHER.model_name(settings, "260904-2300")

        self.assertLessEqual(len(name), 96)
        self.assertIn("d1024-c4-n2332-l5-f35-s2048-e3", name)


if __name__ == "__main__":
    unittest.main()
