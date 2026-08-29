# Branch 50 Linear-Context Evidence Packet

This is the compact, commit-safe evidence from the 2026-08-28 Branch 50 RTX
5080 session. Read `HELIX_BRANCH50_OPERATOR_HANDBOOK.md` at the repository root
for the complete operating contract.

Included:

- `evidence/linear-context-court.json` — 512/1024/2048 optimizer-step FLOP and
  dynamic-memory measurements.
- `evidence/real-corpus-1024-terminal.json` — terminal for the 100-step,
  1024-token real-corpus smoke.
- `evidence/real-corpus-1024-config.json` — the material resolved Helix config.
- `evidence/quality-promotion-smoke-pair.json` — matched one-step admission
  evidence for the paired 512/1024 quality runner.
- `evidence/context-promotion-seed42.json` — deterministic 100M-target paired
  comparison and the `RETAIN_512_SEED42_QUALITY_GATE` verdict.
- `evidence/context-promotion-run-identities.json` — source, harness, common
  runner, corpus-manifest, and MLflow run identities for the matched pair.
- `evidence/SHA256SUMS` — roots for the committed packet.
- `executed/` — byte-exact copies of the two Python harnesses, launcher, and
  shared-runner receipt used for the admitted court.
- `run_branch50_quality_promotion.py` — paired-block 512-vs-1024 runner for the
  next 100M-target quality gate.
- `launch_branch50_quality_promotion.sh` — sequential seed-42 launcher that
  prevents the two variants from contending for the RTX 5080.
- `build_context_promotion_packet.py` — fail-closed comparison court that
  verifies terminals, configs, checkpoints, metric finiteness, matched run
  geometry, quality, and the declared throughput floor.

Excluded intentionally: checkpoints, mutable MLflow spools, Python caches,
the U16 corpus, credentials, and environment dumps.

The files under `executed/` preserve the original workstation paths because
changing them would change the executed bytes. They are evidence, not a
portable launcher. A successor harness must replace absolute paths with an
explicit environment/manifest contract and receive its own source hash.

The quality-promotion runner is a successor, not part of the historical
evidence. It samples common 1024-token base blocks for both variants. The 512
control splits every block into two windows; the 1024 candidate masks the
target at the 512 boundary. Both therefore consume the same 42 base blocks,
43,008 raw positions, and 42,924 causal targets per optimizer update.

The 100M-target seed-42 pair completed. At the declared step-1600 reference,
the 1024 candidate retained 97.11% of control throughput but finished 0.013640 NLL
worse on the matched held-out targets. The preregistered quality gate therefore
retains 512 and stops the 1024 promotion family before additional seeds.

```text
CONTROL_MLFLOW=https://mlflow.thunderline.net/#/experiments/7/runs/3f960461105f479598b1643ee1d34b8c
CANDIDATE_MLFLOW=https://mlflow.thunderline.net/#/experiments/7/runs/4561589acb4c44d39d67f9e9635e267a
SOURCE_HEAD=90e2470c72a9c61b21eb0381bf3e7756348104db
SOURCE_TREE=4317bcaa6de2c2b6c4a807c6989af035daab6336
MODEL_BASE_HEAD=03d0698dd3365c81695d9ed8d4568d35d6044fbb
MODEL_BASE_TREE=745c042db9860bca4cdfa180543f8a60a769c936
```
