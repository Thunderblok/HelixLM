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
- `evidence/SHA256SUMS` — roots for the committed packet.
- `executed/` — byte-exact copies of the two Python harnesses, launcher, and
  shared-runner receipt used for the admitted court.

Excluded intentionally: checkpoints, mutable MLflow spools, Python caches,
the U16 corpus, credentials, and environment dumps.

The files under `executed/` preserve the original workstation paths because
changing them would change the executed bytes. They are evidence, not a
portable launcher. A successor harness must replace absolute paths with an
explicit environment/manifest contract and receive its own source hash.

```text
MLFLOW=https://mlflow.thunderline.net/#/experiments/7/runs/8e1f6c8b33c048cba447e87ee0a1c505
SOURCE_HEAD=03d0698dd3365c81695d9ed8d4568d35d6044fbb
SOURCE_TREE=745c042db9860bca4cdfa180543f8a60a769c936
```
