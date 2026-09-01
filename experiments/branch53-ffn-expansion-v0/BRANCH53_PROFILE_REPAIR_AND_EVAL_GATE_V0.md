# Branch53 Profile Repair and Evaluation Gate V0

Status: `G0_PASS_G1_EXPORT_PASS_DOWNSTREAM_UNAVAILABLE_G2_HELD`

This packet closes the Branch53 profile/config identity defect and records the
checkpoint-evaluation boundary before any further GPU training.

## Source repair

```text
repair_parent_head=cf74c3cbfc3b872ffef089d4697c1b3c1cbfaf14
repair_head=bfe672b5939fd5c2b0f0b7871ef7004eb126b5cd
repair_tree=541f0d1a727c9a39fac5bccdaa542f366dc62863
production_effect=none
```

The runner now derives the Branch53 profile from resolved knobs, optionally
refuses a mismatched `--expected-profile`, binds the FFN 3.0 profile to the
observed 54,771,988 total/trainable parameter count, and records the resolved
invocation in the run contract. The queued FFN launcher explicitly requires
`ffn_expansion_3p0_v0`.

Hostile courts prove:

```text
FFN profile + omitted FFN flag=RED
FFN 3.0 profile + 53,592,340 parameters=RED
FFN 3.0 profile + 54,771,988 parameters=PASS
LR-only resolved knobs=learning_rate_ablation_v0
```

Verification:

```text
python=helix-branch49-5080-scaling-v0/.venv/bin/python
py_compile=PASS
BRANCH53_FFN_EXPANSION_COURTS=PASS
bash_n=PASS
shellcheck=PASS
git_diff_check=PASS
```

## Exact checkpoint export preflights

Both selected checkpoints passed the existing CPU checkpoint-to-Transformers
export, exact model-root reload, tied-weight, log-likelihood, and generation
courts.

### Branch50 control, LR 1.5e-4

```text
checkpoint_sha256=ca7705d7266f3590e9bbc7dd253476eb7888be477cbd97d538ee647927cdada4
checkpoint_step=34700
checkpoint_validation_loss=3.858164131641388
receipt_file_sha256=10623d54b5b642047a6e34c9fcea564029c6c98ca91d872ad943f4d30bf59875
receipt_root=ca96e568991b4b31ff7806fc0091a0ecbffc1031e7a690ed4b98275a135247fe
export_manifest_root=d92c321e88fe134f51a561ef1a3766b3f3e69cb70fc129427dadaf7303e5b5b1
fixed_prompt_perplexity=96.67361982145407
```

### Branch53 candidate, LR 2e-4

```text
checkpoint_sha256=588ed9606ff2b185c00f2fb8238d74c0ec5877ffbdce055674489e397b2a82bf
checkpoint_step=34971
checkpoint_validation_loss=3.8247766494750977
receipt_file_sha256=4ccb865087179c12a6a474b1173753e19c7d898c2d5e06fcdcea7e89d8917950
receipt_root=87f1606c1dfbeee47f86e861c0cc1a77fccaff3b534a4f52ba7d0a5be166a945
export_manifest_root=e4ad037b750bc9205428c2f91bad4e30071332ff10e513c9799a4ad33cd9bdd4
fixed_prompt_perplexity=92.19542624648005
```

The fixed-prompt result favors the LR 2e-4 checkpoint, but it is a six-token
continuation smoke. It is not a downstream capability evaluation and cannot
authorize model or architecture promotion.

## Downstream evaluation boundary

```text
repository_lighteval_task_contract=absent
installed_lighteval_before_attempt=false
isolated_candidate_version=0.13.0
full_dependency_resolution=aborted_before_torch_or_cuda_stack_replacement
lighteval_cli=unavailable
missing_runtime_modules=colorlog,inspect_ai,accelerate
lighteval_executed=false
benchmark_score=not_established
```

The isolated install attempted to select a replacement Torch/CUDA stack and
backtracked across many Inspect-AI releases. That is not a frozen evaluator
environment, so the install was stopped. Installing the Lighteval wheel without
dependencies proved the package identity but not an executable evaluator.

## Decision

```text
G0_PROFILE_IDENTITY=PASS
G1_CHECKPOINT_EXPORT_PREFLIGHT=PASS_BOTH
G1_DOWNSTREAM_EVALUATION=UNAVAILABLE
G2_MATCHED_FFN_PREFLIGHT=HELD
NEW_1P5B_RUN=FORBIDDEN
GPU_TRAINING_STARTED=false
```

The next lawful action is to freeze one evaluator version, task list, dataset
revisions, model adapter, scoring semantics, and dependency lock. Run the same
full task contract against both exported checkpoints. Only a completed paired
evaluation may reopen the matched 2.5-versus-3.0 FFN preflight.
