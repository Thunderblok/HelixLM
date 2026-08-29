#!/usr/bin/env python3
"""Materialize and court a Branch-50 checkpoint before Lighteval.

This script intentionally does not depend on Lighteval.  It proves the model
and tokenizer contract that Lighteval will consume, then emits a custody
receipt.  Run the actual evaluation in a separate, version-pinned environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FIXED_PROMPT = (
    "In a small laboratory, the careful scientist recorded every result "
    "before making a claim."
)
FIXED_CONTINUATION = " The evidence remained reproducible."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_state_root(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
            digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def independent_reload_court(
    *,
    export_dir: Path,
    source_root: Path,
    expected_model_root: str,
) -> dict[str, Any]:
    """Reload the export in a fresh process using the exact registered source.

    Transformers 5.8.1 cannot reliably reconstruct Helix's transitive custom
    module graph from a local ``trust_remote_code`` export: its local dynamic
    module copier copies direct imports, while its hash walk expects transitive
    imports to be present.  Lighteval therefore binds the exact Helix source,
    registers its AutoClasses, and loads weights with remote code disabled.
    """

    code = r'''
import hashlib
import json
import sys
from pathlib import Path

import torch

export_dir = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source_root))

from helix_lm.hf_model import HelixForCausalLM  # noqa: F401
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    export_dir,
    trust_remote_code=False,
    dtype=torch.float32,
    low_cpu_mem_usage=False,
)

digest = hashlib.sha256()
with torch.no_grad():
    for name, tensor in model.state_dict().items():
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())

tie_required = bool(model.config.tie_word_embeddings)
tie_observed = model.lm_head.weight.data_ptr() == model.model.embed.weight.data_ptr()
print(json.dumps({
    "class_module": type(model).__module__,
    "class_name": type(model).__name__,
    "model_root": digest.hexdigest(),
    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    "registered_source_required": True,
    "trust_remote_code": False,
    "tie_word_embeddings": tie_required,
    "tied_weight_alias_observed": tie_observed,
}, sort_keys=True))
'''
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else os.pathsep.join((str(source_root), existing_pythonpath))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(export_dir), str(source_root)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "REFUSED: independent registered-source reload failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("REFUSED: independent reload emitted no result")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "REFUSED: independent reload result was not JSON: "
            f"{lines[-1]!r}"
        ) from exc
    if result.get("model_root") != expected_model_root:
        raise SystemExit(
            "REFUSED: independent reload changed model root: "
            f"{expected_model_root} != {result.get('model_root')}"
        )
    if result.get("tie_word_embeddings") and not result.get(
        "tied_weight_alias_observed"
    ):
        raise SystemExit("REFUSED: independent reload did not restore tied weights")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def refuse_dirty_output(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if not any(path.iterdir()):
        return
    if not overwrite:
        raise SystemExit(f"REFUSED: output directory is not empty: {path}")
    shutil.rmtree(path)


def continuation_loglikelihood(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer(FIXED_PROMPT, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(
        FIXED_CONTINUATION,
        add_special_tokens=False,
    )["input_ids"]
    if not prompt_ids or not continuation_ids:
        raise SystemExit("REFUSED: fixed prompt or continuation tokenized empty")
    combined = torch.tensor(
        [prompt_ids + continuation_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(combined)
    with torch.no_grad():
        logits = model(
            input_ids=combined,
            attention_mask=attention_mask,
        ).logits.float()
    if not torch.isfinite(logits).all():
        raise SystemExit("REFUSED: nonfinite logits in loglikelihood court")
    shift = len(prompt_ids) - 1
    selected_logits = logits[:, shift : shift + len(continuation_ids), :]
    targets = combined[:, len(prompt_ids) :]
    token_logprobs = torch.log_softmax(selected_logits, dim=-1).gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    if token_logprobs.shape != targets.shape:
        raise SystemExit("REFUSED: continuation loglikelihood shape mismatch")
    if not torch.isfinite(token_logprobs).all():
        raise SystemExit("REFUSED: nonfinite continuation loglikelihood")
    total = float(token_logprobs.sum().item())
    return {
        "prompt_token_count": len(prompt_ids),
        "continuation_token_count": len(continuation_ids),
        "total_loglikelihood": total,
        "mean_loglikelihood": total / len(continuation_ids),
        "perplexity": math.exp(-total / len(continuation_ids)),
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("REFUSED: max-new-tokens must be positive")
    for required in (args.checkpoint, args.resolved_config):
        if not required.is_file():
            raise SystemExit(f"REFUSED: required file missing: {required}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA requested but unavailable")

    refuse_dirty_output(args.output_dir, overwrite=args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root))
    from helix_lm.config import HelixConfig
    from helix_lm.hf_model import HelixForCausalLM

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise SystemExit("REFUSED: checkpoint has no model state")

    config = HelixConfig.from_json_file(str(args.resolved_config))
    if config.architectures != ["HelixForCausalLM"]:
        raise SystemExit(
            "REFUSED: architectures must be exactly ['HelixForCausalLM'], "
            f"found {config.architectures!r}"
        )
    model = HelixForCausalLM(config)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=True)
    if missing or unexpected:
        raise SystemExit(
            f"REFUSED: state mismatch missing={missing!r} unexpected={unexpected!r}"
        )
    loaded_root = model_state_root(model)
    checkpoint_root = checkpoint.get("model_root")
    if checkpoint_root is not None and checkpoint_root != loaded_root:
        raise SystemExit(
            "REFUSED: checkpoint model root mismatch: "
            f"declared={checkpoint_root} observed={loaded_root}"
        )

    tokenizer_name = str(config.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id != config.eos_token_id:
        raise SystemExit(
            "REFUSED: tokenizer/config EOS mismatch: "
            f"{tokenizer.eos_token_id} != {config.eos_token_id}"
        )

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    exported_manifest = artifact_manifest(args.output_dir)
    if not exported_manifest:
        raise SystemExit("REFUSED: save_pretrained emitted no files")

    del model
    device = torch.device(args.device)
    reloaded = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).to(device)
    reloaded.eval()
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        args.output_dir,
        trust_remote_code=False,
    )
    reloaded_root = model_state_root(reloaded)
    if reloaded_root != loaded_root:
        raise SystemExit(
            "REFUSED: save/reload changed model state root: "
            f"{loaded_root} != {reloaded_root}"
        )
    tie_required = bool(reloaded.config.tie_word_embeddings)
    tie_observed = (
        reloaded.lm_head.weight.data_ptr()
        == reloaded.model.embed.weight.data_ptr()
    )
    if tie_required and not tie_observed:
        raise SystemExit("REFUSED: save/reload did not restore tied weights")
    independent_reload = independent_reload_court(
        export_dir=args.output_dir,
        source_root=source_root,
        expected_model_root=loaded_root,
    )

    encoded = reloaded_tokenizer(
        FIXED_PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    if input_ids.shape[1] > config.seq_len:
        raise SystemExit("REFUSED: fixed prompt exceeds configured context")
    with torch.no_grad():
        generated = reloaded.generate(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=reloaded_tokenizer.eos_token_id,
            eos_token_id=reloaded_tokenizer.eos_token_id,
        )
    if generated.shape[1] <= input_ids.shape[1]:
        raise SystemExit("REFUSED: fixed-prompt generation emitted no continuation")
    generation_text = reloaded_tokenizer.decode(
        generated[0, input_ids.shape[1] :],
        skip_special_tokens=False,
    )
    likelihood = continuation_loglikelihood(
        reloaded,
        reloaded_tokenizer,
        device=device,
    )

    receipt: dict[str, Any] = {
        "schema": "helix.branch50.lighteval_checkpoint_preflight.v0",
        "status": "PASS",
        "publication_effect": "none",
        "lighteval_executed": False,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "declared_model_root": checkpoint_root,
            "observed_model_root": loaded_root,
            "step": checkpoint.get("step"),
            "val_loss": checkpoint.get("val_loss"),
        },
        "resolved_config": {
            "path": str(args.resolved_config.resolve()),
            "sha256": sha256_file(args.resolved_config),
            "architectures": config.architectures,
            "model_type": config.model_type,
            "seq_len": config.seq_len,
            "vocab_size": config.vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        },
        "tokenizer": {
            "name": tokenizer_name,
            "vocab_size": len(reloaded_tokenizer),
            "pad_token_id": reloaded_tokenizer.pad_token_id,
            "eos_token_id": reloaded_tokenizer.eos_token_id,
            "bos_token_id": reloaded_tokenizer.bos_token_id,
        },
        "export": {
            "path": str(args.output_dir.resolve()),
            "files": exported_manifest,
            "manifest_root": canonical_root(exported_manifest),
            "reloaded_model_root": reloaded_root,
            "tie_word_embeddings": tie_required,
            "tied_weight_alias_observed": tie_observed,
        },
        "independent_reload_court": independent_reload,
        "generation_court": {
            "prompt": FIXED_PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "generated_token_count": int(generated.shape[1] - input_ids.shape[1]),
            "generated_text": generation_text,
        },
        "loglikelihood_court": {
            "prompt": FIXED_PROMPT,
            "continuation": FIXED_CONTINUATION,
            **likelihood,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
        },
    }
    receipt["receipt_root"] = canonical_root(receipt)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
