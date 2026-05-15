"""
WeightWatcher Analysis for HelixLM
====================================

Analyzes trained HelixLM checkpoints using weightwatcher to extract
insights about layer health, spectral properties, and potential
numerical instability indicators.

Install:  pip install weightwatcher

Usage:
    # Analyze a saved checkpoint
    python analyze_weightwatcher.py --checkpoint ./checkpoints_50m_smoke_fixed/final_model

    # Analyze with specific layers highlighted
    python analyze_weightwatcher.py --checkpoint ./checkpoints/final_model --plot

    # Compare two checkpoints (before/after fix)
    python analyze_weightwatcher.py --checkpoint ./checkpoints/baseline \
                                    --compare ./checkpoints/fixed

Outputs:
    - weightwatcher_report.json: Detailed metrics per layer
    - weightwatcher_summary.json: Aggregated statistics
    - Plots (if --plot): alpha distributions, ESD histograms
"""
import argparse
import json
import os
import sys

import torch


def load_model(checkpoint_path):
    """Load a HelixLM checkpoint from directory."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from helix_lm import HelixForCausalLM, HelixConfig

    if os.path.isdir(checkpoint_path):
        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            import json as _json
            with open(config_path) as f:
                config_dict = _json.load(f)
            cfg = HelixConfig(**config_dict)
        else:
            raise FileNotFoundError(f"No config.json in {checkpoint_path}")

        model = HelixForCausalLM(cfg)
        # Load weights
        weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        else:
            # Try safetensors
            try:
                from safetensors.torch import load_file
                st_path = os.path.join(checkpoint_path, "model.safetensors")
                if os.path.exists(st_path):
                    state_dict = load_file(st_path)
                    model.load_state_dict(state_dict, strict=False)
            except ImportError:
                pass
        return model, cfg
    else:
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")


def analyze_with_weightwatcher(model, output_dir, make_plots=False):
    """Run weightwatcher analysis on model weights."""
    try:
        import weightwatcher as ww
    except ImportError:
        print("ERROR: weightwatcher not installed.")
        print("  pip install weightwatcher")
        return None

    watcher = ww.WeightWatcher(model=model)

    print("Running WeightWatcher analysis...")
    print("  (This may take a few minutes for large models)")

    try:
        details = watcher.analyze(layers="all", fix_fingers="clip_xmax")
        summary = watcher.get_summary()
    except Exception as e:
        print(f"WeightWatcher analysis failed: {e}")
        print("  Trying with reduced scope...")
        try:
            details = watcher.analyze(layers="dense", fix_fingers="clip_xmax")
            summary = watcher.get_summary()
        except Exception as e2:
            print(f"Reduced analysis also failed: {e2}")
            return None

    os.makedirs(output_dir, exist_ok=True)

    # Save detailed report
    details_path = os.path.join(output_dir, "weightwatcher_report.json")
    try:
        # Convert DataFrame to dict for JSON serialization
        details_dict = details.to_dict(orient="records") if hasattr(details, "to_dict") else []
        with open(details_path, "w") as f:
            json.dump(details_dict, f, indent=2, default=str)
        print(f"  Detailed report: {details_path}")
    except Exception as e:
        print(f"  Could not save detailed report: {e}")

    # Save summary
    summary_path = os.path.join(output_dir, "weightwatcher_summary.json")
    try:
        # Convert numpy types to Python native types for JSON
        summary_clean = {}
        for k, v in summary.items() if hasattr(summary, "items") else summary:
            try:
                if hasattr(v, "item"):
                    summary_clean[k] = v.item()
                else:
                    summary_clean[k] = float(v) if v is not None else None
            except (TypeError, ValueError):
                summary_clean[k] = str(v)

        # Add interpretation
        interpretation = interpret_weightwatcher_results(summary_clean)
        summary_clean["_interpretation"] = interpretation

        with open(summary_path, "w") as f:
            json.dump(summary_clean, f, indent=2, default=str)
        print(f"  Summary: {summary_path}")
    except Exception as e:
        print(f"  Could not save summary: {e}")

    # Plots
    if make_plots:
        try:
            plot_dir = os.path.join(output_dir, "weightwatcher_plots")
            os.makedirs(plot_dir, exist_ok=True)
            watcher.plot(plot_id=0, savefig=os.path.join(plot_dir, "esd_distribution.png"))
            print(f"  Plots: {plot_dir}/")
        except Exception as e:
            print(f"  Could not generate plots: {e}")

    # Print key metrics
    print("\nKey Metrics:")
    for k, v in summary_clean.items():
        if not k.startswith("_") and v is not None:
            print(f"  {k}: {v}")

    return summary_clean


def interpret_weightwatcher_results(summary):
    """Interpret weightwatcher metrics for training stability."""
    interp = {
        "overall_health": "unknown",
        "numerical_stability_risk": "unknown",
        "recommendations": [],
    }

    alpha = summary.get("alpha", None)
    if alpha is not None:
        if alpha < 2:
            interp["overall_health"] = "good"
            interp["recommendations"].append("Alpha < 2: Well-trained, power-law ESD")
        elif alpha < 4:
            interp["overall_health"] = "fair"
            interp["recommendations"].append("Alpha 2-4: May need more training data or regularization")
        else:
            interp["overall_health"] = "poor"
            interp["numerical_stability_risk"] = "high"
            interp["recommendations"].append("Alpha > 4: Possible rank collapse or poor training - check for NaN/Inf")

    # Check for warning signs
    num_bad = summary.get("num_bad", 0)
    if num_bad and num_bad > 0:
        interp["numerical_stability_risk"] = "elevated"
        interp["recommendations"].append(f"{num_bad} layers flagged as 'bad' - investigate layer norms and activations")

    log_norm = summary.get("log_norm", None)
    if log_norm is not None:
        if log_norm < -1:
            interp["recommendations"].append("Low log_norm: Weights may be too small, consider higher LR")
        elif log_norm > 2:
            interp["recommendations"].append("High log_norm: Weights may be exploding, consider gradient clipping")

    if not interp["recommendations"]:
        interp["recommendations"].append("No specific issues detected")

    return interp


def compare_checkpoints(checkpoint_paths, labels=None):
    """Compare weightwatcher metrics across multiple checkpoints."""
    if labels is None:
        labels = [f"ckpt_{i}" for i in range(len(checkpoint_paths))]

    results = {}
    for path, label in zip(checkpoint_paths, labels):
        print(f"\n{'='*60}")
        print(f"Analyzing: {label} ({path})")
        print(f"{'='*60}")
        try:
            model, _ = load_model(path)
            summary = analyze_with_weightwatcher(model, os.path.dirname(path) or ".")
            results[label] = summary
        except Exception as e:
            print(f"  FAILED: {e}")
            results[label] = {"error": str(e)}

    # Comparison report
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    metrics = ["alpha", "log_norm", "num_bad", "num_layers"]
    for metric in metrics:
        print(f"\n{metric}:")
        for label, result in results.items():
            val = result.get(metric, "N/A") if result else "N/A"
            print(f"  {label}: {val}")

    return results


def main():
    parser = argparse.ArgumentParser(description="WeightWatcher analysis for HelixLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--output", type=str, default="./weightwatcher_output",
                        help="Output directory for reports")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    parser.add_argument("--compare", type=str, default=None,
                        help="Second checkpoint for comparison")
    parser.add_argument("--compare-label", type=str, default=None,
                        help="Label for comparison checkpoint")
    args = parser.parse_args()

    if args.compare:
        labels = [args.checkpoint, args.compare]
        if args.compare_label:
            labels = ["baseline", args.compare_label]
        compare_checkpoints([args.checkpoint, args.compare], labels)
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        model, cfg = load_model(args.checkpoint)
        params = model.count_parameters()["total"]
        print(f"Parameters: {params:,}")

        output_dir = args.output
        analyze_with_weightwatcher(model, output_dir, make_plots=args.plot)
        print(f"\nDone. Reports in: {output_dir}/")


if __name__ == "__main__":
    main()
