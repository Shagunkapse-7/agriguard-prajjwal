"""Evaluate trained crop model on isolated test set."""

import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from agriguard.datasets.registry import DatasetRegistry
from agriguard.datasets.torch_dataset import AgriGuardDataset
from agriguard.evaluation.evaluator import ModelEvaluator
from agriguard.preprocessing.transforms import get_eval_transforms
from agriguard.training.model_factory import create_model
from agriguard.utils.logger import get_logger

logger = get_logger("evaluate_model")


def evaluate_crop_model(
    crop: str,
    arch: str = "mobilenet_v3_small",
    exported_dir: str = "models/exported",
    manifests_dir: str = "data/manifests",
    reports_dir: str = "data/reports",
    device: str = "cpu",
):
    model_path = Path(exported_dir) / f"{crop}_{arch}.pt"
    if not model_path.exists():
        logger.error(f"Model bundle not found at {model_path}. Train model first.")
        sys.exit(1)

    bundle = torch.load(model_path, map_location=device, weights_only=False)
    classes = bundle["classes"]
    state_dict = bundle["model_state_dict"]

    model = create_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(state_dict)

    registry = DatasetRegistry(manifests_dir)
    test_records = registry.get_records_by_split(crop, "test")

    if not test_records:
        logger.error(f"No test split records found for {crop} in manifest.")
        sys.exit(1)

    logger.info(f"Loaded {len(test_records)} test samples for crop '{crop}'. Running evaluation...")
    test_ds = AgriGuardDataset(
        records=test_records,
        classes=classes,
        transform=get_eval_transforms((224, 224)),
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    evaluator = ModelEvaluator(model, classes=classes, device=device)
    eval_results = evaluator.evaluate(test_loader)

    # Output reports
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    json_file = rep_dir / f"evaluation_{crop}_{arch}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    md_content = evaluator.generate_markdown_report(eval_results, crop, arch)
    md_file = rep_dir / f"evaluation_{crop}_{arch}.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Scientific evaluation reports successfully saved to {json_file} and {md_file}")
    logger.info(f"Summary Accuracy: {eval_results['metrics']['accuracy']*100:.2f}% | Macro F1: {eval_results['metrics']['macro_f1']:.4f}")

    # Format and print the evaluation summary table
    m = eval_results["metrics"]
    headers = ["Category", "Crop", "Architecture", "Classes", "Test N", "Accuracy", "Macro F1", "Latency", "Status"]
    row = [
        "grain_crop_leaf",
        crop,
        arch,
        f"{len(classes)} cls",
        str(eval_results["num_test_samples"]),
        f"{m['accuracy']*100:.1f}%",
        f"{m['macro_f1']:.4f}",
        f"{m.get('avg_latency_ms', 0.0):.2f} ms",
        "Operational",
    ]
    col_widths = [max(len(h), len(r)) for h, r in zip(headers, row)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    data_row = "| " + " | ".join(f"{val:<{col_widths[i]}}" for i, val in enumerate(row)) + " |"

    print("\n" + "=" * len(sep))
    print(f"AGRIGUARD AI -- EVALUATION RESULT: {crop.upper()} ({arch})".center(len(sep)))
    print("=" * len(sep))
    print(sep)
    print(header_row)
    print(sep)
    print(data_row)
    print(sep)

    # Per-class breakdown table
    pcm = eval_results.get("per_class_metrics", {})
    if pcm:
        pc_headers = ["Class / Condition", "Precision", "Recall", "F1-Score", "Test Samples"]
        pc_rows = [
            [
                cls_name.replace("_", " ").title(),
                f"{stats.get('precision', 0.0)*100:.1f}%",
                f"{stats.get('recall', 0.0)*100:.1f}%",
                f"{stats.get('f1_score', 0.0):.4f}",
                str(int(stats.get('support', 0))),
            ]
            for cls_name, stats in pcm.items()
        ]
        pc_widths = [len(h) for h in pc_headers]
        for r in pc_rows:
            for i, val in enumerate(r):
                pc_widths[i] = max(pc_widths[i], len(val))
        pc_sep = "+" + "+".join("-" * (w + 2) for w in pc_widths) + "+"
        pc_header_row = "| " + " | ".join(f"{h:<{pc_widths[i]}}" for i, h in enumerate(pc_headers)) + " |"

        print("\n" + pc_sep)
        print("| " + "PER-CLASS BREAKDOWN".center(len(pc_sep) - 4) + " |")
        print(pc_sep)
        print(pc_header_row)
        print(pc_sep)
        for r in pc_rows:
            print("| " + " | ".join(f"{val:<{pc_widths[i]}}" for i, val in enumerate(r)) + " |")
        print(pc_sep)

    print(f"\nDetailed report saved to: {md_file}\n")
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate crop disease model on test set")
    parser.add_argument("--crop", default="rice_leaf", help="Target crop")
    parser.add_argument("--arch", default="mobilenet_v3_small", help="Architecture name")
    parser.add_argument("--exported-dir", default="models/exported", help="Exported models directory")
    parser.add_argument("--reports-dir", default="data/reports", help="Evaluation reports directory")
    parser.add_argument("--device", default="cpu", help="Compute device")
    args = parser.parse_args()

    evaluate_crop_model(
        crop=args.crop,
        arch=args.arch,
        exported_dir=args.exported_dir,
        reports_dir=args.reports_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
