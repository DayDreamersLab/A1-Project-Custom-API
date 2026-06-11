from argparse import ArgumentParser
from pathlib import Path
import json

from pytorch_route_ranker.app.config import RANKER_ROOT, Settings
from pytorch_route_ranker.app.evaluation import evaluate_checkpoint


def parse_args():
    parser = ArgumentParser(description="Evaluate the trained AMIDS route ranker.")
    parser.add_argument(
        "--data",
        type=Path,
        default=RANKER_ROOT / "data" / "expert_training_examples.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Settings().checkpoint_path,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Settings().registry_path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the complete machine-readable JSON report.",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="Maximum number of incorrect predictions to print.",
    )
    return parser.parse_args()


def print_metric_block(prefix: str, metrics: dict) -> None:
    for name, value in metrics.items():
        print(f"{prefix}{name}={value}")


def main() -> None:
    args = parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.registry, args.data)

    print(f"examples={report['overall']['examples']} model={report['model']['version']}")
    for metric in [
        "scopeAccuracy",
        "topRouteAccuracy",
        "relevantRouteRecall",
        "fallbackRate",
        "averageLatencyMs",
    ]:
        print(f"{metric}={report['overall'][metric]}")
    print_metric_block("multipleRoute.", report["multipleRoute"])
    print_metric_block("critical.", report["critical"])
    for category, metrics in report["categories"].items():
        print_metric_block(f"category.{category}.", metrics)

    for index, mismatch in enumerate(
        report["mismatches"][: max(0, args.show_mismatches)],
        start=1,
    ):
        print(f"mismatch[{index}]={json.dumps(mismatch, ensure_ascii=True)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"{json.dumps(report, indent=2, ensure_ascii=True)}\n",
            encoding="utf-8",
        )
        print(f"evaluationReport={args.output.resolve()}")


if __name__ == "__main__":
    main()
