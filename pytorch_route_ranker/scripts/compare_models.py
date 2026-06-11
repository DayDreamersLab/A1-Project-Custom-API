from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json

from pytorch_route_ranker.app.config import RANKER_ROOT, Settings
from pytorch_route_ranker.app.evaluation import evaluate_checkpoint


def parse_args():
    parser = ArgumentParser(
        description="Compare one candidate ranker against the active ranker on the same test set."
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=RANKER_ROOT / "models" / "route_ranker.pt",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=RANKER_ROOT / "data" / "held_out_test.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Settings().registry_path,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--maximum-fallback-rate-increase", type=float, default=0.02)
    parser.add_argument("--maximum-average-latency-ms", type=float, default=100.0)
    parser.add_argument("--maximum-category-top-route-regression", type=float, default=0.0)
    parser.add_argument("--minimum-category-examples", type=int, default=3)
    return parser.parse_args()


def gate(name: str, passed: bool, current, candidate, requirement: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "current": current,
        "candidate": candidate,
        "requirement": requirement,
    }


def main() -> None:
    args = parse_args()
    current_path = args.current.resolve()
    candidate_path = args.candidate.resolve()
    registry_path = args.registry.resolve()
    data_path = args.data.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else candidate_path.parent / "comparison.json"
    )

    if current_path == candidate_path:
        raise ValueError("Current and candidate checkpoints must be different files.")

    current = evaluate_checkpoint(current_path, registry_path, data_path)
    candidate = evaluate_checkpoint(candidate_path, registry_path, data_path)
    current_overall = current["overall"]
    candidate_overall = candidate["overall"]
    current_multiple = current["multipleRoute"]
    candidate_multiple = candidate["multipleRoute"]
    checks = [
        gate(
            "registry-fingerprint-match",
            current["model"]["registryFingerprint"] == candidate["model"]["registryFingerprint"],
            current["model"]["registryFingerprint"],
            candidate["model"]["registryFingerprint"],
            "Candidate and current model must use the same approved route registry.",
        ),
        gate(
            "overall-top-route-accuracy",
            candidate_overall["topRouteAccuracy"] >= current_overall["topRouteAccuracy"],
            current_overall["topRouteAccuracy"],
            candidate_overall["topRouteAccuracy"],
            "Must not decrease.",
        ),
        gate(
            "overall-scope-accuracy",
            candidate_overall["scopeAccuracy"] >= current_overall["scopeAccuracy"],
            current_overall["scopeAccuracy"],
            candidate_overall["scopeAccuracy"],
            "Must not decrease.",
        ),
        gate(
            "multiple-route-recall",
            candidate_multiple["relevantRouteRecall"] >= current_multiple["relevantRouteRecall"],
            current_multiple["relevantRouteRecall"],
            candidate_multiple["relevantRouteRecall"],
            "Must not decrease.",
        ),
        gate(
            "multiple-route-exact-set-accuracy",
            candidate_multiple["exactSetAccuracy"] >= current_multiple["exactSetAccuracy"],
            current_multiple["exactSetAccuracy"],
            candidate_multiple["exactSetAccuracy"],
            "Must not decrease.",
        ),
        gate(
            "fallback-rate",
            candidate_overall["fallbackRate"]
            <= current_overall["fallbackRate"] + args.maximum_fallback_rate_increase,
            current_overall["fallbackRate"],
            candidate_overall["fallbackRate"],
            f"May increase by at most {args.maximum_fallback_rate_increase:.3f}.",
        ),
        gate(
            "average-latency",
            candidate_overall["averageLatencyMs"] <= args.maximum_average_latency_ms,
            current_overall["averageLatencyMs"],
            candidate_overall["averageLatencyMs"],
            f"Candidate must remain at or below {args.maximum_average_latency_ms:.2f} ms.",
        ),
    ]

    if current["critical"]["examples"] > 0:
        checks.extend(
            [
                gate(
                    "critical-top-route-accuracy",
                    candidate["critical"]["topRouteAccuracy"]
                    >= current["critical"]["topRouteAccuracy"],
                    current["critical"]["topRouteAccuracy"],
                    candidate["critical"]["topRouteAccuracy"],
                    "Must not decrease on examples marked critical.",
                ),
                gate(
                    "critical-relevant-route-recall",
                    candidate["critical"]["relevantRouteRecall"]
                    >= current["critical"]["relevantRouteRecall"],
                    current["critical"]["relevantRouteRecall"],
                    candidate["critical"]["relevantRouteRecall"],
                    "Must not decrease on examples marked critical.",
                ),
            ]
        )

    for category, current_metrics in current["categories"].items():
        candidate_metrics = candidate["categories"].get(category)
        if (
            not candidate_metrics
            or current_metrics["examples"] < args.minimum_category_examples
        ):
            continue
        allowed_minimum = (
            current_metrics["topRouteAccuracy"]
            - args.maximum_category_top_route_regression
        )
        checks.append(
            gate(
                f"category-{category}-top-route-accuracy",
                candidate_metrics["topRouteAccuracy"] >= allowed_minimum,
                current_metrics["topRouteAccuracy"],
                candidate_metrics["topRouteAccuracy"],
                "Category regression must remain within "
                f"{args.maximum_category_top_route_regression:.3f}.",
            )
        )

    failed_checks = [check for check in checks if not check["passed"]]
    comparison = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "eligibleForPromotion": not failed_checks,
        "current": current,
        "candidate": candidate,
        "policy": {
            "maximumFallbackRateIncrease": args.maximum_fallback_rate_increase,
            "maximumAverageLatencyMs": args.maximum_average_latency_ms,
            "maximumCategoryTopRouteRegression": args.maximum_category_top_route_regression,
            "minimumCategoryExamples": args.minimum_category_examples,
        },
        "checks": checks,
        "failedChecks": [check["name"] for check in failed_checks],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{json.dumps(comparison, indent=2, ensure_ascii=True)}\n",
        encoding="utf-8",
    )
    print(f"eligibleForPromotion={str(comparison['eligibleForPromotion']).lower()}")
    print(f"passedChecks={len(checks) - len(failed_checks)}")
    print(f"failedChecks={len(failed_checks)}")
    for failed_check in failed_checks:
        print(
            "failedGate="
            f"{failed_check['name']} current={failed_check['current']} "
            f"candidate={failed_check['candidate']} requirement={failed_check['requirement']}"
        )
    print(f"comparisonReport={output_path}")


if __name__ == "__main__":
    main()
