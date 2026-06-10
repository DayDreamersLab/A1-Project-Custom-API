from argparse import ArgumentParser
from pathlib import Path
import json
import time

from pytorch_route_ranker.app.config import RANKER_ROOT, Settings
from pytorch_route_ranker.app.schemas import RankRequest
from pytorch_route_ranker.app.service import RoutingService


def parse_args():
    parser = ArgumentParser(description="Evaluate the trained AMIDS route ranker.")
    parser.add_argument(
        "--data",
        type=Path,
        default=RANKER_ROOT / "data" / "expert_training_examples.jsonl",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="Maximum number of incorrect predictions to print.",
    )
    return parser.parse_args()


def load_test_examples(data_path: Path, valid_route_ids: set[str]) -> list[dict]:
    if not data_path.exists():
        raise ValueError(f"Held-out test file does not exist: {data_path}")

    examples: list[dict] = []
    with data_path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{data_path}:{line_number} is not one valid JSON object: {error}"
                ) from error

            if not isinstance(example.get("query"), str) or not example["query"].strip():
                raise ValueError(f"{data_path}:{line_number} has no valid query.")
            if example.get("scope") not in {"single", "multiple"}:
                raise ValueError(
                    f"{data_path}:{line_number} scope must be exactly 'single' or 'multiple'."
                )

            route_ids = example.get("relevantRouteIds")
            if not isinstance(route_ids, list) or not route_ids:
                raise ValueError(
                    f"{data_path}:{line_number} relevantRouteIds must be a non-empty list."
                )

            invalid_ids = set(route_ids) - valid_route_ids
            if invalid_ids:
                raise ValueError(
                    f"{data_path}:{line_number} contains route IDs absent from the active registry: "
                    f"{sorted(invalid_ids)}"
                )
            examples.append(example)

    if not examples:
        raise ValueError(f"Held-out test file contains no examples: {data_path}")
    return examples


def main() -> None:
    args = parse_args()
    service = RoutingService(Settings())
    examples = load_test_examples(args.data, set(service.routes_by_id))

    correct_scope = 0
    correct_top_route = 0
    relevant_routes_found = 0
    relevant_routes_total = 0
    fallback_count = 0
    durations: list[float] = []
    expected_scope_counts = {"single": 0, "multiple": 0}
    predicted_scope_counts = {"single": 0, "multiple": 0}
    mismatches: list[dict] = []

    for example in examples:
        started_at = time.perf_counter()
        result = service.rank(RankRequest(query=example["query"]))
        durations.append((time.perf_counter() - started_at) * 1000)
        expected_ids = set(example["relevantRouteIds"])
        selected_ids = set(result.routeIds or ([result.routeId] if result.routeId else []))
        expected_scope_counts[example["scope"]] += 1
        predicted_scope_counts[result.requestScope] += 1

        correct_scope += int(result.requestScope == example["scope"])
        correct_top_route += int(result.routeId in expected_ids)
        relevant_routes_found += len(selected_ids & expected_ids)
        relevant_routes_total += len(expected_ids)
        fallback_count += int(result.needsFallback)
        if result.requestScope != example["scope"] or result.routeId not in expected_ids:
            mismatches.append(
                {
                    "query": example["query"],
                    "expectedScope": example["scope"],
                    "predictedScope": result.requestScope,
                    "expectedRouteIds": sorted(expected_ids),
                    "predictedRouteIds": sorted(selected_ids),
                    "confidence": result.confidence,
                    "needsFallback": result.needsFallback,
                }
            )

    example_count = max(1, len(examples))
    print(f"examples={len(examples)} model={service.model_version}")
    print(f"scopeAccuracy={correct_scope / example_count:.3f}")
    print(f"topRouteAccuracy={correct_top_route / example_count:.3f}")
    print(f"relevantRouteRecall={relevant_routes_found / max(1, relevant_routes_total):.3f}")
    print(f"fallbackRate={fallback_count / example_count:.3f}")
    print(f"averageLatencyMs={sum(durations) / example_count:.2f}")
    print(
        "scopeCounts="
        f"expected(single={expected_scope_counts['single']},multiple={expected_scope_counts['multiple']}) "
        f"predicted(single={predicted_scope_counts['single']},multiple={predicted_scope_counts['multiple']})"
    )

    for index, mismatch in enumerate(mismatches[: max(0, args.show_mismatches)], start=1):
        print(f"mismatch[{index}]={json.dumps(mismatch, ensure_ascii=True)}")


if __name__ == "__main__":
    main()
