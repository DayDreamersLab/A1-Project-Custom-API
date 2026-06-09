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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = RoutingService(Settings())
    examples = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correct_scope = 0
    correct_top_route = 0
    relevant_routes_found = 0
    relevant_routes_total = 0
    fallback_count = 0
    durations: list[float] = []

    for example in examples:
        started_at = time.perf_counter()
        result = service.rank(RankRequest(query=example["query"]))
        durations.append((time.perf_counter() - started_at) * 1000)
        expected_ids = set(example["relevantRouteIds"])
        selected_ids = set(result.routeIds or ([result.routeId] if result.routeId else []))

        correct_scope += int(result.requestScope == example["scope"])
        correct_top_route += int(result.routeId in expected_ids)
        relevant_routes_found += len(selected_ids & expected_ids)
        relevant_routes_total += len(expected_ids)
        fallback_count += int(result.needsFallback)

    example_count = max(1, len(examples))
    print(f"examples={len(examples)} model={service.model_version}")
    print(f"scopeAccuracy={correct_scope / example_count:.3f}")
    print(f"topRouteAccuracy={correct_top_route / example_count:.3f}")
    print(f"relevantRouteRecall={relevant_routes_found / max(1, relevant_routes_total):.3f}")
    print(f"fallbackRate={fallback_count / example_count:.3f}")
    print(f"averageLatencyMs={sum(durations) / example_count:.2f}")


if __name__ == "__main__":
    main()
