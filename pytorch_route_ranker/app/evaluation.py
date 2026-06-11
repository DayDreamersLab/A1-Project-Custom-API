from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import re
import time

import torch

from .config import Settings
from .schemas import RankRequest
from .service import RoutingService


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_category(value: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", str(value).lower()))


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
            if (
                not isinstance(route_ids, list)
                or not route_ids
                or any(not isinstance(route_id, str) for route_id in route_ids)
            ):
                raise ValueError(
                    f"{data_path}:{line_number} relevantRouteIds must be a non-empty list of strings."
                )
            invalid_ids = set(route_ids) - valid_route_ids
            if invalid_ids:
                raise ValueError(
                    f"{data_path}:{line_number} contains route IDs absent from the active registry: "
                    f"{sorted(invalid_ids)}"
                )

            categories = example.get("categories", [])
            if categories is not None and (
                not isinstance(categories, list)
                or any(not isinstance(category, str) for category in categories)
            ):
                raise ValueError(
                    f"{data_path}:{line_number} categories must be a list of strings."
                )
            if "critical" in example and not isinstance(example["critical"], bool):
                raise ValueError(f"{data_path}:{line_number} critical must be true or false.")
            examples.append(
                {
                    **example,
                    "categories": [
                        normalized
                        for category in (categories or [])
                        if (normalized := normalize_category(category))
                    ],
                    "critical": example.get("critical", False),
                }
            )

    if not examples:
        raise ValueError(f"Held-out test file contains no examples: {data_path}")
    return examples


def checkpoint_settings(checkpoint_path: Path, registry_path: Path) -> Settings:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return replace(
        Settings(),
        checkpoint_path=checkpoint_path,
        registry_path=registry_path,
        feature_dimension=int(checkpoint["feature_dimension"]),
        hidden_dimension=int(checkpoint["hidden_dimension"]),
    )


def inferred_categories(example: dict, routes_by_id: dict[str, dict]) -> list[str]:
    if example["categories"]:
        return sorted(set(example["categories"]))

    categories = []
    for route_id in example["relevantRouteIds"]:
        route = routes_by_id[route_id]
        first_keyword = next(
            (
                normalize_category(keyword)
                for keyword in route.get("keywords", [])
                if normalize_category(keyword)
            ),
            "",
        )
        if first_keyword:
            categories.append(first_keyword)
    return sorted(set(categories)) or ["uncategorized"]


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def rounded(value: float) -> float:
    return round(float(value), 6)


def aggregate_predictions(records: list[dict]) -> dict:
    if not records:
        return {
            "examples": 0,
            "scopeAccuracy": 0.0,
            "topRouteAccuracy": 0.0,
            "exactSetAccuracy": 0.0,
            "routePrecision": 0.0,
            "relevantRouteRecall": 0.0,
            "routeF1": 0.0,
            "routeCountMeanAbsoluteError": 0.0,
            "fallbackRate": 0.0,
            "averageLatencyMs": 0.0,
        }

    count = len(records)
    correct_scope = sum(record["predictedScope"] == record["expectedScope"] for record in records)
    correct_top = sum(record["predictedTopRouteId"] in record["expectedRouteIds"] for record in records)
    exact_sets = sum(record["predictedRouteIds"] == record["expectedRouteIds"] for record in records)
    relevant_found = sum(len(record["predictedRouteIds"] & record["expectedRouteIds"]) for record in records)
    predicted_total = sum(len(record["predictedRouteIds"]) for record in records)
    relevant_total = sum(len(record["expectedRouteIds"]) for record in records)
    precision = safe_ratio(relevant_found, predicted_total)
    recall = safe_ratio(relevant_found, relevant_total)
    f1 = safe_ratio(2 * precision * recall, precision + recall)

    return {
        "examples": count,
        "scopeAccuracy": rounded(safe_ratio(correct_scope, count)),
        "topRouteAccuracy": rounded(safe_ratio(correct_top, count)),
        "exactSetAccuracy": rounded(safe_ratio(exact_sets, count)),
        "routePrecision": rounded(precision),
        "relevantRouteRecall": rounded(recall),
        "routeF1": rounded(f1),
        "routeCountMeanAbsoluteError": rounded(
            sum(
                abs(len(record["predictedRouteIds"]) - len(record["expectedRouteIds"]))
                for record in records
            )
            / count
        ),
        "fallbackRate": rounded(
            safe_ratio(sum(record["needsFallback"] for record in records), count)
        ),
        "averageLatencyMs": rounded(
            sum(record["durationMs"] for record in records) / count
        ),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    registry_path: Path,
    data_path: Path,
) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    registry_path = registry_path.resolve()
    data_path = data_path.resolve()
    service = RoutingService(checkpoint_settings(checkpoint_path, registry_path))
    examples = load_test_examples(data_path, set(service.routes_by_id))
    records: list[dict] = []

    for example in examples:
        started_at = time.perf_counter()
        result = service.rank(RankRequest(query=example["query"]))
        duration_ms = (time.perf_counter() - started_at) * 1000
        expected_ids = set(example["relevantRouteIds"])
        predicted_ids = set(result.routeIds or ([result.routeId] if result.routeId else []))
        records.append(
            {
                "query": example["query"],
                "categories": inferred_categories(example, service.routes_by_id),
                "critical": example["critical"],
                "expectedScope": example["scope"],
                "predictedScope": result.requestScope,
                "expectedRouteIds": expected_ids,
                "predictedRouteIds": predicted_ids,
                "predictedTopRouteId": result.routeId,
                "confidence": result.confidence,
                "needsFallback": result.needsFallback,
                "fallbackReasons": result.fallbackReasons,
                "durationMs": duration_ms,
            }
        )

    categories = sorted({category for record in records for category in record["categories"]})
    category_metrics = {
        category: aggregate_predictions(
            [record for record in records if category in record["categories"]]
        )
        for category in categories
    }
    mismatches = [
        {
            "query": record["query"],
            "categories": record["categories"],
            "critical": record["critical"],
            "expectedScope": record["expectedScope"],
            "predictedScope": record["predictedScope"],
            "expectedRouteIds": sorted(record["expectedRouteIds"]),
            "predictedRouteIds": sorted(record["predictedRouteIds"]),
            "confidence": record["confidence"],
            "needsFallback": record["needsFallback"],
            "fallbackReasons": record["fallbackReasons"],
        }
        for record in records
        if record["predictedScope"] != record["expectedScope"]
        or record["predictedTopRouteId"] not in record["expectedRouteIds"]
        or (
            record["expectedScope"] == "multiple"
            and record["predictedRouteIds"] != record["expectedRouteIds"]
        )
    ]

    return {
        "schemaVersion": 1,
        "model": {
            "version": service.model_version,
            "checkpointPath": str(checkpoint_path),
            "checkpointSha256": file_sha256(checkpoint_path),
            "registryFingerprint": service.registry_fingerprint,
        },
        "evaluationData": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
        },
        "overall": aggregate_predictions(records),
        "multipleRoute": aggregate_predictions(
            [record for record in records if record["expectedScope"] == "multiple"]
        ),
        "critical": aggregate_predictions([record for record in records if record["critical"]]),
        "categories": category_metrics,
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
    }
