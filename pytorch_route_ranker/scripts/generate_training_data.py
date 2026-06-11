from collections import Counter, defaultdict
from pathlib import Path
import json
import re

from pytorch_route_ranker.app.config import RANKER_ROOT
from pytorch_route_ranker.app.registry import load_registry


REGISTRY_PATH = RANKER_ROOT / "data" / "route_registry.json"
OUTPUT_PATH = RANKER_ROOT / "data" / "generated_training_examples.jsonl"

PURPOSE_LABELS = {
    "overview": ["overview", "summary dashboard", "quick situation awareness"],
    "current-observations": ["current observations", "latest conditions", "observed data"],
    "forecast-trend": ["forecast trend", "forecast", "expected conditions"],
    "alert-monitor": ["alerts", "warnings", "active advisories"],
    "runway-impact": ["runway data", "runway impact", "landing and takeoff impact"],
    "route-impact": ["route impact", "enroute impact", "airway impact"],
    "alternate-comparison": ["alternate comparison", "diversion planning", "alternate data"],
    "pilot-briefing": ["pilot briefing", "flight deck briefing", "crew briefing"],
    "atc-operations": ["ATC operations", "controller operations", "flow and spacing"],
    "dispatcher-planning": ["dispatcher planning", "dispatch planning", "release planning"],
    "map-overlay": ["map overlay", "weather map", "geospatial overlay"],
    "source-reports": ["source reports", "official reports", "source documents"],
    "procedure-checklist": ["procedure checklist", "operational checklist", "response procedure"],
    "threshold-monitor": ["threshold monitor", "limits and minima", "exceedance monitor"],
    "history-trend": ["history trend", "historical data", "recent trend"],
    "audit-trail": ["audit trail", "usage history", "traceability"],
}


def normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def route_topic(route: dict) -> str:
    for keyword in route["keywords"]:
        normalized_keyword = normalize_label(keyword)
        if normalized_keyword:
            return normalized_keyword

    return normalize_label(route["title"]) or normalize_label(route["id"])


def route_purpose(route: dict) -> str | None:
    route_id = str(route["id"]).lower()
    for purpose in sorted(PURPOSE_LABELS, key=len, reverse=True):
        if route_id.endswith(f"-{purpose}"):
            return purpose

    searchable_metadata = normalize_label(
        " ".join(
            [
                route["title"],
                route["description"],
                *route["keywords"],
            ]
        )
    )
    for purpose, labels in PURPOSE_LABELS.items():
        if any(normalize_label(label) in searchable_metadata for label in labels):
            return purpose

    return None


def add_example(
    examples_by_query: dict[str, dict],
    conflicting_queries: set[str],
    query: str,
    scope: str,
    route_ids: list[str],
    source: str,
) -> None:
    normalized_query = normalize_label(query)
    normalized_route_ids = sorted(set(route_ids))
    if not normalized_query or not normalized_route_ids or normalized_query in conflicting_queries:
        return

    example = {
        "query": query.strip(),
        "scope": scope,
        "relevantRouteIds": normalized_route_ids,
        "source": source,
    }
    existing_example = examples_by_query.get(normalized_query)
    if not existing_example:
        examples_by_query[normalized_query] = example
        return

    existing_label = (
        existing_example["scope"],
        tuple(existing_example["relevantRouteIds"]),
    )
    new_label = (scope, tuple(normalized_route_ids))
    if existing_label != new_label:
        examples_by_query.pop(normalized_query)
        conflicting_queries.add(normalized_query)


def main() -> None:
    routes = load_registry(REGISTRY_PATH)
    routes_by_topic: dict[str, list[dict]] = defaultdict(list)
    routes_by_purpose: dict[str, list[dict]] = defaultdict(list)
    routes_by_topic_and_purpose: dict[tuple[str, str], list[dict]] = defaultdict(list)
    keyword_counts = Counter(
        str(keyword).lower()
        for route in routes
        for keyword in route["keywords"]
    )

    for route in routes:
        topic = route_topic(route)
        routes_by_topic[topic].append(route)
        purpose = route_purpose(route)
        if purpose:
            routes_by_purpose[purpose].append(route)
            routes_by_topic_and_purpose[(topic, purpose)].append(route)

    examples_by_query: dict[str, dict] = {}
    conflicting_queries: set[str] = set()

    for route in routes:
        topic = route_topic(route)
        purpose = route_purpose(route)
        title = route["title"].strip()
        route_ids = [route["id"]]

        command_queries = {
            f"open {title}",
            f"show me {title}",
            f"navigate to {title}",
            f"find {title}",
        }
        fragment_queries = {
            title,
            f"{title} please",
            f"{title} information",
            f"looking for {title}",
            f"where is {title}",
            f"where can I find {title}",
        }

        unique_keywords = [
            normalize_label(keyword)
            for keyword in route["keywords"]
            if normalize_label(keyword)
            and keyword_counts[str(keyword).lower()] <= 2
        ][:3]
        for keyword in unique_keywords:
            fragment_queries.update(
                {
                    keyword,
                    f"{keyword} please",
                    f"{keyword} information",
                    f"looking for {keyword}",
                }
            )

        if purpose and len(routes_by_topic_and_purpose[(topic, purpose)]) == 1:
            purpose_label = PURPOSE_LABELS[purpose][0]
            fragment_queries.update(
                {
                    f"{topic} {purpose_label}",
                    f"{topic} {purpose_label} please",
                    f"where is the {topic} {purpose_label}",
                }
            )

        for query in sorted(command_queries):
            add_example(
                examples_by_query,
                conflicting_queries,
                query,
                "single",
                route_ids,
                "generated-route-command",
            )
        for query in sorted(fragment_queries):
            add_example(
                examples_by_query,
                conflicting_queries,
                query,
                "single",
                route_ids,
                "generated-route-fragment",
            )

    for topic, topic_routes in routes_by_topic.items():
        if len(topic_routes) < 2:
            continue
        route_ids = [route["id"] for route in topic_routes]
        command_queries = {
            f"show all {topic} data",
            f"give me every {topic} source",
            f"open the complete set of {topic} information",
            f"I need everything related to {topic}",
        }
        fragment_queries = {
            topic,
            f"{topic} data",
            f"{topic} information",
            f"{topic} sources",
            f"all {topic}",
            f"all {topic} data",
            f"every {topic} source",
            f"complete {topic} information",
            f"everything related to {topic}",
            f"what {topic} information is available",
        }
        for query in sorted(command_queries):
            add_example(
                examples_by_query,
                conflicting_queries,
                query,
                "multiple",
                route_ids,
                "generated-topic-command",
            )
        for query in sorted(fragment_queries):
            add_example(
                examples_by_query,
                conflicting_queries,
                query,
                "multiple",
                route_ids,
                "generated-topic-fragment",
            )

    for purpose, purpose_routes in routes_by_purpose.items():
        if len(purpose_routes) < 2:
            continue
        route_ids = [route["id"] for route in purpose_routes]
        for purpose_label in PURPOSE_LABELS[purpose]:
            command_queries = {
                f"show all {purpose_label}",
                f"open every {purpose_label} source",
                f"give me complete {purpose_label} information",
            }
            fragment_queries = {
                purpose_label,
                f"{purpose_label} information",
                f"{purpose_label} sources",
                f"all {purpose_label}",
                f"every {purpose_label} source",
                f"complete {purpose_label} information",
            }
            for query in sorted(command_queries):
                add_example(
                    examples_by_query,
                    conflicting_queries,
                    query,
                    "multiple",
                    route_ids,
                    "generated-purpose-command",
                )
            for query in sorted(fragment_queries):
                add_example(
                    examples_by_query,
                    conflicting_queries,
                    query,
                    "multiple",
                    route_ids,
                    "generated-purpose-fragment",
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for normalized_query in sorted(examples_by_query):
            output_file.write(
                json.dumps(examples_by_query[normalized_query], ensure_ascii=True) + "\n"
            )

    print(
        f"Generated {len(examples_by_query)} training examples at {OUTPUT_PATH}; "
        f"removed {len(conflicting_queries)} conflicting queries."
    )


if __name__ == "__main__":
    main()
