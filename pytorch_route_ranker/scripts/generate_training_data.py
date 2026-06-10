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


def write_example(output_file, query: str, scope: str, route_ids: list[str], source: str) -> None:
    output_file.write(
        json.dumps(
            {
                "query": query,
                "scope": scope,
                "relevantRouteIds": route_ids,
                "source": source,
            },
            ensure_ascii=True,
        )
        + "\n"
    )


def main() -> None:
    routes = load_registry(REGISTRY_PATH)
    routes_by_topic: dict[str, list[dict]] = defaultdict(list)
    routes_by_purpose: dict[str, list[dict]] = defaultdict(list)
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for route in routes:
            topic = route_topic(route)
            purpose = route_purpose(route)
            queries = {
                f"open {route['title']}",
                f"show me {route['title']}",
                f"navigate to {route['title']}",
                f"I need {topic}",
                f"give me {topic} data",
            }
            if purpose:
                purpose_label = PURPOSE_LABELS[purpose][0]
                queries.update(
                    {
                        f"I need {topic} {purpose_label}",
                        f"give me {topic} {purpose_label} data",
                    }
                )
            unique_keywords = [
                normalize_label(keyword)
                for keyword in route["keywords"]
                if normalize_label(keyword)
                and keyword_counts[str(keyword).lower()] <= 2
            ][:2]
            queries.update(f"find {keyword}" for keyword in unique_keywords)

            for query in sorted(queries):
                write_example(output_file, query, "single", [route["id"]], "generated-route")

        for topic, topic_routes in routes_by_topic.items():
            if len(topic_routes) < 2:
                continue
            route_ids = [route["id"] for route in topic_routes]
            for query in [
                f"show all {topic} data",
                f"give me every {topic} source",
                f"open the complete set of {topic} information",
                f"I need everything related to {topic}",
            ]:
                write_example(output_file, query, "multiple", route_ids, "generated-topic")

        for purpose, purpose_routes in routes_by_purpose.items():
            if len(purpose_routes) < 2:
                continue
            route_ids = [route["id"] for route in purpose_routes]
            for purpose_label in PURPOSE_LABELS[purpose]:
                for query in [
                    f"show all {purpose_label}",
                    f"open every {purpose_label} source",
                    f"give me complete {purpose_label} information",
                ]:
                    write_example(output_file, query, "multiple", route_ids, "generated-purpose")

    line_count = sum(1 for _ in OUTPUT_PATH.open("r", encoding="utf-8"))
    print(f"Generated {line_count} training examples at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
