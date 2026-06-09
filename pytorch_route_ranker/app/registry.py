from hashlib import sha256
from pathlib import Path
import json


REQUIRED_ROUTE_FIELDS = {"id", "title", "path", "description", "keywords"}


def load_registry(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as registry_file:
        routes = json.load(registry_file)

    if not isinstance(routes, list) or not routes:
        raise ValueError("The route registry must be a non-empty JSON array.")

    route_ids: set[str] = set()
    for index, route in enumerate(routes):
        if not isinstance(route, dict) or set(route) != REQUIRED_ROUTE_FIELDS:
            raise ValueError(
                f"Route {index} must contain exactly: {sorted(REQUIRED_ROUTE_FIELDS)}"
            )
        if not isinstance(route["id"], str) or not route["id"]:
            raise ValueError(f"Route {index} has an invalid id.")
        if route["id"] in route_ids:
            raise ValueError(f"Duplicate route id: {route['id']}")
        if not isinstance(route["keywords"], list):
            raise ValueError(f"Route {route['id']} keywords must be a list.")
        route_ids.add(route["id"])

    return routes


def registry_fingerprint(routes: list[dict]) -> str:
    encoded = json.dumps(routes, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def route_text(route: dict) -> str:
    keywords = " ".join(str(keyword) for keyword in route["keywords"])
    return f"{route['title']} {route['description']} {keywords}"
