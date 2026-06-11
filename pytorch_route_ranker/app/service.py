from pathlib import Path
import re
import time

import torch

from .config import Settings
from .model import AmidsRouteRanker
from .registry import load_registry, registry_fingerprint, route_text
from .schemas import RankRequest, RankResponse, RankedRoute
from .text_features import HashingTextVectorizer


NAVIGATION_WORDS = {
    "bring",
    "display",
    "find",
    "get",
    "give",
    "go",
    "load",
    "navigate",
    "need",
    "open",
    "show",
    "take",
    "view",
}
BROAD_PATTERN = re.compile(
    r"\b(all|every|everything|each|complete|comprehensive|entire|full|whole)\b",
    re.IGNORECASE,
)
BROAD_NEGATION_PATTERN = re.compile(
    r"\b(not|don't|do not|without)(?:\s+[a-z0-9/-]+){0,3}\s+"
    r"(all|every|everything|each|complete|comprehensive|entire|full|whole)\b",
    re.IGNORECASE,
)
SINGLE_PATTERN = re.compile(
    r"\b(best|closest|most appropriate|single|one route|one page|only one)\b",
    re.IGNORECASE,
)


class RoutingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.routes = load_registry(settings.registry_path)
        self.routes_by_id = {route["id"]: route for route in self.routes}
        self.registry_fingerprint = registry_fingerprint(self.routes)
        self.vectorizer = HashingTextVectorizer(settings.feature_dimension)
        self.route_vectors = self.vectorizer.transform([route_text(route) for route in self.routes])
        self.model, self.model_version = self._load_model(settings.checkpoint_path)

    def _load_model(self, checkpoint_path: Path) -> tuple[AmidsRouteRanker, str]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint_path}. Run npm run ranker:train first."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_fingerprint = checkpoint.get("registry_fingerprint")
        if checkpoint_fingerprint != self.registry_fingerprint:
            raise ValueError(
                "The model was trained against a different route registry. "
                "Export the registry and retrain the model."
            )
        if int(checkpoint["feature_dimension"]) != self.settings.feature_dimension:
            raise ValueError(
                "The checkpoint feature dimension does not match AMIDS_RANKER_FEATURE_DIMENSION."
            )

        model = AmidsRouteRanker(
            feature_dimension=int(checkpoint["feature_dimension"]),
            hidden_dimension=int(checkpoint["hidden_dimension"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        return model, str(checkpoint.get("model_version", "unknown"))

    def _query_text(self, request: RankRequest) -> str:
        role_context = f" user role {request.roleKey}" if request.roleKey else ""
        return f"{request.query}{role_context}"

    def _scope_constraint(self, query: str) -> str | None:
        if BROAD_NEGATION_PATTERN.search(query):
            return "single"
        if BROAD_PATTERN.search(query):
            return "multiple"
        if SINGLE_PATTERN.search(query):
            return "single"
        return None

    def _should_open(self, query: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        return bool(tokens & NAVIGATION_WORDS) or self._scope_constraint(query) == "multiple"

    def _route_bias_tensor(self, request: RankRequest) -> torch.Tensor:
        return torch.tensor(
            [request.routeBiases.get(route["id"], 0.0) for route in self.routes],
            dtype=torch.float32,
        )

    def rank(self, request: RankRequest) -> RankResponse:
        started_at = time.perf_counter()
        query_vector = self.vectorizer.transform_one(self._query_text(request)).unsqueeze(0)

        with torch.inference_mode():
            relevance_logits, scope_logits = self.model(query_vector, self.route_vectors)
            relevance_logits = relevance_logits.squeeze(0) + self._route_bias_tensor(request)
            relevance_scores = torch.sigmoid(relevance_logits)
            scope_probability = float(torch.sigmoid(scope_logits)[0])

        ranked_indices = torch.argsort(relevance_scores, descending=True).tolist()
        top_index = ranked_indices[0]
        top_score = float(relevance_scores[top_index])
        second_score = float(relevance_scores[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0

        scope_constraint = self._scope_constraint(request.query)
        wants_multiple = (
            scope_constraint == "multiple"
            or (scope_constraint is None and scope_probability >= self.settings.scope_threshold)
        )
        if scope_constraint == "single":
            wants_multiple = False

        route_limit = min(request.maxRoutes, self.settings.maximum_routes)
        if wants_multiple:
            selection_threshold = max(
                self.settings.minimum_relevance,
                top_score * self.settings.relative_multiple_threshold,
            )
            selected_indices = [
                index
                for index in ranked_indices
                if float(relevance_scores[index]) >= selection_threshold
            ][:route_limit]
        else:
            selected_indices = [top_index]

        selected_routes = [self.routes[index] for index in selected_indices]
        selected_scores = [float(relevance_scores[index]) for index in selected_indices]
        confidence = (
            sum(selected_scores) / len(selected_scores)
            if wants_multiple and selected_scores
            else max(0.0, min(1.0, top_score * 0.8 + max(0.0, top_score - second_score) * 0.2))
        )
        scope_is_uncertain = (
            scope_constraint is None
            and abs(scope_probability - self.settings.scope_threshold) < 0.12
        )
        fallback_reasons = []
        if confidence < self.settings.minimum_confidence:
            fallback_reasons.append("low-confidence")
        if not selected_routes:
            fallback_reasons.append("no-route-selected")
        if wants_multiple and len(selected_routes) < 2:
            fallback_reasons.append("insufficient-multiple-routes")
        if scope_is_uncertain:
            fallback_reasons.append("uncertain-request-scope")
        needs_fallback = bool(fallback_reasons)
        if fallback_reasons:
            explanation = (
                "PyTorch ranker could not confirm the route because "
                + ", ".join(fallback_reasons)
                + "."
            )
        else:
            explanation = (
                f"PyTorch ranker selected {len(selected_routes)} approved route"
                f"{'s' if len(selected_routes) != 1 else ''} with {confidence:.0%} confidence."
            )
        request_scope = "multiple" if wants_multiple else "single"
        route_ids = [route["id"] for route in selected_routes]
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

        return RankResponse(
            requestScope=request_scope,
            shouldOpen=self._should_open(request.query),
            routeId=route_ids[0] if route_ids else None,
            routeIds=route_ids if wants_multiple else [],
            routes=[
                RankedRoute(
                    id=route["id"],
                    title=route["title"],
                    path=route["path"],
                    description=route["description"],
                    score=round(score, 6),
                )
                for route, score in zip(selected_routes, selected_scores)
            ],
            confidence=round(confidence, 6),
            scopeProbability=round(scope_probability, 6),
            needsFallback=needs_fallback,
            fallbackReasons=fallback_reasons,
            explanation=explanation,
            modelVersion=self.model_version,
            registryFingerprint=self.registry_fingerprint,
            durationMs=duration_ms,
        )
