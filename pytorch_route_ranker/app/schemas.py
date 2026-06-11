from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    roleKey: str | None = Field(default=None, max_length=100)
    maxRoutes: int = Field(default=8, ge=1, le=32)
    routeBiases: dict[str, float] = Field(default_factory=dict)

    @field_validator("routeBiases")
    @classmethod
    def bound_route_biases(cls, route_biases: dict[str, float]) -> dict[str, float]:
        return {
            route_id: max(-2.0, min(2.0, float(bias)))
            for route_id, bias in route_biases.items()
        }


class RankedRoute(BaseModel):
    id: str
    title: str
    path: str
    description: str
    score: float


class RankResponse(BaseModel):
    requestScope: Literal["single", "multiple"]
    shouldOpen: bool
    routeId: str | None
    routeIds: list[str]
    routes: list[RankedRoute]
    confidence: float
    scopeProbability: float
    needsFallback: bool
    fallbackReasons: list[
        Literal[
            "low-confidence",
            "no-route-selected",
            "insufficient-multiple-routes",
            "uncertain-request-scope",
        ]
    ]
    explanation: str
    modelVersion: str
    registryFingerprint: str
    durationMs: float
