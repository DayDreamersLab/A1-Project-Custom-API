from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .config import settings
from .schemas import RankRequest, RankResponse
from .service import RoutingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.routing_service = RoutingService(settings)
    yield


app = FastAPI(
    title="AMIDS PyTorch Route Ranker",
    version="1.0.0",
    description="Local learned route relevance and single/multiple-scope API.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    service: RoutingService = request.app.state.routing_service
    return {
        "ok": True,
        "modelVersion": service.model_version,
        "registeredRoutes": len(service.routes),
        "registryFingerprint": service.registry_fingerprint,
        "checkpointPath": str(settings.checkpoint_path),
    }


@app.post("/rank", response_model=RankResponse)
def rank_routes(payload: RankRequest, request: Request) -> RankResponse:
    try:
        service: RoutingService = request.app.state.routing_service
        return service.rank(payload)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
