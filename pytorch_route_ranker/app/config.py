from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANKER_ROOT = PROJECT_ROOT / "pytorch_route_ranker"


@dataclass(frozen=True)
class Settings:
    registry_path: Path = Path(
        os.getenv("AMIDS_RANKER_REGISTRY_PATH", RANKER_ROOT / "data" / "route_registry.json")
    )
    checkpoint_path: Path = Path(
        os.getenv("AMIDS_RANKER_CHECKPOINT_PATH", RANKER_ROOT / "models" / "route_ranker.pt")
    )
    feature_dimension: int = int(os.getenv("AMIDS_RANKER_FEATURE_DIMENSION", "4096"))
    hidden_dimension: int = int(os.getenv("AMIDS_RANKER_HIDDEN_DIMENSION", "128"))
    scope_threshold: float = float(os.getenv("AMIDS_RANKER_SCOPE_THRESHOLD", "0.5"))
    minimum_relevance: float = float(os.getenv("AMIDS_RANKER_MINIMUM_RELEVANCE", "0.45"))
    relative_multiple_threshold: float = float(
        os.getenv("AMIDS_RANKER_RELATIVE_MULTIPLE_THRESHOLD", "0.72")
    )
    minimum_confidence: float = float(os.getenv("AMIDS_RANKER_MINIMUM_CONFIDENCE", "0.55"))
    maximum_routes: int = int(os.getenv("AMIDS_RANKER_MAXIMUM_ROUTES", "8"))


settings = Settings()
