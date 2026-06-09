import torch
from torch import nn


class AmidsRouteRanker(nn.Module):
    """Learns route relevance and whether a query requests multiple routes."""

    def __init__(self, feature_dimension: int = 4096, hidden_dimension: int = 128):
        super().__init__()
        self.feature_dimension = feature_dimension
        self.hidden_dimension = hidden_dimension

        self.query_projection = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.ReLU(),
            nn.LayerNorm(hidden_dimension),
        )
        self.route_projection = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.ReLU(),
            nn.LayerNorm(hidden_dimension),
        )
        self.relevance_head = nn.Sequential(
            nn.Linear(hidden_dimension * 4, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dimension, 1),
        )
        self.scope_head = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dimension, 1),
        )

    def rank_logits(
        self,
        query_vectors: torch.Tensor,
        route_vectors: torch.Tensor,
    ) -> torch.Tensor:
        query_embeddings = self.query_projection(query_vectors)
        route_embeddings = self.route_projection(route_vectors)

        query_embeddings = query_embeddings[:, None, :]
        route_embeddings = route_embeddings[None, :, :]
        query_embeddings = query_embeddings.expand(-1, route_embeddings.shape[1], -1)
        route_embeddings = route_embeddings.expand(query_embeddings.shape[0], -1, -1)

        pair_features = torch.cat(
            [
                query_embeddings,
                route_embeddings,
                torch.abs(query_embeddings - route_embeddings),
                query_embeddings * route_embeddings,
            ],
            dim=-1,
        )
        return self.relevance_head(pair_features).squeeze(-1)

    def scope_logits(self, query_vectors: torch.Tensor) -> torch.Tensor:
        return self.scope_head(query_vectors).squeeze(-1)

    def forward(
        self,
        query_vectors: torch.Tensor,
        route_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rank_logits(query_vectors, route_vectors), self.scope_logits(query_vectors)
