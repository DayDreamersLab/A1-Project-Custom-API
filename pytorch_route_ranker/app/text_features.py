from hashlib import blake2b
import math
import re

import torch


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class HashingTextVectorizer:
    """Turns text into fixed-size word and character n-gram vectors."""

    def __init__(self, dimension: int = 4096):
        self.dimension = dimension

    def _index_and_sign(self, feature: str) -> tuple[int, float]:
        digest = blake2b(
            feature.encode("utf-8"),
            digest_size=8,
            person=b"amids-rank",
        ).digest()
        raw_value = int.from_bytes(digest, byteorder="little", signed=False)
        return raw_value % self.dimension, 1.0 if raw_value & 1 else -1.0

    def _features(self, text: str) -> list[str]:
        normalized = " ".join(TOKEN_PATTERN.findall(str(text).lower()))
        words = normalized.split()
        features = [f"w1:{word}" for word in words]
        features.extend(f"w2:{words[index]}_{words[index + 1]}" for index in range(len(words) - 1))

        compact = normalized.replace(" ", "_")
        for size in (3, 4):
            features.extend(
                f"c{size}:{compact[index:index + size]}"
                for index in range(max(0, len(compact) - size + 1))
            )
        return features

    def transform_one(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.dimension, dtype=torch.float32)
        features = self._features(text)

        for feature in features:
            index, sign = self._index_and_sign(feature)
            vector[index] += sign

        norm = math.sqrt(float(torch.dot(vector, vector)))
        if norm > 0:
            vector /= norm
        return vector

    def transform(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.dimension), dtype=torch.float32)
        return torch.stack([self.transform_one(text) for text in texts])
