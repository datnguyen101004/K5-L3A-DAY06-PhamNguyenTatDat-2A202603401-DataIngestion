"""Alibaba Cloud embedding provider adapter."""
import os
import time
from typing import Protocol
import requests


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class AlibabaEmbedder:
    """Embedding adapter for Alibaba Cloud MaaS OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "qwen3.7-text-embedding",
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int = 1024,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ALIBABA_API_KEY")
        )
        if not self.api_key:
            raise ValueError("API key is required for AlibabaEmbedder. Set OPENAI_API_KEY in .env.")

        raw_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or "https://ws-b0ojoo4bja1v9457.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )
        # Handle full URL if passed by user
        if "/chat/completions" in raw_url:
            raw_url = raw_url.replace("/chat/completions", "")
        if "/embeddings" in raw_url:
            raw_url = raw_url.replace("/embeddings", "")
        self.base_url = raw_url.rstrip("/")

        self.endpoint = f"{self.base_url}/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_api(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    items = sorted(data["data"], key=lambda x: x["index"])
                    vectors = [item["embedding"] for item in items]
                    return vectors
                else:
                    err_msg = f"API error {response.status_code}: {response.text}"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(err_msg)
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embedding request failed: {e}")

        raise RuntimeError("Embedding failed after max retries")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = self._call_api(batch)
            all_vectors.extend(batch_vectors)
        return all_vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = self._call_api([question])
        return vectors[0]


MyEmbedder = AlibabaEmbedder
