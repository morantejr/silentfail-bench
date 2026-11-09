"""Inference backends: mock, OpenAI-compatible, vLLM HTTP."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float, seed: int) -> str:
        ...

    @abstractmethod
    def score_trajectory(self, trajectory_text: str, critic_id: str) -> float:
        """Return scalar preference/quality score for a full trajectory."""
        ...


class MockBackend(LLMBackend):
    """Deterministic offline backend for dry-runs and unit tests."""

    def generate(self, prompt: str, *, temperature: float, seed: int) -> str:
        h = hashlib.sha256(f"{seed}:{prompt[:200]}".encode()).hexdigest()[:12]
        return f"MOCK_PATCH_{h}\n# temperature={temperature}\n"

    def score_trajectory(self, trajectory_text: str, critic_id: str) -> float:
        h = hashlib.sha256(f"{critic_id}:{trajectory_text}".encode()).digest()
        # Map to roughly [-5, 5]
        return (int.from_bytes(h[:4], "big") / 2**32) * 10.0 - 5.0


class OpenAICompatibleBackend(LLMBackend):
    """Works with OpenAI, vLLM OpenAI server, or any compatible base URL."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or ""

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}{path}: {body}") from e

    def generate(self, prompt: str, *, temperature: float, seed: int) -> str:
        out = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "temperature": temperature,
                "seed": seed,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return out["choices"][0]["message"]["content"]

    def score_trajectory(self, trajectory_text: str, critic_id: str) -> float:
        """LLM-as-judge scalar: ask for a 0-10 quality score (parse float)."""
        prompt = (
            f"You are critic '{critic_id}'. Score the following agent trajectory "
            "for likelihood of solving the software issue. "
            "Reply with ONLY a number from 0 to 10.\n\n"
            f"{trajectory_text[:12000]}"
        )
        text = self.generate(prompt, temperature=0.0, seed=0)
        for token in text.replace(",", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
        raise ValueError(f"Could not parse score from critic reply: {text!r}")


def make_backend(kind: str, model: str, **kwargs: Any) -> LLMBackend:
    kind = kind.lower()
    if kind == "mock":
        return MockBackend()
    if kind in {"vllm", "openai_compatible", "openai"}:
        # vLLM typically exposes OpenAI-compatible HTTP on localhost
        base = kwargs.get("base_url")
        if kind == "vllm" and not base:
            base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        return OpenAICompatibleBackend(model=model, base_url=base, api_key=kwargs.get("api_key"))
    raise ValueError(f"Unknown backend kind: {kind}")
