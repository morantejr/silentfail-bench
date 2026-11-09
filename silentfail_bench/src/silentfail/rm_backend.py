"""True scalar reward models on Apple Silicon (MPS).

This closes the construct gap in the free-compute path. With LLM-as-judge
critics, u is *judge-prompt disagreement*; the proposal theorizes about
*reward-model ensemble disagreement*. These are genuine `*ForSequenceClassification`
heads emitting a scalar preference score, which is the quantity the design
actually names.

Models are loaded ONE AT A TIME -- an 8B in fp16 is ~16 GB on a 24 GB machine,
so the runner must stay staged by model.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class RewardModel:
    """Wraps a scalar RM. `score()` returns the reward head's logit."""

    repo_id: str
    critic_id: str
    device: str
    max_length: int = 4096
    _tok: object = None
    _model: object = None

    @classmethod
    def load(cls, repo_id: str, critic_id: str, *, max_length: int = 4096) -> "RewardModel":
        device = pick_device()
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
            num_labels=1,
        ).to(device).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tok.pad_token_id
        return cls(repo_id=repo_id, critic_id=critic_id, device=device,
                   max_length=max_length, _tok=tok, _model=model)

    @torch.no_grad()
    def score(self, task_prompt: str, trajectory_text: str) -> float:
        """Scalar reward for (task, trajectory), via the model's chat template."""
        msgs = [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": trajectory_text},
        ]
        try:
            text = self._tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = f"User: {task_prompt}\n\nAssistant: {trajectory_text}"
        enc = self._tok(text, return_tensors="pt", truncation=True,
                        max_length=self.max_length).to(self.device)
        out = self._model(**enc)
        return float(out.logits[0].float().flatten()[0].item())

    def unload(self) -> None:
        """Free the weights before the next panel member loads."""
        self._model = None
        self._tok = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
