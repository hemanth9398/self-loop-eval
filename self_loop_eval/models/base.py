"""Base LLM wrapper with provider-agnostic interface."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from self_loop_eval.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    content: str
    reasoning: str = ""
    score: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "score": self.score,
            "metadata": self.metadata,
        }


class LLMWrapper(ABC):
    """Abstract base wrapper for LLM providers.

    Subclasses implement the actual API call; this base provides
    common prompt formatting and response parsing.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._local_model = None
        self._local_tokenizer = None

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Send a prompt to the LLM and return the raw text response.

        Args:
            prompt: The user prompt.
            system_prompt: An optional system/role prompt.

        Returns:
            The model's text response.
        """

    def _load_local_model(self) -> None:
        """Load a local HuggingFace model with optional quantization."""
        if self._local_model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = self.config.local_model_path or self.config.model_name

        model_kwargs: dict = {"device_map": self.config.device_map}

        if self.config.quantization in ("4bit", "8bit"):
            try:
                import torch
                from transformers import BitsAndBytesConfig

                if self.config.quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
            except ImportError:
                logger.warning(
                    "bitsandbytes not available — loading model without quantization. "
                    "Install with: pip install bitsandbytes"
                )
            except Exception:
                logger.warning(
                    "Quantization failed (CUDA may not be available) — "
                    "loading model without quantization."
                )
                model_kwargs.pop("quantization_config", None)

        logger.info("Loading local model: %s", model_path)
        self._local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            )
        except Exception:
            # Fall back without quantization if initial load fails
            model_kwargs.pop("quantization_config", None)
            logger.warning("Retrying model load without quantization.")
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs
            )

        if self._local_tokenizer.pad_token is None:
            self._local_tokenizer.pad_token = self._local_tokenizer.eos_token

    def _generate_local(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response using a local HuggingFace model.

        Args:
            prompt: The user prompt.
            system_prompt: An optional system/role prompt.

        Returns:
            The model's text response.
        """
        self._load_local_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            input_text = self._local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without a chat template
            parts = []
            if system_prompt:
                parts.append(f"System: {system_prompt}")
            parts.append(f"User: {prompt}")
            parts.append("Assistant:")
            input_text = "\n".join(parts)

        inputs = self._local_tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self._local_model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        import torch

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=max(self.config.temperature, 1e-7),
                do_sample=self.config.temperature > 0,
                pad_token_id=self._local_tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        return self._local_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def generate_structured(
        self, prompt: str, system_prompt: str = ""
    ) -> LLMResponse:
        """Generate a response and attempt to parse structured fields.

        The model is asked to respond in JSON with 'content', 'reasoning',
        and optionally 'score' fields. Falls back to unstructured if parsing fails.

        Args:
            prompt: The user prompt.
            system_prompt: System prompt.

        Returns:
            An LLMResponse with parsed fields.
        """
        structured_prompt = (
            f"{prompt}\n\n"
            "Respond in JSON format with the following fields:\n"
            '- "content": your solution or answer\n'
            '- "reasoning": your step-by-step reasoning / chain of thought\n'
            '- "score": (optional) a self-assessed score from 0.0 to 1.0\n'
        )

        raw = self.generate(structured_prompt, system_prompt)
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> LLMResponse:
        """Parse a raw LLM response into an LLMResponse object."""
        # Try to extract JSON from the response
        try:
            # Handle ```json ... ``` blocks
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw.strip()

            data = json.loads(json_str)
            return LLMResponse(
                content=data.get("content", raw),
                reasoning=data.get("reasoning", ""),
                score=data.get("score"),
                metadata={"raw": raw, "parsed": True},
            )
        except (json.JSONDecodeError, IndexError):
            logger.debug("Could not parse structured response, using raw text.")
            return LLMResponse(
                content=raw,
                reasoning="",
                score=None,
                metadata={"raw": raw, "parsed": False},
            )
