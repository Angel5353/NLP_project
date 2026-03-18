from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


class LLMClientError(Exception):
    """Raised when an LLM request fails."""


@dataclass
class LLMConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay_sec: float = 2.0
    timeout: Optional[float] = None


class BaseLLMClient:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def generate_batch(self, prompts: List[str], temperature: float = 0.0) -> List[str]:
        return [self.generate(prompt, temperature=temperature) for prompt in prompts]


class OpenAILLMClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        timeout: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass api_key explicitly."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            ) from e

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)
                else:
                    break

        provider_name = "OpenAI-compatible"
        if self.base_url and "11434" in self.base_url:
            provider_name = "Ollama"
        raise LLMClientError(
            f"{provider_name} generation failed after {self.max_retries} attempts: {last_error}"
        )


class DummyLLMClient(BaseLLMClient):
    def __init__(self, default_response: str = "dummy output") -> None:
        self.default_response = default_response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        prompt_lower = prompt.lower()
        if "reply with exactly one word" in prompt_lower:
            return "insufficient"
        if "rewrite the following legal question" in prompt_lower:
            return "legal clause termination without notice"
        return self.default_response

    def generate_batch(self, prompts: List[str], temperature: float = 0.0) -> List[str]:
        return [self.generate(p, temperature=temperature) for p in prompts]


class OllamaLLMClient(OpenAILLMClient):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key or "ollama",
            base_url=base_url or "http://localhost:11434/v1",
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            timeout=timeout,
        )


_HF_PIPELINE_CACHE: Dict[Tuple[Any, ...], Any] = {}


class HFLocalLLMClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.batch_size = batch_size

        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for hf_local. Install them with: pip install transformers torch accelerate"
            ) from e

        resolved_dtype = None
        if self.torch_dtype == "float16":
            resolved_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            resolved_dtype = torch.bfloat16
        elif self.torch_dtype == "float32":
            resolved_dtype = torch.float32

        cache_key = (
            self.model_name,
            self.device_map,
            self.torch_dtype,
            self.trust_remote_code,
        )
        if cache_key in _HF_PIPELINE_CACHE:
            self.pipe = _HF_PIPELINE_CACHE[cache_key]
            return

        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model_name,
            "trust_remote_code": self.trust_remote_code,
        }
        if resolved_dtype is not None:
            pipeline_kwargs["dtype"] = resolved_dtype
        if self.device_map == "auto":
            pipeline_kwargs["device_map"] = "auto"
        elif self.device_map == "cpu":
            pipeline_kwargs["device"] = -1
        else:
            pipeline_kwargs["device_map"] = self.device_map

        self.pipe = pipeline(**pipeline_kwargs)
        _HF_PIPELINE_CACHE[cache_key] = self.pipe

    def _build_gen_kwargs(self, temperature: float = 0.0, batch_size: Optional[int] = None) -> Dict[str, Any]:
        do_sample = temperature > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "return_full_text": False,
            "batch_size": batch_size or self.batch_size,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
        return gen_kwargs

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        outputs = self.generate_batch([prompt], temperature=temperature)
        return outputs[0] if outputs else ""

    def generate_batch(self, prompts: List[str], temperature: float = 0.0) -> List[str]:
        if not prompts:
            return []

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                outputs = self.pipe(prompts, **self._build_gen_kwargs(temperature=temperature))
                results: List[str] = []
                for item in outputs:
                    if not item:
                        results.append("")
                    elif isinstance(item, list):
                        results.append((item[0].get("generated_text", "") or "").strip())
                    else:
                        results.append((item.get("generated_text", "") or "").strip())
                return results
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)
                else:
                    break

        raise LLMClientError(
            f"Hugging Face local batch generation failed after {self.max_retries} attempts: {last_error}"
        )


def build_llm_client(
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
    timeout: Optional[float] = None,
    hf_max_new_tokens: int = 256,
    hf_device_map: str = "auto",
    hf_torch_dtype: str = "auto",
    hf_trust_remote_code: bool = False,
    hf_batch_size: int = 4,
) -> BaseLLMClient:
    provider = provider.lower()

    if provider == "openai":
        return OpenAILLMClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            timeout=timeout,
        )
    if provider == "ollama":
        return OllamaLLMClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            timeout=timeout,
        )
    if provider == "hf_local":
        return HFLocalLLMClient(
            model_name=model_name,
            max_new_tokens=hf_max_new_tokens,
            device_map=hf_device_map,
            torch_dtype=hf_torch_dtype,
            trust_remote_code=hf_trust_remote_code,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            batch_size=hf_batch_size,
        )
    if provider == "dummy":
        return DummyLLMClient()
    raise ValueError(f"Unsupported provider: {provider}")
