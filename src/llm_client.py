from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


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
    """
    Minimal interface expected by pipelines.py.
    """

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI chat-completions wrapper.

    Reads API key from:
        - explicit api_key argument, or
        - OPENAI_API_KEY environment variable
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        timeout: Optional[float] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
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

        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)
                else:
                    break

        raise LLMClientError(
            f"OpenAI generation failed after {self.max_retries} attempts: {last_error}"
        )


class DummyLLMClient(BaseLLMClient):
    """
    Useful for debugging pipeline wiring without making API calls.
    """

    def __init__(self, default_response: str = "dummy output") -> None:
        self.default_response = default_response

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        prompt_lower = prompt.lower()

        # crude behavior so agentic pipeline can be smoke-tested
        if "reply with exactly one word" in prompt_lower:
            return "insufficient"
        if "rewrite the following legal question" in prompt_lower:
            return "legal clause termination without notice"
        return self.default_response


def build_llm_client(
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
    timeout: Optional[float] = None,
) -> BaseLLMClient:
    """
    Factory function for creating an LLM client.
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAILLMClient(
            model_name=model_name,
            api_key=api_key,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            timeout=timeout,
        )

    if provider == "dummy":
        return DummyLLMClient()

    raise ValueError(f"Unsupported provider: {provider}")