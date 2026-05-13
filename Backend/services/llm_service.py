"""
LangChain-based LLM Service - Modular provider support
Supports: Groq, NVIDIA
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class LLMService:
    """Unified LLM service that supports only Groq and NVIDIA."""

    SUPPORTED_PROVIDERS = ("groq", "nvidia")

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        logger=None,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported: {list(self.SUPPORTED_PROVIDERS)}"
            )

        self.client = self._initialize_client()

        if self.logger:
            self.logger.info(f"LLMService initialized with provider: {self.provider}")
            self.logger.info(f"Model: {self.model_name}, Temperature: {self.temperature}")

    def _initialize_client(self) -> BaseChatModel:
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            return ChatGroq(
                api_key=api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        if self.provider == "nvidia":
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not found in environment")
            return ChatNVIDIA(
                api_key=api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_client_with_overrides(
        self, temperature: Optional[float], max_tokens: Optional[int]
    ) -> BaseChatModel:
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.provider == "groq":
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name=self.model_name,
                temperature=temp,
                max_tokens=tokens,
            )

        if self.provider == "nvidia":
            return ChatNVIDIA(
                api_key=os.getenv("NVIDIA_API_KEY"),
                model=self.model_name,
                temperature=temp,
                max_tokens=tokens,
            )

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        client = (
            self._get_client_with_overrides(temperature, max_tokens)
            if temperature is not None or max_tokens is not None
            else self.client
        )
        response = client.invoke(messages)

        if not response or not response.content:
            raise RuntimeError("LLM returned empty response")

        return response.content

    @staticmethod
    def from_env(logger=None) -> "LLMService":
        provider = os.getenv("LLM_PROVIDER")
        if not provider:
            raise ValueError("LLM_PROVIDER not set in environment")

        provider_lower = provider.lower()
        if provider_lower not in LLMService.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

        model_name = os.getenv("LLM_MODEL_NAME")
        if not model_name:
            raise ValueError("LLM_MODEL_NAME not set in environment")

        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        except ValueError:
            temperature = 0.7
        if not (0.0 <= temperature <= 2.0):
            temperature = 0.7

        try:
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        except ValueError:
            max_tokens = 2000
        if max_tokens < 1:
            max_tokens = 2000

        if provider_lower == "groq" and not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY required for Groq provider")
        if provider_lower == "nvidia" and not os.getenv("NVIDIA_API_KEY"):
            raise ValueError("NVIDIA_API_KEY required for NVIDIA provider")

        return LLMService(
            provider=provider_lower,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logger=logger,
        )
