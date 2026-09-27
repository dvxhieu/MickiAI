"""LLM client abstraction.

Supports Anthropic, OpenAI and Ollama (local) backends through a single
interface. The Agent only ever talks to this class — swapping providers is a
one-line config change.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around a provider's chat-completion API."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or settings.llm_provider
        self._client = None
        self._model = settings.default_model
        self._init_client()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _init_client(self) -> None:
        if self.provider == "anthropic":
            from anthropic import Anthropic

            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is empty")
            self._client = Anthropic(api_key=settings.anthropic_api_key)
            self._model = settings.default_model or "claude-sonnet-4-6"

        elif self.provider == "openai":
            from openai import OpenAI

            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is empty")
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.default_model or "gpt-4o"

        elif self.provider == "ollama":
            from ollama import Client

            self._client = Client(host=settings.ollama_base_url)
            self._model = settings.ollama_model

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ------------------------------------------------------------------
    # Non-streaming completion (used by the Agent tool loop)
    # ------------------------------------------------------------------
    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        """Send messages to the LLM and return the raw response object."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": settings.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        if self.provider == "anthropic":
            return self._client.messages.create(**kwargs)

        if self.provider == "openai":
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = settings.max_tokens
            return self._client.chat.completions.create(**kwargs)

        # Ollama
        kwargs.pop("max_tokens", None)
        return self._client.chat(model=self._model, messages=messages, tools=tools or [])

    # ------------------------------------------------------------------
    # Streaming completion (used by the SSE endpoint)
    # ------------------------------------------------------------------
    async def generate_stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive from the provider."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        if self.provider == "anthropic":
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        elif self.provider == "openai":
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        else:  # Ollama
            async for chunk in await self._client.chat(
                model=self._model, messages=messages, stream=True
            ):
                delta = chunk["message"]["content"]
                if delta:
                    yield delta

    # ------------------------------------------------------------------
    # Response parsing helpers (normalize provider differences)
    # ------------------------------------------------------------------
    def extract_text(self, response: Any) -> str:
        """Pull the concatenated text out of a provider response."""
        if self.provider == "anthropic":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )
        if self.provider == "openai":
            return response.choices[0].message.content or ""
        return response["message"]["content"] or ""

    def extract_tool_calls(self, response: Any) -> list[dict]:
        """Return a normalized list of tool-call dicts.

        Each dict has keys: id, name, input.
        """
        if self.provider == "anthropic":
            return [
                {
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
                for block in response.content
                if block.type == "tool_use"
            ]

        if self.provider == "openai":
            calls = response.choices[0].message.tool_calls or []
            import json

            return [
                {
                    "id": c.id,
                    "name": c.function.name,
                    "input": json.loads(c.function.arguments or "{}"),
                }
                for c in calls
            ]

        # Ollama
        return [
            {
                "id": f"call_{i}",
                "name": tc["function"]["name"],
                "input": tc["function"]["arguments"],
            }
            for i, tc in enumerate(response.get("message", {}).get("tool_calls") or [])
        ]

    def build_assistant_message(self, response: Any) -> dict:
        """Build the assistant message dict to append to history."""
        if self.provider == "anthropic":
            return {"role": "assistant", "content": response.content}

        if self.provider == "openai":
            return {"role": "assistant", "content": response.choices[0].message.content or ""}

        return {"role": "assistant", "content": response.get("message", {}).get("content", "")}

    def build_tool_result_message(self, tool_call_id: str, content: str) -> dict:
        """Build the tool-result message dict for the given provider."""
        if self.provider == "anthropic":
            return {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}
                ],
            }
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}