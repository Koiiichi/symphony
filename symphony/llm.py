"""LLM client abstraction — OpenAI and Google Gemini support."""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin provider-agnostic wrapper around OpenAI and Gemini SDKs.

    Provider is selected by which API key is present in the environment,
    or by passing *provider* explicitly. If both keys are set, the
    provider must be specified.

    Supported providers:
      - ``openai``   — requires OPENAI_API_KEY
      - ``gemini``   — requires GEMINI_API_KEY
    """

    OPENAI_MODELS = {
        "default": "gpt-5.4-mini",
        "fast": "gpt-5.4-nano",
    }
    GEMINI_MODELS = {
        "default": "gemini-3-flash-preview",
        "fast": "gemini-3.1-flash-lite-preview",
    }

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider, self.model, self._api_key = self._resolve(
            provider, model, api_key
        )
        logger.info("LLM client: provider=%s model=%s", self.provider, self.model)

    # ---- public API ------------------------------------------------

    def complete(
        self,
        messages: List[dict],
        *,
        system: str = "",
        max_tokens: int = 4096,
        response_schema: Optional[type] = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Pass *response_schema* (a Pydantic BaseModel subclass) to enable
        provider-native structured output enforcement:
          - OpenAI: uses ``client.chat.completions.parse`` with ``response_format``
          - Gemini: sets ``response_mime_type="application/json"`` and
            ``response_schema`` in ``GenerateContentConfig``

        The return value is always a JSON string of the validated object.
        """
        if self.provider == "openai":
            return self._complete_openai(
                messages, system=system, max_tokens=max_tokens,
                response_schema=response_schema,
            )
        return self._complete_gemini(
            messages, system=system, max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def count_tokens(self, text: str) -> int:
        """Return an accurate token count for *text* using the provider's
        native method where available.

        - OpenAI: tiktoken local encoding (no API call needed).
        - Gemini: ``client.models.count_tokens`` native endpoint (one API call).

        Use this for accurate pre-request accounting. For budget enforcement
        inside the PromptCompiler use ``estimate_tokens`` instead — it is
        fast and requires no network round-trip.
        """
        if self.provider == "openai":
            return self._count_tokens_openai(text)
        return self._count_tokens_gemini(text)

    def _count_tokens_openai(self, text: str) -> int:
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4

    def _count_tokens_gemini(self, text: str) -> int:
        try:
            from google import genai
        except ImportError:
            return len(text) // 4
        client = genai.Client(api_key=self._api_key)
        response = client.models.count_tokens(
            model=self.model,
            contents=text,
        )
        return response.total_tokens

    # ---- helpers ---------------------------------------------------

    @staticmethod
    def _gemini_schema(model_cls: type) -> Dict[str, Any]:
        """Convert a Pydantic model to a Gemini-compatible JSON schema.

        Gemini rejects ``additionalProperties`` regardless of its value —
        it appears in Pydantic-generated schemas wherever ``dict[str, Any]``
        is used as a field type. This strips the key recursively.
        """
        def _strip(node: Any) -> Any:
            if isinstance(node, dict):
                node.pop("additionalProperties", None)
                for v in node.values():
                    _strip(v)
            elif isinstance(node, list):
                for item in node:
                    _strip(item)
            return node

        return _strip(model_cls.model_json_schema())

    # ---- provider implementations ----------------------------------

    def _complete_openai(
        self,
        messages: List[dict],
        *,
        system: str,
        max_tokens: int,
        response_schema: Optional[type] = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: uv sync --extra openai"
            )
        client = OpenAI(api_key=self._api_key)
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        if response_schema is not None:
            response = client.chat.completions.parse(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                response_format=response_schema,
            )
            parsed = response.choices[0].message.parsed
            return parsed.model_dump_json() if parsed is not None else ""
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _complete_gemini(
        self,
        messages: List[dict],
        *,
        system: str,
        max_tokens: int,
        response_schema: Optional[type] = None,
    ) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai package not installed. "
                "Run: uv sync --extra gemini"
            )
        client = genai.Client(api_key=self._api_key)
        # Convert OpenAI-style messages to Gemini Content objects
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
        )
        if system:
            config.system_instruction = system
        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = self._gemini_schema(response_schema)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""

    # ---- resolution ------------------------------------------------

    @classmethod
    def _resolve(
        cls,
        provider: Optional[str],
        model: Optional[str],
        api_key: Optional[str],
    ) -> tuple:
        if provider is None:
            has_openai = bool(os.environ.get("OPENAI_API_KEY"))
            has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
            if has_openai and has_gemini:
                raise RuntimeError(
                    "Both OPENAI_API_KEY and GEMINI_API_KEY are set. "
                    "Pass --provider openai|gemini to select one."
                )
            if has_openai:
                provider = "openai"
            elif has_gemini:
                provider = "gemini"
            else:
                raise RuntimeError(
                    "No API key found. Set OPENAI_API_KEY or GEMINI_API_KEY."
                )

        if provider == "openai":
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            resolved_model = model or cls.OPENAI_MODELS["default"]
        elif provider == "gemini":
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            resolved_model = model or cls.GEMINI_MODELS["default"]
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose 'openai' or 'gemini'."
            )

        return provider, resolved_model, resolved_key

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Create a client by auto-detecting the available API key."""
        return cls()
