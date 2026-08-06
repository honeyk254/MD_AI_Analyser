"""LLM client abstraction.

The reporting layer talks to this interface only, so the platform runs
end-to-end with no API key (see ``narrator.TemplateNarrator``) and a hosted
model is a drop-in swap rather than a dependency.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("md_ai_analyzer")

DEFAULT_MODEL = "claude-3-5-haiku-20241022"

# USD per million tokens, by model id. Used for the per-report cost estimate in
# the audit trail; update alongside the provider's published pricing.
PRICING_USD_PER_MTOK: Dict[str, "tuple[float, float]"] = {
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}
FALLBACK_PRICING = (3.00, 15.00)


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMReply:
    """One model turn: free text, tool calls, and token usage."""

    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw_content: Any = None


class LLMClient(ABC):
    """Minimal tool-calling chat interface."""

    model: str = DEFAULT_MODEL

    @abstractmethod
    def reply(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMReply:
        """Produce the next assistant turn.

        ``messages`` uses the Anthropic content-block format; a client for another
        provider is responsible for translating it.
        """

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = PRICING_USD_PER_MTOK.get(self.model, FALLBACK_PRICING)
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


class AnthropicClient(LLMClient):
    """Anthropic Messages API implementation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        timeout_seconds: float = 60.0,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'anthropic' package is required for hosted reporting; "
                "install md-platform[llm] or use the offline template narrator."
            ) from exc

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout_seconds)

    def reply(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMReply:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        return LLMReply(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_content=[block.model_dump() for block in response.content],
        )


def parse_sections(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract the ``sections`` array from a model response.

    Models wrap JSON in prose or fences often enough that a strict
    ``json.loads`` would reject otherwise valid reports, so the first balanced
    JSON object in the text is used.
    """
    if not text:
        return None

    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    sections = payload.get("sections")
                    if isinstance(sections, list) and sections:
                        return [
                            {
                                "heading": str(s.get("heading", "Untitled")),
                                "body": str(s.get("body", "")),
                            }
                            for s in sections
                            if isinstance(s, dict)
                        ]
                    break
        start = text.find("{", start + 1)
    return None
