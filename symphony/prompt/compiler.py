"""Prompt Compiler — assembles context-aware prompts with optional token budgets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Token estimator (tiktoken-based when available, chars/4 fallback)
# ------------------------------------------------------------------

def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate token count using tiktoken if available, else chars/4.

    When *model* is provided, ``encoding_for_model`` resolves the exact
    encoding (e.g. o200k_base for gpt-4o-class models, cl100k_base for
    older ones). Without a model name, o200k_base is used — the encoding
    for all current-generation OpenAI models and a reasonable approximation
    for Gemini, which tokenises at a similar rate (~4 chars/token).
    """
    try:
        import tiktoken
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("o200k_base")
        else:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


# ------------------------------------------------------------------
# Context blocks
# ------------------------------------------------------------------

@dataclass
class ContextBlock:
    """A named block of dynamic context to inject into the prompt."""

    name: str
    content: str
    priority: int = 0  # higher = more important, kept first when budgeting
    token_count: int = 0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


# ------------------------------------------------------------------
# System prompt (constant, small)
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Symphony, an expert web-application reliability engineer.
Your job: given a goal, evidence from the running application, and
prior failures, produce a precise plan or code patch that moves the
application toward passing all acceptance criteria.

Rules:
- Only emit structured Flow DSL actions or code patches. Never emit raw scripts.
- Always justify changes with evidence from failures and traces.
- Prefer minimal, targeted fixes over broad refactors.
- When uncertain, request more evidence rather than guessing.
"""


# ------------------------------------------------------------------
# Prompt Compiler
# ------------------------------------------------------------------

@dataclass
class PromptCompiler:
    """Assembles prompts from system prompt + dynamic context blocks.

    When *token_budget* is None (default), all blocks are included
    without truncation. Pass an integer to enable budget enforcement
    with priority-based truncation.

    Pass *model* to use the correct tiktoken encoding for the target
    model when estimating token counts (e.g. "gpt-4o-mini"). Omit it
    to use the o200k_base default, which is accurate for all current
    OpenAI models and a close approximation for Gemini.

    For accurate Gemini token counts before a request, use
    ``LLMClient.count_tokens()`` — it calls the native
    ``client.models.count_tokens`` endpoint rather than estimating locally.
    """

    token_budget: Optional[int] = None
    system_prompt: str = SYSTEM_PROMPT
    model: Optional[str] = None
    _system_tokens: int = field(init=False, default=0)

    def __post_init__(self):
        self._system_tokens = estimate_tokens(self.system_prompt, self.model)

    @property
    def available_budget(self) -> Optional[int]:
        if self.token_budget is None:
            return None
        return self.token_budget - self._system_tokens

    def compile(
        self,
        goal: str,
        blocks: List[ContextBlock],
        *,
        user_instruction: str = "",
    ) -> List[dict]:
        """Return a messages list (system + user).

        If *token_budget* is set, blocks are sorted by priority (desc)
        and included until the budget is exhausted. Otherwise all
        blocks are included in priority order.
        """
        sorted_blocks = sorted(blocks, key=lambda b: b.priority, reverse=True)

        goal_section = f"## Goal\n{goal}\n"
        included_sections: List[str] = [goal_section]

        if user_instruction:
            included_sections.append(f"## Instructions\n{user_instruction}\n")

        if self.token_budget is None:
            for block in sorted_blocks:
                included_sections.append(f"## {block.name}\n{block.content}\n")
        else:
            remaining = self.available_budget or 0
            remaining -= estimate_tokens(goal_section, self.model)
            if user_instruction:
                remaining -= estimate_tokens(included_sections[-1], self.model)

            for block in sorted_blocks:
                if remaining <= 0:
                    break
                if block.token_count <= remaining:
                    section = f"## {block.name}\n{block.content}\n"
                    included_sections.append(section)
                    remaining -= block.token_count
                else:
                    chars_available = remaining * 4
                    truncated = block.content[:chars_available]
                    included_sections.append(
                        f"## {block.name} (truncated)\n{truncated}\n"
                    )
                    remaining = 0

        user_content = "\n".join(included_sections)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def compile_with_usage(
        self,
        goal: str,
        blocks: List[ContextBlock],
        **kwargs: Any,
    ) -> tuple:
        """Compile and return (messages, usage_info)."""
        messages = self.compile(goal, blocks, **kwargs)
        total = sum(estimate_tokens(m["content"], self.model) for m in messages)
        usage: dict = {"total_tokens": total}
        if self.token_budget is not None:
            usage["budget"] = self.token_budget
            usage["remaining"] = self.token_budget - total
        return messages, usage
