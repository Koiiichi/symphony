"""Prompt Compiler — assembles context-aware prompts within token budgets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Token estimator (tiktoken-based when available, char fallback)
# ------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken if available, else chars/4."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
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
    priority: int = 0  # higher = more important, kept first
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
    """Assembles prompts from system prompt + dynamic context blocks,
    respecting a token budget with priority-based truncation."""

    token_budget: int = 16_000
    system_prompt: str = SYSTEM_PROMPT
    _system_tokens: int = field(init=False, default=0)

    def __post_init__(self):
        self._system_tokens = estimate_tokens(self.system_prompt)

    @property
    def available_budget(self) -> int:
        return self.token_budget - self._system_tokens

    def compile(
        self,
        goal: str,
        blocks: list[ContextBlock],
        *,
        user_instruction: str = "",
    ) -> list[dict[str, str]]:
        """Return a messages list (system + user) fitting within budget.

        Blocks are sorted by priority (desc) and included until the budget
        is exhausted. Lower-priority blocks are truncated or dropped.
        """
        sorted_blocks = sorted(blocks, key=lambda b: b.priority, reverse=True)

        remaining = self.available_budget
        goal_section = f"## Goal\n{goal}\n"
        remaining -= estimate_tokens(goal_section)

        included_sections: list[str] = [goal_section]

        if user_instruction:
            instr_section = f"## Instructions\n{user_instruction}\n"
            instr_tokens = estimate_tokens(instr_section)
            if instr_tokens <= remaining:
                included_sections.append(instr_section)
                remaining -= instr_tokens

        for block in sorted_blocks:
            if remaining <= 0:
                break
            if block.token_count <= remaining:
                section = f"## {block.name}\n{block.content}\n"
                included_sections.append(section)
                remaining -= block.token_count
            else:
                # Truncate block to fit
                chars_available = remaining * 4  # rough chars estimate
                truncated = block.content[:chars_available]
                section = f"## {block.name} (truncated)\n{truncated}\n"
                included_sections.append(section)
                remaining = 0

        user_content = "\n".join(included_sections)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def compile_with_usage(
        self,
        goal: str,
        blocks: list[ContextBlock],
        **kwargs: Any,
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        """Compile and return (messages, usage_info)."""
        messages = self.compile(goal, blocks, **kwargs)
        total = sum(estimate_tokens(m["content"]) for m in messages)
        return messages, {
            "total_tokens": total,
            "budget": self.token_budget,
            "remaining": self.token_budget - total,
        }
