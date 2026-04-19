"""Unit tests for Prompt Compiler and token budget management."""

import pytest

from symphony.prompt.compiler import (
    ContextBlock,
    PromptCompiler,
    estimate_tokens,
    SYSTEM_PROMPT,
)


class TestEstimateTokens:
    def test_non_negative(self):
        assert estimate_tokens("") >= 0

    def test_longer_text_more_tokens(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short

    def test_empty_string(self):
        assert estimate_tokens("") == 0


class TestContextBlock:
    def test_auto_token_count(self):
        block = ContextBlock(name="test", content="some content here")
        assert block.token_count > 0

    def test_explicit_token_count(self):
        block = ContextBlock(name="test", content="x", token_count=42)
        assert block.token_count == 42

    def test_priority_default(self):
        block = ContextBlock(name="test", content="x")
        assert block.priority == 0


class TestPromptCompiler:
    def test_basic_compile(self):
        compiler = PromptCompiler(token_budget=16_000)
        messages = compiler.compile("fix the login page", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "fix the login page" in messages[1]["content"]

    def test_system_prompt_included(self):
        compiler = PromptCompiler()
        messages = compiler.compile("goal", [])
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_blocks_included_by_priority(self):
        compiler = PromptCompiler(token_budget=16_000)
        blocks = [
            ContextBlock(name="Low", content="low priority", priority=1),
            ContextBlock(name="High", content="high priority", priority=10),
        ]
        messages = compiler.compile("goal", blocks)
        user_content = messages[1]["content"]
        # High priority should appear before low priority
        assert user_content.index("High") < user_content.index("Low")

    def test_budget_truncation(self):
        # Very tight budget
        compiler = PromptCompiler(token_budget=200)
        big_block = ContextBlock(
            name="BigBlock",
            content="x " * 5000,
            priority=1,
        )
        messages = compiler.compile("goal", [big_block])
        # Should still produce valid output
        assert len(messages) == 2
        # Content should be truncated
        total_chars = sum(len(m["content"]) for m in messages)
        assert total_chars < len("x " * 5000)

    def test_budget_drops_low_priority(self):
        compiler = PromptCompiler(token_budget=300)
        blocks = [
            ContextBlock(name="Important", content="critical info", priority=100),
            ContextBlock(name="Unimportant", content="y " * 3000, priority=1),
        ]
        messages = compiler.compile("goal", blocks)
        user_content = messages[1]["content"]
        assert "Important" in user_content

    def test_user_instruction_included(self):
        compiler = PromptCompiler(token_budget=16_000)
        messages = compiler.compile(
            "goal", [], user_instruction="Be concise"
        )
        assert "Be concise" in messages[1]["content"]

    def test_compile_with_usage(self):
        compiler = PromptCompiler(token_budget=16_000)
        messages, usage = compiler.compile_with_usage("goal", [])
        assert "total_tokens" in usage
        assert "budget" in usage
        assert "remaining" in usage
        assert usage["budget"] == 16_000
        assert usage["remaining"] > 0

    def test_available_budget(self):
        compiler = PromptCompiler(token_budget=16_000)
        assert compiler.available_budget < 16_000  # system prompt takes some
        assert compiler.available_budget > 0

    def test_custom_system_prompt(self):
        compiler = PromptCompiler(
            token_budget=16_000,
            system_prompt="You are a test bot."
        )
        messages = compiler.compile("goal", [])
        assert messages[0]["content"] == "You are a test bot."
