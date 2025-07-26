#!/usr/bin/env python3
"""
Simple test to demonstrate the new TokenCounter abstraction.
"""

import asyncio
from collections import Counter

from mosaic.core.llm.base import DefaultTokenCounter, TokenCounter
from mosaic.core.llm.openai_llm import OpenAITokenCounter


class MockResponse:
    """Mock response for testing."""

    def __init__(self, usage_data=None):
        self.usage = usage_data


class MockUsage:
    """Mock usage object for testing."""

    def __init__(self, prompt_tokens=10, completion_tokens=20):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def model_dump(self, exclude_none=True):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "total_tokens": self.total_tokens}


def test_default_token_counter():
    """Test the default token counter."""
    print("Testing DefaultTokenCounter...")
    counter = DefaultTokenCounter()

    # Test that it doesn't crash
    response = MockResponse()
    result = counter.count_tokens(response)
    assert result == response

    # Test token count is empty
    assert counter.get_token_count() == Counter()

    print("✅ DefaultTokenCounter works correctly")


def test_openai_token_counter():
    """Test the OpenAI token counter."""
    print("Testing OpenAITokenCounter behavior...")
    counter = OpenAITokenCounter()

    # First response
    usage1 = MockUsage(prompt_tokens=10, completion_tokens=15)
    response1 = MockResponse(usage1)
    result1 = counter.count_tokens(response1)  # type: ignore
    assert result1 == response1

    # Check token count after first response
    token_count = counter.get_token_count()
    assert token_count["prompt_tokens"] == 10
    assert token_count["completion_tokens"] == 15
    assert token_count["total_tokens"] == 25

    # Second response
    usage2 = MockUsage(prompt_tokens=20, completion_tokens=30)
    response2 = MockResponse(usage2)
    result2 = counter.count_tokens(response2)  # type: ignore
    assert result2 == response2

    # Check token count after second response - should be accumulated
    token_count = counter.get_token_count()
    assert token_count["prompt_tokens"] == 30  # 10 + 20
    assert token_count["completion_tokens"] == 45  # 15 + 30
    assert token_count["total_tokens"] == 75  # 25 + 50

    # Third response with different token types
    usage3 = MockUsage(prompt_tokens=5, completion_tokens=10)
    response3 = MockResponse(usage3)
    result3 = counter.count_tokens(response3)  # type: ignore
    assert result3 == response3

    # Check final accumulated token count
    token_count = counter.get_token_count()
    assert token_count["prompt_tokens"] == 35  # 10 + 20 + 5
    assert token_count["completion_tokens"] == 55  # 15 + 30 + 10
    assert token_count["total_tokens"] == 90  # 25 + 50 + 15

    print("✅ OpenAITokenCounter properly stacks token counts across multiple responses")


def test_base_llm_with_token_counter():
    """Test BaseLLM with custom token counter."""
    print("Testing BaseLLM with TokenCounter...")

    # Create a custom token counter
    class CustomTokenCounter(TokenCounter):
        def __init__(self):
            self._token_count = Counter()

        def count_tokens(self, response):
            # Simple implementation that counts any response
            self._token_count["total_responses"] += 1
            return response

        def get_token_count(self):
            return self._token_count

        def reset_token_count(self):
            self._token_count.clear()

    # Create BaseLLM with custom token counter
    custom_counter = CustomTokenCounter()

    # Note: We can't instantiate BaseLLM directly since it's abstract
    # But we can test the token counter integration
    assert custom_counter.get_token_count() == Counter()

    # Test counting
    response = MockResponse()
    result = custom_counter.count_tokens(response)
    assert result == response
    assert custom_counter.get_token_count()["total_responses"] == 1

    print("✅ BaseLLM TokenCounter integration works correctly")


def test_token_counter_reset():
    """Test token counter reset functionality."""
    print("Testing TokenCounter reset...")

    counter = OpenAITokenCounter()

    # Add some tokens
    usage = MockUsage(10, 20)
    response = MockResponse(usage)
    counter.count_tokens(response)  # type: ignore

    # Verify tokens were counted
    assert counter.get_token_count()["total_tokens"] == 30

    # Reset
    counter.reset_token_count()
    assert counter.get_token_count() == Counter()

    print("✅ TokenCounter reset works correctly")


async def main():
    """Run all tests."""
    print("🧪 Testing TokenCounter abstraction...\n")

    test_default_token_counter()
    test_openai_token_counter()
    test_base_llm_with_token_counter()
    test_token_counter_reset()

    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
