from abc import ABC, abstractmethod
from collections import Counter
from inspect import iscoroutinefunction
import logging
from typing import Callable, Optional, TypeVar

from mosaic.core.common.logger_mixin import LoggerMixin

T = TypeVar("T")


class TokenCounter(ABC):
    """
    Abstract base class for token counting strategies.
    Allows different LLM providers to implement their own token counting logic.

    Example usage with different providers:

    class AnthropicTokenCounter(TokenCounter):
        def count_tokens(self, response: AnthropicResponse) -> AnthropicResponse:
            # Extract token usage from Anthropic's response format
            if hasattr(response, 'usage'):
                self._token_count.update({
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                })
            return response

    class CohereTokenCounter(TokenCounter):
        def count_tokens(self, response: CohereResponse) -> CohereResponse:
            # Extract token usage from Cohere's response format
            if hasattr(response, 'meta') and hasattr(response.meta, 'billed_units'):
                self._token_count.update({
                    'prompt_tokens': response.meta.billed_units.input_tokens,
                    'completion_tokens': response.meta.billed_units.output_tokens,
                    'total_tokens': response.meta.billed_units.input_tokens + response.meta.billed_units.output_tokens
                })
            return response
    """

    @abstractmethod
    def count_tokens(self, response: T) -> T:
        """
        Count tokens in the response and update internal counters.

        Args:
            response (T): The response to count tokens from.

        Returns:
            T: The response (unchanged, for chaining).
        """
        pass

    @abstractmethod
    def get_token_count(self) -> Counter:
        """
        Get the current token count statistics.

        Returns:
            Counter: Token count statistics.
        """
        pass

    @abstractmethod
    def reset_token_count(self) -> None:
        """
        Reset the token count statistics.
        """
        pass


class DefaultTokenCounter(TokenCounter):
    """
    Default token counter implementation that does nothing.
    Used when no specific token counting is needed.
    """

    def __init__(self):
        self._token_count = Counter()

    def count_tokens(self, response: T) -> T:
        """Default implementation does nothing."""
        return response

    def get_token_count(self) -> Counter:
        return self._token_count

    def reset_token_count(self) -> None:
        self._token_count.clear()


class BaseLLM(LoggerMixin, ABC):
    def __init__(self, token_counter: Optional[TokenCounter] = None, verbose: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self._token_counter = token_counter or DefaultTokenCounter()
        self._verbose = verbose
        self._middlewares: list[Callable] = []
        self._initialize_default_middlewares()
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    @property
    def token_count(self) -> Counter:
        return self._token_counter.get_token_count()

    @property
    def verbose(self) -> bool:
        return self._verbose

    def _initialize_default_middlewares(self):
        """
        Initializes the default middlewares.
        """
        self._middlewares.append(self._token_counter.count_tokens)
        self.logger.debug("Initialized default middlewares")

    def add_middleware(self, middleware: Callable[[T], T]):
        """
        Adds a custom middleware to the agent.

        Args:
            middleware (Callable): The middleware to add.
        """
        self._middlewares.append(middleware)
        self.logger.debug(f"Added middleware: {middleware.__name__}")

    async def _apply_middlewares(self, response: T) -> T:
        """
        Applies the middlewares to the response.

        Args:
            response (T): The response to apply the middlewares to.

        Returns:
            T: The response with the middlewares applied.
        """
        self.logger.debug(f"Applying middlewares from {self.__class__.__name__}")
        for middleware in self._middlewares:
            try:
                if iscoroutinefunction(middleware):
                    response = await middleware(response)
                else:
                    response = middleware(response)
            except Exception as e:
                self.exception(e, level=logging.ERROR)
        self.logger.debug(f"Finished applying middlewares from {self.__class__.__name__}")
        return response

    def reset_token_count(self) -> None:
        """
        Reset the token count statistics.
        """
        self._token_counter.reset_token_count()
        self.logger.debug("Reset token count")
