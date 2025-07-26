from __future__ import annotations

from collections import Counter
import json
import logging
import os
from typing import (
    cast,
    Iterable,
    List,
    Literal,
    Optional,
    overload,
    Type,
    TypeVar,
    Union,
)

from mosaic.core.ai.llm.base import BaseLLM, S, TokenCounter

import httpx
from httpx import TimeoutException
import numpy as np
from openai import AsyncOpenAI, NOT_GIVEN, NotGiven, Timeout
from openai._types import Body, Headers, Query
from openai.types import ChatModel, ImageModel, Metadata, Reasoning, ResponsesModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
)
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from openai.types.chat.completion_create_params import (
    WebSearchOptions,
)
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from openai.types.images_response import ImagesResponse
from openai.types.responses import (
    Response,
    ResponseIncludable,
    ResponseInputParam,
    ResponseTextConfigParam,
    ToolParam,
)
from openai.types.responses.response_create_params import ToolChoice
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)

Completion = Union[ChatCompletion, ParsedChatCompletion, ChatCompletionChunk]
CompletionT = TypeVar("CompletionT")


CHAT_MODEL: ChatModel = cast(ChatModel, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
REASONING_MODEL: ChatModel = cast(ChatModel, os.getenv("OPENAI_REASONING_MODEL", "o3-mini"))
IMAGE_MODEL: Union[str, ImageModel] = cast(Union[str, ImageModel], os.getenv("OPENAI_IMAGE_MODEL", "gpt-4o-mini"))


class OpenAITokenCounter(TokenCounter):
    """
    Token counter implementation for OpenAI API responses.
    Extracts token usage information from OpenAI completion objects.
    """

    def __init__(self):
        self._token_count = Counter()

    def count_tokens(self, completion: Completion | Response) -> Completion | Response:
        """
        Counts the tokens in the completion.

        Args:
            completion (Completion | Response): The completion to count the tokens of.

        Returns:
            Completion | Response: The completion with the token count.
        """
        try:
            if usage := completion.usage:
                usage_dict = usage.model_dump(exclude_none=True)
                completion_details = usage_dict.pop("completion_tokens_details", None)
                prompt_details = usage_dict.pop("prompt_tokens_details", None)
                usage_dict.pop("search_context_size", None)

                self._token_count.update(usage_dict)
                if completion_details:
                    self._token_count.update(completion_details)
                if prompt_details:
                    self._token_count.update(prompt_details)
        except Exception as e:
            # Log error but don't fail the request
            logging.getLogger(__name__).error(f"Error counting tokens: {e}")

        return completion

    def get_token_count(self) -> Counter:
        return self._token_count

    def reset_token_count(self) -> None:
        self._token_count.clear()


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize BaseLLM with OpenAI-specific token counter
        super().__init__(token_counter=OpenAITokenCounter(), verbose=verbose, logger=logger)

        if client is None:
            client = AsyncOpenAI()

        self.__client = client

    @property
    def client(self) -> AsyncOpenAI:
        return self.__client

    @overload
    async def generate(self, prompt: str, output_format: Optional[Type[S]] = None) -> Union[str, None]: ...

    @overload
    async def generate(self, prompt: str, output_format: Type[S]) -> Union[S, None]: ...

    async def generate(self, prompt: str, output_format: Optional[Type[S]] = None) -> Union[str, S, None]:
        if output_format is None:
            response = await self.completion(messages=[{"role": "user", "content": prompt}])
            if content := response.get("content"):
                return str(content)
        else:
            response = await self.structured_completion(output_format, messages=[{"role": "user", "content": prompt}])
            if response.parsed:
                return response.parsed

        return None

    @overload
    async def completion(
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: Literal[1] | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> ChatCompletionMessageParam: ...

    @overload
    async def completion(
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int = 2,
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> List[ChatCompletionMessageParam]: ...

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def completion(  # noqa: C901
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int | NotGiven | None = NOT_GIVEN,
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> Union[ChatCompletionMessageParam, List[ChatCompletionMessageParam]]:
        """
        Gets a completion from the OpenAI API.

        Args:
            model (ChatModel): The model to use for completions.
            messages (list[ChatCompletionMessageParam]): The messages to send to the API.
            n (int | NotGiven): Number of completions to generate. If not provided, returns a single completion.
                               If provided, returns a list of completions.

        Returns:
            Union[ChatCompletionMessageParam, List[ChatCompletionMessageParam]]:
                If n is not provided or is 1, returns a single completion message.
                If n > 1, returns a list of completion messages.
        """
        self.logger.debug(f"Starting completion with model={model}, messages_count={len(messages)}, n={n}")

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, NotGiven) else REASONING_MODEL

        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                metadata=metadata,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )
            self.logger.debug("Received completion from OpenAI API")
            completion = await self._apply_middlewares(completion)

            if not completion.choices:
                self.logger.warning("No choices returned in completion")
                raise ValueError("No choices returned in completion")

            # Return all choices if n > 1, otherwise return the first choice
            if isinstance(n, int) and n > 1:
                return [choice.message for choice in completion.choices]  # type: ignore
            return completion.choices[0].message  # type: ignore

        except (TimeoutException, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            self.exception(e, level=logging.ERROR)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def response(
        self,
        input: str | ResponseInputParam,
        model: ResponsesModel = CHAT_MODEL,
        include: List[ResponseIncludable] | NotGiven | None = NOT_GIVEN,
        instructions: str | NotGiven | None = NOT_GIVEN,
        max_output_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: Metadata | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven | None = NOT_GIVEN,
        previous_response_id: str | NotGiven | None = NOT_GIVEN,
        reasoning: Reasoning | NotGiven | None = None,
        service_tier: NotGiven | Literal["auto", "default", "flex"] | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        text: ResponseTextConfigParam | NotGiven = NOT_GIVEN,
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        truncation: NotGiven | Literal["auto", "disabled"] | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | Timeout | NotGiven | None = NOT_GIVEN,
    ) -> Response:
        self.logger.debug(f"Starting response with model={model}, input_type={type(input)}")

        try:
            response: Response = await self.client.responses.create(
                input=input,
                model=model,
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                metadata=metadata,
                parallel_tool_calls=parallel_tool_calls,
                previous_response_id=previous_response_id,
                reasoning=reasoning,  # type: ignore
                stream=False,
                service_tier=service_tier,
                store=store,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                top_p=top_p,
                truncation=truncation,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            self.logger.debug("Received response from OpenAI API")

            return await self._apply_middlewares(response)

        except (TimeoutException, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            self.exception(e, level=logging.ERROR)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def stream(  # noqa: C901
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        n: int | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ):
        """
        Gets a streaming completion from the OpenAI API.

        Args:
            model (ChatModel): The model to use for completions.

        Returns:
            A stream of completion chunks with middlewares applied.
        """
        self.logger.debug(f"Starting streaming completion with model={model}, messages_count={len(messages)}")

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, NotGiven) else REASONING_MODEL

        try:
            # Ensure stream_options includes usage information
            if isinstance(stream_options, NotGiven):
                stream_options = {"include_usage": True}
            elif stream_options is None:
                stream_options = {"include_usage": True}
            elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
                stream_options["include_usage"] = True

            # Create a stream manager

            async def stream_manager():
                accumulated_content = ""
                async with await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    max_completion_tokens=max_completion_tokens,
                    max_tokens=max_tokens,
                    metadata=metadata,
                    n=n,
                    parallel_tool_calls=parallel_tool_calls,
                    presence_penalty=presence_penalty,
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    service_tier=service_tier,
                    stop=stop,
                    store=store,
                    stream=True,
                    stream_options=stream_options,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_logprobs=top_logprobs,
                    top_p=top_p,
                    user=user,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                ) as stream:
                    async for chunk in stream:
                        await self._apply_middlewares(chunk)
                        if isinstance(chunk, ChatCompletionChunk) and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                            accumulated_content += chunk.choices[0].delta.content
                            yield accumulated_content, "delta"
                    yield accumulated_content, "done"

            return stream_manager()
        except (TimeoutException, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            self.exception(e, level=logging.ERROR)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def structured_stream(  # noqa: C901
        self,
        response_format: Type[T],
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        n: int | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ):
        """
        Gets a streaming structured completion from the OpenAI API.

        See examples usage:https://platform.openai.com/docs/guides/structured-outputs

        Args:
            response_format (Type[T]): The expected response format.
            model (ChatModel): The model to use for completions.
            messages (list[ChatCompletionMessageParam]): The messages to send to the API.
            stream_options (ChatCompletionStreamOptionsParam): Options for streaming.

        Returns:
            A stream of structured completion chunks with middlewares applied.
        """
        self.logger.debug(f"Starting stream_structured_completion with model={model}, response_format={response_format.__name__}, messages_count={len(messages)}")

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, NotGiven) else REASONING_MODEL

        try:
            # Ensure stream_options includes usage information
            if isinstance(stream_options, NotGiven) or stream_options is None:
                stream_options = {"include_usage": True}
            elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
                stream_options["include_usage"] = True

            # Create an async generator that handles the stream context manager
            async def stream_generator():
                async with self.client.beta.chat.completions.stream(
                    model=model,
                    messages=messages,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    logprobs=logprobs,
                    max_completion_tokens=max_completion_tokens,
                    max_tokens=max_tokens,
                    metadata=metadata,
                    n=n,
                    parallel_tool_calls=parallel_tool_calls,
                    presence_penalty=presence_penalty,
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    service_tier=service_tier,
                    stop=stop,
                    store=store,
                    stream_options=stream_options,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_logprobs=top_logprobs,
                    top_p=top_p,
                    user=user,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    response_format=response_format,
                ) as stream:
                    # Process each chunk in the stream
                    async for event in stream:
                        # Apply middlewares to the accumulated completion if it's the final chunk
                        if event.type == "chunk":
                            await self._apply_middlewares(event.chunk)  # type: ignore

                        # Yield the original chunk
                        yield event

                    yield stream.get_final_completion()

            # Return the stream generator
            return stream_generator()

        except (TimeoutException, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            self.exception(e, level=logging.ERROR)
            raise

    @overload
    async def structured_completion(
        self,
        response_format: Type[T],
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: Literal[1] | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        web_search_options: WebSearchOptions | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> ParsedChatCompletionMessage[T]: ...

    @overload
    async def structured_completion(
        self,
        response_format: Type[T],
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int = 2,
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        web_search_options: WebSearchOptions | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> List[ParsedChatCompletionMessage[T]]: ...

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def structured_completion(  # noqa: C901
        self,
        response_format: Type[T] | NotGiven = NOT_GIVEN,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int | NotGiven | None = NOT_GIVEN,
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        web_search_options: WebSearchOptions | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> Union[ParsedChatCompletionMessage[T], List[ParsedChatCompletionMessage[T]]]:
        """
        Gets a structured completion from the OpenAI API.

        Args:
            response_format (Type[T]): The expected response format.
            model (ChatModel): The model to use for completions.
            messages (list[ChatCompletionMessageParam]): The messages to send to the API.
            n (int | NotGiven): Number of completions to generate. If not provided, returns a single completion.
                               If provided, returns a list of completions.

        Returns:
            Union[ParsedChatCompletionMessage[T], List[ParsedChatCompletionMessage[T]]]:
                If n is not provided or is 1, returns a single parsed completion message.
                If n > 1, returns a list of parsed completion messages.
        """
        self.logger.debug(
            f"Starting structured_completion with model={model}, response_format={response_format.__name__ if not isinstance(response_format, NotGiven) else None}, messages_count={len(messages)}, n={n}"  # noqa: E501
        )

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, NotGiven) else REASONING_MODEL

        try:
            completion = await self.client.beta.chat.completions.parse(
                response_format=response_format,
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                web_search_options=web_search_options,
                max_tokens=max_tokens,
                metadata=metadata,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )
            self.logger.debug("Received structured completion from OpenAI API")
            completion = await self._apply_middlewares(completion)

            if not completion.choices:
                self.logger.warning("No choices returned in structured completion")
                raise ValueError("No choices returned in structured completion")

            # Return all choices if n > 1, otherwise return the first choice
            if isinstance(n, int) and n > 1:
                return [choice.message for choice in completion.choices]
            return completion.choices[0].message

        except (TimeoutException, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            self.exception(e, level=logging.ERROR)
            raise

    @retry(stop=stop_after_attempt(5), reraise=True)
    async def generate_image(
        self,
        prompt: str,
        model: Union[str, ImageModel, None] | NotGiven = IMAGE_MODEL,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        quality: Literal["standard", "hd"] | NotGiven = NOT_GIVEN,
        response_format: Optional[Literal["url", "b64_json"]] | NotGiven = NOT_GIVEN,
        size: (Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] | NotGiven) = "256x256",
        style: Optional[Literal["vivid", "natural"]] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ImagesResponse:
        """
        Generates an image using the OpenAI DALL-E model.

        Args:
            prompt (str): The prompt to generate the image.
            model (str): The model to use for image generation.
            size (str): The size of the image. Defaults to "1024x1024".
            quality (str): The quality of the image. Defaults to "standard".
            n (int): The number of images to generate. Defaults to 1.

        Returns:
            List[str]: A list of URLs to the generated images.
        """
        self.logger.debug(f"Starting image generation with prompt='{prompt}', model='{model}', size='{size}', quality='{quality}', n={n}")
        try:
            response = await self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
                response_format=response_format,
                style=style,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )
            self.logger.debug("Received response from image generation API")
            return response
        except (TimeoutException, httpx.RequestError, json.JSONDecodeError, Exception) as e:
            self.exception(e, level=logging.ERROR)
            raise

    def confidence_score(self, top_logprob: TopLogprob) -> float:
        """
        Calculates the confidence score for a single token in the completion.

        Args:
            logprobs (ChoiceLogprobs): The log probabilities for the completion.

        Returns:
            float: The confidence score.
        """
        _, _, confidence_score = self.token_abstraction(top_logprob=top_logprob)
        return confidence_score

    def token_abstraction(self, top_logprob: TopLogprob):
        return (
            top_logprob.token,
            top_logprob.logprob,
            np.round(np.exp(top_logprob.logprob) * 100, 2),
        )

    def perplexity(self, choice_logprobs: ChoiceLogprobs):
        """
        Calculates how perplexed the llm is based on the logprobs.

        When looking to assess the model's confidence in a result, it can be useful to calculate perplexity,
        which is a measure of the uncertainty

        Args:
            logprobs (ChoiceLogprobs): The log probabilities for the completion.

        Returns:
            float: The perplexity score.
        """

        # TODO: Explore normalization of perplexity score to that values can 1 to infinity depending on the number of tokens. Ensuring that the score is
        # normalized properly in order take the appropriate action requires a better understanding of the theoretical limits

        if choice_logprobs.content is None:
            return 0.0
        logprobs = np.array([lp.logprob for lp in choice_logprobs.content])
        perplexity_score: float = np.exp(-np.mean(logprobs))
        return perplexity_score
