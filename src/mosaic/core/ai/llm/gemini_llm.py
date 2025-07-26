from collections import Counter
from dataclasses import dataclass, field
import logging
import os
from typing import Optional, TypeVar, Union

from mosaic.core.ai.llm.base import BaseLLM, TokenCounter
from mosaic.core.common.utils.pydantic import merge_pydantic_models

from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    ContentListUnion,
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentConfigOrDict,
    GenerateContentResponse,
    GenerateImagesConfigOrDict,
    GenerateImagesResponse,
    GenerateVideosConfigOrDict,
    GenerateVideosResponse,
    GoogleSearch,
    ImageOrDict,
    Tool,
)
import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


async def _get_final_url(url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308) and "location" in response.headers:
                return response.headers["location"]
            return str(response.url)
    except Exception as e:
        logging.error(f"Error getting final URL for {url}: {e}")
        return url


CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

ResponseT = TypeVar("ResponseT")


API_KEY = os.getenv("GEMINI_API_KEY")


"""
Docs: https://ai.google.dev/gemini-api/docs/text-generation
"""


class GeminiTokenCounter(TokenCounter):
    """
    Token counter implementation for Gemini API responses.
    Extracts token usage information from Gemini response objects.
    """

    def __init__(self):
        self._token_count = Counter()

    def count_tokens(self, response: Union[GenerateContentResponse, GenerateImagesResponse, GenerateVideosResponse]) -> Union[GenerateContentResponse, GenerateImagesResponse, GenerateVideosResponse]:
        """
        Counts the tokens in the response.
        """
        if not isinstance(response, GenerateContentResponse):
            logging.getLogger(__name__).debug(f"Skipping token count for {type(response)} token count not yet supported for this response type.")
            return response

        try:
            if usage := response.usage_metadata:
                usage_dict = usage.model_dump(exclude_none=True)

                if "prompt_tokens_details" in usage_dict:
                    usage_dict.pop("prompt_tokens_details")
                if "cache_tokens_details" in usage_dict:
                    usage_dict.pop("cache_tokens_details")
                if "candidates_tokens_details" in usage_dict:
                    usage_dict.pop("candidates_tokens_details")

                self._token_count.update(usage_dict)

        except Exception as e:
            logging.getLogger(__name__).error(f"Error counting tokens: {e}")

        return response

    def get_token_count(self) -> Counter:
        return self._token_count

    def reset_token_count(self) -> None:
        self._token_count.clear()


@dataclass
class SearchSegment:
    """Model for a segment of text from a Gemini response with its confidence score."""

    text: str
    confidence: float


@dataclass
class GeminiSource:
    """Model for a source used in a Gemini response."""

    title: str
    url: str
    confidence: float
    segments: list[SearchSegment] = field(default_factory=list)
    summary: str = ""


@dataclass
class ExtractedSearchResponse:
    """Model for a complete Gemini response with answer and sources."""

    answer: str
    sources: list[GeminiSource] = field(default_factory=list)


class GeminiLLM(BaseLLM):
    """
    Gemini LLM
    """

    # Cache the search tool as a class variable for better performance
    _google_search_tool = Tool(
        google_search=GoogleSearch(),
    )

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize BaseLLM with Gemini-specific token counter
        super().__init__(token_counter=GeminiTokenCounter(), verbose=verbose, logger=logger)

        if client is None:
            client = genai.Client(api_key=API_KEY)

        self.__client = client

    @property
    def client(self) -> genai.Client:
        return self.__client

    @property
    def google_search_tool(self) -> Tool:
        """Get the cached Google search tool."""
        return self._google_search_tool

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_content(
        self,
        model: str = CHAT_MODEL,
        contents: Union[ContentListUnion, ContentListUnionDict, str] = [],
        config: Optional[GenerateContentConfigOrDict] = None,
        use_search: bool = True,
    ):
        if isinstance(contents, str):
            contents = [contents]

        # Always merge with search config
        if use_search:
            config = self.__default_search_config(config)

        try:
            self.logger.debug(f"Generating text with model={model}, contents_type={type(contents)} contents_count={len(contents) if isinstance(contents, list) else 1}")

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            self.logger.debug("Received response from Gemini API")
            response = await self._apply_middlewares(response)
            return response

        except Exception as e:
            self.exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_content_stream(  # noqa: C901
        self,
        model: str = CHAT_MODEL,
        contents: Union[ContentListUnion, ContentListUnionDict, str] = [],
        config: Optional[GenerateContentConfigOrDict] = None,
        use_search: bool = False,
    ):
        self.logger.debug(f"Generating text with model={model}, contents_type={type(contents)} contents_count={len(contents) if isinstance(contents, list) else 1}")

        try:
            if use_search:
                config = self.__default_search_config(config)

            stream = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            async for chunk in stream:
                chunk = await self._apply_middlewares(chunk)
                yield chunk

        except ClientError as e:
            self.exception(e, level=logging.ERROR)
            raise e

        except ServerError as e:
            self.exception(e, level=logging.ERROR)
            return
        except Exception as e:
            self.exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str = "veo-2.0-generate-001",
        image: Optional[ImageOrDict] = None,
        config: Optional[GenerateVideosConfigOrDict] = None,
    ):
        config_keys = config.model_dump(exclude_none=True).keys() if isinstance(config, BaseModel) else "None" if config is None else config.keys()
        self.logger.debug(f"Generating videos with model={model}, prompt_type={type(prompt)} image_type={type(image)} config_keys={config_keys}")
        try:
            response = await self.client.aio.models.generate_videos(
                model=model,
                prompt=prompt,
                image=image,
                config=config,
            )

            self.logger.debug("Received response from Gemini API")
            response = await self._apply_middlewares(response)
            return response

        except Exception as e:
            self.exception(e, level=logging.ERROR)
            raise e

    # TODO: Configure model typing
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
        retry=retry_if_not_exception_type((ServerError,)),
    )
    async def generate_images(
        self,
        prompt: str,
        model: str = "imagen-3.0-generate-002",
        config: Optional[GenerateImagesConfigOrDict] = None,
    ):
        self.logger.debug(
            f"Generating image with model={model}, prompt_type={type(prompt)} config_keys={config.model_dump(exclude_none=True).keys() if isinstance(config, BaseModel) else 'None' if config is None else config.keys()}"  # noqa: E501
        )

        try:
            response = await self.client.aio.models.generate_images(
                model=model,
                prompt=prompt,
                config=config,
            )

            self.logger.debug("Received response from Gemini API")
            response = await self._apply_middlewares(response)
            return response

        except Exception as e:
            self.exception(e, level=logging.ERROR)
            raise e

    def __default_search_config(self, config: Optional[GenerateContentConfigOrDict] = None) -> GenerateContentConfig:
        """
        Adds a Google Search tool to the config if it is not already present.
        """
        self.logger.debug("Adding Google Search tool to config")
        config_with_search = GenerateContentConfig(
            tools=[self.google_search_tool],
            response_modalities=["TEXT"],
        )

        if config is None:
            self.logger.debug("Config is None, returning config with search")
            return config_with_search

        self.logger.debug("Config is not None, merging config with search")
        config = config if isinstance(config, GenerateContentConfig) else GenerateContentConfig(**config)  # type: ignore
        return merge_pydantic_models(config, config_with_search)

    def get_token_usage(self) -> dict[str, int]:
        """
        Get current token usage statistics.

        Returns:
            dict: Token usage statistics including prompt_tokens, completion_tokens, total_tokens
        """
        return dict(self.token_count)

    @classmethod
    async def extract_response_sources_and_answer(cls, response: GenerateContentResponse) -> ExtractedSearchResponse:  # noqa: C901
        """
        Extract answer and sources with their supporting segments from a Gemini response object.
        Also fetches and summarizes content from URLs using UrlContext.

        Args:
            response: The Gemini GenerateContentResponse object

        Returns:
            ExtractedSearchResponse containing answer and sources with summaries
        """
        try:
            # Extract the answer text from the first candidate
            answer = response.text
            # If the answer is empty, try to extract it from the first candidate
            if not answer and response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts or []:
                    if part.text:
                        answer = part.text
                        break

            # Early return if no grounding metadata
            if not response.candidates or not response.candidates[0].grounding_metadata:
                return ExtractedSearchResponse(answer=answer or "", sources=[])

            grounding = response.candidates[0].grounding_metadata
            sources: list[GeminiSource] = []
            source_map: dict[str, int] = {}  # url -> source index

            # First pass: collect all unique sources
            if grounding.grounding_chunks:
                for chunk in grounding.grounding_chunks:
                    if chunk.web and chunk.web.uri and chunk.web.uri not in source_map:
                        source_map[chunk.web.uri] = len(sources)
                        sources.append(
                            GeminiSource(
                                title=chunk.web.title if chunk.web.title else "Unknown Source",
                                url=await _get_final_url(chunk.web.uri),
                                confidence=0.0,
                                segments=[],
                            )
                        )

            # Second pass: process grounding supports to add segments
            if grounding.grounding_supports:
                for support in grounding.grounding_supports:
                    if support.segment and support.segment.text and support.grounding_chunk_indices and support.confidence_scores:
                        segment_text = support.segment.text
                        for idx, score in zip(
                            support.grounding_chunk_indices,
                            support.confidence_scores,
                        ):
                            if idx < len(sources):
                                sources[idx].confidence = max(sources[idx].confidence, float(score))
                                # Add segment to source
                                sources[idx].segments.append(
                                    SearchSegment(
                                        text=segment_text,
                                        confidence=float(score),
                                    )
                                )

            return ExtractedSearchResponse(answer=answer or "", sources=sources)

        except Exception as e:
            logging.error(f"Error extracting Gemini response: {e}")
            return ExtractedSearchResponse(answer="", sources=[])
