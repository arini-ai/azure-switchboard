from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Literal, TypeVar, cast, overload

import wrapt
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    RateLimitError,
)
from openai.lib.streaming.chat import AsyncChatCompletionStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ParsedChatCompletion,
)
from openai.types.completion_usage import CompletionUsage
from opentelemetry.trace import Span
from pydantic import BaseModel

from . import telemetry
from .deployment import ModelDeployment

_T = TypeVar("_T", bound=BaseModel)

logger = logging.getLogger(__name__)


class OpenAIDeployment(ModelDeployment):
    """A model deployment speaking the Chat Completions API."""

    ratelimit_error = RateLimitError
    timeout_error = APITimeoutError
    connection_error = APIConnectionError

    # Assigned by Foundry.add, which shares one client per URL.
    client: AsyncOpenAI

    @property
    def url(self) -> str | None:
        if self.endpoint:
            return self.endpoint
        base = self.resource.base
        return f"{base}/openai/v1/" if base else None

    def new_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.resource.api_key,
            base_url=self.url,
            timeout=self.resource.timeout,
        )

    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(self, **kwargs) -> ChatCompletion: ...

    async def create(
        self, *, stream: bool = False, **kwargs
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """
        Send a chat completion request to this deployment.
        Tracks usage metrics for load balancing.
        """

        # add input token estimate before we send the request so utilization is
        # kept up to date for other requests that might be executing concurrently.
        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            if stream:
                logger.debug("creating streaming completion on %s", self.name)
                stream_options = kwargs.pop("stream_options", {"include_usage": True})
                response_stream = await self.client.chat.completions.create(
                    model=self.name,
                    stream=True,
                    stream_options=stream_options,
                    **kwargs,
                )

                return _AsyncStreamWrapper(
                    stream=response_stream,
                    model=self,
                    offset=_preflight_estimate,
                )

            logger.debug("creating chat completion on %s", self.name)
            response = cast(
                ChatCompletion,
                await self.client.chat.completions.create(model=self.name, **kwargs),
            )

            if response.usage:
                # dont double-count our preflight estimate
                self.spend_tokens(response.usage.total_tokens - _preflight_estimate)
                self._record_usage(response.usage)

            return response

        except Exception as e:
            self._handle_error(e, "completion")
            raise

    async def open_stream(self, **kwargs) -> AsyncChatCompletionStream:
        """Enter the SDK's stream helper, which accumulates a terminal completion.

        Unlike create(stream=True) the caller may never iterate — awaiting
        get_final_completion() alone is enough — so usage is not tapped per
        chunk here. reconcile_stream charges it once the stream is done.
        """
        self.spend_tokens(self._estimate_token_usage(kwargs))
        self.spend_request()

        try:
            logger.debug("opening completion stream on %s", self.name)
            # popped before the call: relying on a keyword being evaluated
            # before the ** unpacking beside it is needlessly subtle
            stream_options = kwargs.pop("stream_options", {"include_usage": True})
            return await self.client.chat.completions.stream(
                model=self.name,
                stream_options=stream_options,
                **kwargs,
            ).__aenter__()
        except Exception as e:
            self._handle_error(e, "stream")
            raise

    def reconcile_stream(
        self,
        stream: AsyncChatCompletionStream,
        offset: int,
        span: Span | None = None,
    ) -> None:
        """Charge what the stream actually accumulated against the estimate."""
        usage = stream.current_completion_snapshot.usage
        if usage:
            self.spend_tokens(usage.total_tokens - offset)
            self._record_usage(usage, span)

    async def parse(
        self, *, response_format: type[_T], **kwargs
    ) -> ParsedChatCompletion[_T]:
        """
        Send a structured output parse request to this deployment.
        Tracks usage metrics for load balancing.
        """

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            logger.debug("creating parsed completion on %s", self.name)
            response = await self.client.chat.completions.parse(
                model=self.name, response_format=response_format, **kwargs
            )

            if response.usage:
                self.spend_tokens(response.usage.total_tokens - _preflight_estimate)
                self._record_usage(response.usage)

            return response
        except Exception as e:
            self._handle_error(e, "parse")
            raise

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # ~4 chars per token, input only.
        t_input = sum(len(m.get("content", "")) for m in kwargs.get("messages", []))
        return t_input // 4

    def _record_usage(self, usage: CompletionUsage, span: Span | None = None) -> None:
        prompt = usage.prompt_tokens_details
        completion = usage.completion_tokens_details
        telemetry.record_usage(
            self,
            span,
            input=usage.prompt_tokens,
            output=usage.completion_tokens,
            cached=prompt.cached_tokens if prompt else None,
            reasoning=completion.reasoning_tokens if completion else None,
        )


class _AsyncStreamWrapper(wrapt.ObjectProxy):
    """Wrap an openai.AsyncStream to track usage.

    Chunks are consumed after create() returns, so the attempt span that opened
    the stream is long closed by the time usage lands. This owns a child of it
    that stays open until the last chunk, which is also the only span whose
    duration means anything for a stream.
    """

    def __init__(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        model: OpenAIDeployment,
        offset: int = 0,
    ):
        super().__init__(stream)
        self._self_model: OpenAIDeployment = model
        self._self_offset: int = offset
        self._self_span = telemetry.start_stream_span(model)

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        error: Exception | None = None
        try:
            async for chunk in self.__wrapped__:
                chunk = cast(ChatCompletionChunk, chunk)
                # only the last chunk contains the usage info
                if chunk.usage:
                    self._self_model.spend_tokens(
                        # dont double-count our preflight estimate
                        chunk.usage.total_tokens - self._self_offset
                    )
                    self._self_model._record_usage(chunk.usage, self._self_span)

                yield chunk
        except Exception as e:
            error = e
            self._self_model._handle_error(e, "stream")
            raise
        finally:
            # A consumer that breaks out of the loop or is cancelled leaves
            # through GeneratorExit or CancelledError, neither of which is an
            # Exception -- so the span is closed here rather than per branch,
            # and an abandoned stream does not leave one open forever.
            self._end_span(error)

    def _end_span(self, error: Exception | None = None) -> None:
        # a span that has already ended stops recording, which makes this safe
        # to reach from both iteration and close()
        if self._self_span.is_recording():
            telemetry.end_stream_span(self._self_span, error)

    async def close(self) -> None:
        """Closing the stream finishes it, iterated or not."""
        try:
            await self.__wrapped__.close()
        finally:
            self._end_span()
