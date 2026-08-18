from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, TypeVar, cast, overload

import wrapt
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    AsyncAnthropicFoundry,
    AsyncStream,
    RateLimitError,
)
from anthropic.types import Message, ParsedMessage, RawMessageStreamEvent
from anthropic.types.usage import Usage
from loguru import logger
from pydantic import BaseModel

from .model import ModelBase

_T = TypeVar("_T", bound=BaseModel)

# AsyncAnthropicFoundry subclasses AsyncAnthropic, so the base type covers both
# the Foundry client and the first-party one.
AnthropicClient = AsyncAnthropic


class AnthropicDeployment(ModelBase):
    """A model deployment speaking the Anthropic Messages API."""

    @property
    def url(self) -> str | None:
        if self.endpoint:
            return self.endpoint
        base = self.foundry.base
        return f"{base}anthropic/" if base else None

    @property
    def client(self) -> AnthropicClient:
        # AsyncAnthropicFoundry overrides auth to send Azure's api-key header,
        # so it is not interchangeable with the first-party client. No URL
        # means we are not on Foundry at all.
        foundry, url = self.foundry, self.url
        if url is None:
            return foundry.anthropic_client(
                None,
                lambda: AsyncAnthropic(
                    api_key=foundry.api_key, timeout=foundry.timeout
                ),
            )
        return foundry.anthropic_client(
            url,
            lambda: AsyncAnthropicFoundry(
                base_url=url, api_key=foundry.api_key, timeout=foundry.timeout
            ),
        )

    @overload
    async def create(
        self, *, stream: Literal[True], **kwargs
    ) -> AsyncStream[RawMessageStreamEvent]: ...

    @overload
    async def create(self, **kwargs) -> Message: ...

    async def create(
        self, *, stream: bool = False, **kwargs
    ) -> Message | AsyncStream[RawMessageStreamEvent]:
        """
        Send a Messages API request to this deployment.
        Tracks usage metrics for load balancing.
        """

        # add input token estimate before we send the request so utilization is
        # kept up to date for other requests that might be executing concurrently.
        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            if stream:
                logger.trace("Creating streaming message")
                response_stream = await self.client.messages.create(
                    model=self.name, stream=True, **kwargs
                )

                return _AsyncMessageStreamWrapper(
                    stream=response_stream,
                    model=self,
                    offset=_preflight_estimate,
                )

            logger.trace("Creating message")
            response = cast(
                Message,
                await self.client.messages.create(model=self.name, **kwargs),
            )
            self._reconcile_usage(response.usage, _preflight_estimate)
            return response

        except Exception as e:
            self._handle_error(e, "message")
            raise

    async def parse(self, *, output_format: type[_T], **kwargs) -> ParsedMessage[_T]:
        """
        Send a structured output request to this deployment.
        Tracks usage metrics for load balancing.
        """

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            logger.trace("Creating parsed message")
            response = await self.client.messages.parse(
                model=self.name, output_format=output_format, **kwargs
            )
            self._reconcile_usage(response.usage, _preflight_estimate)
            return response
        except Exception as e:
            self._handle_error(e, "parse")
            raise

    def _handle_error(self, exc: Exception, op: str, log=logger) -> None:
        """Scope a markdown to what the error actually implicates.

        A 429 is one deployment's quota; a connection error is the whole
        resource. Timeouts during upstream-wide slowdowns are uncorrelated with
        which deployment was chosen, so they mark down nothing.
        """
        if isinstance(exc, RateLimitError):
            log.exception(f"Marking down model for rate limit on {op}")
            self.mark_down()
        elif isinstance(exc, APITimeoutError):
            log.warning(f"Upstream timeout on {op}; not marking down")
        elif isinstance(exc, APIConnectionError):
            log.exception(f"Marking down foundry for connection error on {op}")
            self.foundry.mark_down()

    def _reconcile_usage(self, usage: Usage | None, offset: int) -> None:
        """Charge real usage against the preflight estimate.

        The Messages API reports input and output separately; there is no
        total_tokens field.
        """
        if not usage:
            return
        # dont double-count our preflight estimate
        self.spend_tokens(usage.input_tokens + usage.output_tokens - offset)
        self._set_span_attributes(usage)

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # ~4 chars per token, as in the chat path. Unlike chat, content may be
        # a list of blocks and the system prompt lives outside messages.
        chars = _content_len(kwargs.get("system", ""))
        for m in kwargs.get("messages", []):
            chars += _content_len(m.get("content", ""))
        return chars // 4

    def _set_span_attributes(self, usage: Usage) -> None:
        self._record_token_details(
            cached=getattr(usage, "cache_read_input_tokens", None),
            reasoning=getattr(
                getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
            ),
        )


def _content_len(content: object) -> int:
    """Character count of Messages API content, which may be blocks."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


class _AsyncMessageStreamWrapper(wrapt.ObjectProxy):
    """Wrap an anthropic.AsyncStream to track usage"""

    def __init__(
        self,
        stream: AsyncStream[RawMessageStreamEvent],
        model: AnthropicDeployment,
        offset: int = 0,
    ):
        super().__init__(stream)
        self._self_model: AnthropicDeployment = model
        self._self_offset: int = offset
        self._self_output_spent: int = 0
        # Events are consumed after create() returns, once its contextualize
        # scope is gone, so bind the context for mid-stream error logs.
        self._self_logger = logger.bind(
            foundry=model.foundry.name,
            model=model.name,
        )

    async def __aiter__(self) -> AsyncIterator[RawMessageStreamEvent]:
        try:
            async for event in self.__wrapped__:
                # Usage arrives in two places: message_start carries the input
                # tokens, and each message_delta carries a running output total.
                if event.type == "message_start":
                    usage = event.message.usage
                    self._self_model.spend_tokens(
                        # dont double-count our preflight estimate
                        usage.input_tokens - self._self_offset
                    )
                    self._self_model._set_span_attributes(usage)
                elif event.type == "message_delta" and event.usage:
                    output = event.usage.output_tokens or 0
                    # output_tokens is cumulative — only spend what's new
                    self._self_model.spend_tokens(output - self._self_output_spent)
                    self._self_output_spent = output

                yield event
        except Exception as e:
            self._self_model._handle_error(e, "stream", log=self._self_logger)
            raise
