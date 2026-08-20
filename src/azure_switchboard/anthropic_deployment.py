from __future__ import annotations

import logging
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
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import Message, ParsedMessage, RawMessageStreamEvent
from anthropic.types.usage import Usage
from opentelemetry.trace import Span
from pydantic import BaseModel

from . import telemetry
from .deployment import ModelDeployment

_T = TypeVar("_T", bound=BaseModel)

logger = logging.getLogger(__name__)


class AnthropicDeployment(ModelDeployment):
    """A model deployment speaking the Anthropic Messages API."""

    ratelimit_error = RateLimitError
    timeout_error = APITimeoutError
    connection_error = APIConnectionError

    # Assigned by Foundry.add, which shares one client per URL.
    client: AsyncAnthropic

    @property
    def url(self) -> str | None:
        if self.endpoint:
            return self.endpoint
        base = self.resource.base
        return f"{base}/anthropic/" if base else None

    def new_client(self) -> AsyncAnthropic:
        # AsyncAnthropicFoundry overrides auth to send Azure's api-key header,
        # so it is not interchangeable with the first-party client. No URL
        # means we are not on Foundry at all.
        if self.url is None:
            return AsyncAnthropic(
                api_key=self.resource.api_key, timeout=self.resource.timeout
            )
        return AsyncAnthropicFoundry(
            base_url=self.url,
            api_key=self.resource.api_key,
            timeout=self.resource.timeout,
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
                logger.debug("creating streaming message on %s", self.name)
                response_stream = await self.client.messages.create(
                    model=self.name, stream=True, **kwargs
                )

                return _AsyncMessageStreamWrapper(
                    stream=response_stream,
                    model=self,
                    offset=_preflight_estimate,
                )

            logger.debug("creating message on %s", self.name)
            response = cast(
                Message,
                await self.client.messages.create(model=self.name, **kwargs),
            )
            self._reconcile_usage(response.usage, _preflight_estimate)
            return response

        except Exception as e:
            self._handle_error(e, "message")
            raise

    async def open_stream(self, **kwargs) -> AsyncMessageStream:
        """Enter the SDK's stream helper, which accumulates a terminal Message.

        Unlike create(stream=True) the caller may never iterate — awaiting
        get_final_message() alone is enough — so usage is not tapped per event
        here. reconcile_stream charges it once the stream is done.
        """
        self.spend_tokens(self._estimate_token_usage(kwargs))
        self.spend_request()

        try:
            logger.debug("opening message stream on %s", self.name)
            return await self.client.messages.stream(
                model=self.name, **kwargs
            ).__aenter__()
        except Exception as e:
            self._handle_error(e, "stream")
            raise

    def reconcile_stream(
        self,
        stream: AsyncMessageStream,
        offset: int,
        span: Span | None = None,
    ) -> None:
        """Charge what the stream actually accumulated against the estimate."""
        try:
            usage = stream.current_message_snapshot.usage
        except AssertionError:
            # the SDK asserts on the snapshot before the first event lands, so
            # nothing was consumed and the preflight estimate stands
            return
        self._reconcile_usage(usage, offset, span)

    async def parse(self, *, output_format: type[_T], **kwargs) -> ParsedMessage[_T]:
        """
        Send a structured output request to this deployment.
        Tracks usage metrics for load balancing.
        """

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.spend_tokens(_preflight_estimate)
        self.spend_request()

        try:
            logger.debug("creating parsed message on %s", self.name)
            response = await self.client.messages.parse(
                model=self.name, output_format=output_format, **kwargs
            )
            self._reconcile_usage(response.usage, _preflight_estimate)
            return response
        except Exception as e:
            self._handle_error(e, "parse")
            raise

    def _reconcile_usage(
        self, usage: Usage | None, offset: int, span: Span | None = None
    ) -> None:
        """Charge real usage against the preflight estimate.

        The Messages API reports input and output separately; there is no
        total_tokens field.
        """
        if not usage:
            return
        # dont double-count our preflight estimate
        self.spend_tokens(usage.input_tokens + usage.output_tokens - offset)
        self._record_usage(usage, span)

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # ~4 chars per token, as in the chat path. Unlike chat, content may be
        # a list of blocks and the system prompt lives outside messages.
        chars = _content_len(kwargs.get("system", ""))
        for m in kwargs.get("messages", []):
            chars += _content_len(m.get("content", ""))
        return chars // 4

    def _record_usage(
        self,
        usage: Usage,
        span: Span | None = None,
        *,
        output: int | None = None,
    ) -> None:
        # named fields rather than getattr, so a field that goes away is a type
        # error instead of an attribute that silently reports nothing
        details = usage.output_tokens_details
        telemetry.record_usage(
            self,
            span,
            input=usage.input_tokens,
            # streaming reports a running output total on later events, so the
            # caller passes the delta rather than this snapshot's value
            output=usage.output_tokens if output is None else output,
            cached=usage.cache_read_input_tokens,
            reasoning=details.thinking_tokens if details else None,
        )


def _content_len(content: object) -> int:
    """Character count of Messages API content, which may be blocks."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


class _AsyncMessageStreamWrapper(wrapt.ObjectProxy):
    """Wrap an anthropic.AsyncStream to track usage.

    Events are consumed after create() returns, so the attempt span that opened
    the stream is long closed by the time usage lands. This owns a child of it
    that stays open until the last event, which is also the only span whose
    duration means anything for a stream.
    """

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
        self._self_span = telemetry.start_stream_span(model)

    async def __aiter__(self) -> AsyncIterator[RawMessageStreamEvent]:
        error: Exception | None = None
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
                    # output is charged from the deltas below, not from here
                    self._self_model._record_usage(usage, self._self_span, output=0)
                elif event.type == "message_delta" and event.usage:
                    output = event.usage.output_tokens or 0
                    # output_tokens is cumulative — only spend what's new
                    delta = output - self._self_output_spent
                    self._self_model.spend_tokens(delta)
                    self._self_output_spent = output
                    telemetry.tokens.add(
                        delta,
                        {
                            "model": self._self_model.name,
                            "resource": self._self_model.resource.name,
                            "kind": "output",
                        },
                    )

                yield event
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
