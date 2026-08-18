from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast, overload

import wrapt
from loguru import logger
from pydantic import BaseModel

from .deployment import Api, DeploymentBase
from .exceptions import SwitchboardError
from .model import Model

if TYPE_CHECKING:
    from anthropic import AsyncAnthropicFoundry, AsyncStream
    from anthropic.types import Message, RawMessageStreamEvent

_T = TypeVar("_T", bound=BaseModel)

_INSTALL_HINT = (
    "The anthropic SDK is required for Messages API deployments. "
    "Install it with: pip install 'azure-switchboard[anthropic]'"
)


def _anthropic():
    """Import the anthropic SDK, or explain how to get it.

    Kept optional so OpenAI-only users don't pay for a second SDK.
    """
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise SwitchboardError(_INSTALL_HINT) from e
    return anthropic


@dataclass
class AnthropicConfig:
    """Configuration for an Anthropic deployment on Azure AI Foundry.

    Set `resource` to target a Foundry resource, which resolves to
    https://{resource}.services.ai.azure.com/anthropic/. Pass `base_url`
    instead to point at an arbitrary endpoint.
    """

    name: str
    resource: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    # 30s suits interactive workloads; override for long-running batch jobs
    timeout: float = 30.0
    models: list[Model] = field(default_factory=list)

    def get_client(self) -> AsyncAnthropicFoundry:
        anthropic = _anthropic()
        if not self.resource and not self.base_url:
            raise SwitchboardError(
                f"{self.name}: one of resource or base_url is required"
            )
        return anthropic.AsyncAnthropicFoundry(
            resource=self.resource,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )


class AnthropicDeployment(DeploymentBase):
    """Runtime state of a deployment speaking the Anthropic Messages API"""

    api: ClassVar[Api] = "messages"

    def __init__(self, config: AnthropicConfig) -> None:
        super().__init__(config.name, config.models)
        self.config = config
        self.client = config.get_client()
        # Bind once so error handlers can reference the SDK's exception classes.
        # These share names with the openai ones but are distinct classes.
        self._errors = _anthropic()

    @overload
    async def messages(
        self, *, model: str, stream: Literal[True], **kwargs
    ) -> AsyncStream[RawMessageStreamEvent]: ...

    @overload
    async def messages(self, *, model: str, **kwargs) -> Message: ...

    async def messages(
        self,
        *,
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> Message | AsyncStream[RawMessageStreamEvent]:
        """
        Send a Messages API request to this client.
        Tracks usage metrics for load balancing.
        """

        self._check_model(model)

        # add input token estimate before we send the request so utilization is
        # kept up to date for other requests that might be executing concurrently.
        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.models[model].spend_tokens(_preflight_estimate)
        self.models[model].spend_request()

        try:
            if stream:
                logger.trace("Creating streaming message")
                response_stream = await self.client.messages.create(
                    model=model, stream=True, **kwargs
                )

                # streaming util gets updated inside _AsyncMessageStreamWrapper
                return _AsyncMessageStreamWrapper(
                    stream=response_stream,
                    deployment=self,
                    model=self.models[model],
                    offset=_preflight_estimate,
                )

            logger.trace("Creating message")
            response = cast(
                "Message",
                await self.client.messages.create(model=model, **kwargs),
            )
            self._reconcile_usage(model, response.usage, _preflight_estimate)
            return response

        except self._errors.RateLimitError:
            logger.exception("Marking down model for rate limit")
            self.models[model].mark_down()
            raise
        except self._errors.APITimeoutError:
            # Timeouts during upstream-wide slowdowns are uncorrelated with
            # which deployment was chosen — marking it down wastes capacity.
            logger.warning("Upstream timeout on message; not marking down")
            raise
        except self._errors.APIConnectionError:
            logger.exception("Marking down model for connection error")
            self.models[model].mark_down()
            raise

    async def parse(
        self,
        *,
        model: str,
        output_format: type[_T],
        **kwargs,
    ) -> Any:
        """
        Send a structured output request to this client.
        Tracks usage metrics for load balancing.
        """

        self._check_model(model)

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.models[model].spend_tokens(_preflight_estimate)
        self.models[model].spend_request()

        try:
            logger.trace("Creating parsed message")
            response = await self.client.messages.parse(
                model=model, output_format=output_format, **kwargs
            )
            self._reconcile_usage(model, response.usage, _preflight_estimate)
            return response
        except self._errors.RateLimitError:
            logger.exception("Marking down model for rate limit on parse")
            self.models[model].mark_down()
            raise
        except self._errors.APITimeoutError:
            logger.warning("Upstream timeout on parse; not marking down")
            raise
        except self._errors.APIConnectionError:
            logger.exception("Marking down model for connection error on parse")
            self.models[model].mark_down()
            raise

    def _reconcile_usage(self, model: str, usage, offset: int) -> None:
        """Charge real usage against the preflight estimate.

        The Messages API reports input and output separately; there is no
        total_tokens field.
        """
        if not usage:
            return
        # dont double-count our preflight estimate
        self.models[model].spend_tokens(
            usage.input_tokens + usage.output_tokens - offset
        )
        self._set_span_attributes(usage)

    def _estimate_token_usage(self, kwargs: dict) -> int:
        # loose estimate of token cost, mirroring the chat path's ~4 chars per
        # token heuristic. Unlike chat, content may be a list of blocks and the
        # system prompt lives outside messages.
        chars = _content_len(kwargs.get("system", ""))
        for m in kwargs.get("messages", []):
            chars += _content_len(m.get("content", ""))
        return chars // 4

    def _set_span_attributes(self, usage) -> None:
        self._record_token_details(
            cached=getattr(usage, "cache_read_input_tokens", None),
            reasoning=getattr(
                getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
            ),
        )


def _content_len(content: Any) -> int:
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
        deployment: AnthropicDeployment,
        model: Model,
        offset: int = 0,
    ):
        super().__init__(stream)
        self._self_deployment: AnthropicDeployment = deployment
        self._self_model: Model = model
        self._self_offset: int = offset
        self._self_output_spent: int = 0
        # Stream events are consumed after Switchboard.messages() returns, so
        # the contextualized logger scope from messages() is no longer active
        # here. Keep a bound logger on the wrapper to retain deployment/model
        # context for mid-stream error logs.
        self._self_logger = logger.bind(
            deployment=deployment.name,
            model=model.name,
        )

    async def __aiter__(self) -> AsyncIterator[RawMessageStreamEvent]:
        errors = self._self_deployment._errors
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
                    self._self_deployment._set_span_attributes(usage)
                elif event.type == "message_delta" and event.usage:
                    output = event.usage.output_tokens or 0
                    # output_tokens is cumulative — only spend what's new
                    self._self_model.spend_tokens(output - self._self_output_spent)
                    self._self_output_spent = output

                yield event
        except errors.RateLimitError:
            self._self_logger.exception("Marking down model for rate limit on stream")
            self._self_model.mark_down()
            raise
        except errors.APITimeoutError:
            self._self_logger.warning("Upstream timeout on stream; not marking down")
            raise
        except errors.APIConnectionError:
            self._self_logger.exception(
                "Marking down model for connection error on stream"
            )
            self._self_model.mark_down()
            raise
