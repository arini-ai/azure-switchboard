from __future__ import annotations

import asyncio
import random
from collections import OrderedDict
from functools import cached_property
from typing import (
    Awaitable,
    Callable,
    Literal,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from anthropic import AsyncStream as AsyncAnthropicStream
from anthropic.types import Message, ParsedMessage, RawMessageStreamEvent
from loguru import logger
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from opentelemetry import metrics
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from azure_switchboard.model import UtilStats

from .openai_api import OpenAIConfig, OpenAIDeployment
from .deployment import DeploymentBase
from .exceptions import SwitchboardError
from .anthropic_api import AnthropicConfig, AnthropicDeployment

_T = TypeVar("_T", bound=BaseModel)
_R = TypeVar("_R")

DeploymentSpec = OpenAIConfig | AnthropicConfig

meter = metrics.get_meter("azure_switchboard.switchboard")
deployment_util = meter.create_gauge(
    name="switchboard.deployment.model.utilization",
    description="Utilization of a model on a deployment",
    unit="%",
)
healthy_deployments_gauge = meter.create_gauge(
    name="healthy_deployments_count",
    description="Number of healthy deployments available for a model",
    unit="1",
)
deployment_failures_counter = meter.create_counter(
    name="deployment_failures",
    description="Number of deployment failures",
    unit="1",
)
request_counter = meter.create_counter(
    name="requests",
    description="Number of requests sent through the switchboard",
    unit="1",
)


def two_random_choices(model: str, options: list[DeploymentBase]) -> DeploymentBase:
    """Power of two random choices algorithm.

    Randomly select 2 deployments and return the one
    with lower util for the given model.
    """
    selected = random.sample(options, min(2, len(options)))
    return min(selected, key=lambda d: d.util(model))


DEFAULT_FAILOVER_POLICY = AsyncRetrying(
    stop=stop_after_attempt(2),
    retry=retry_if_not_exception_type(SwitchboardError),
    reraise=True,
)


def _build_deployment(config: DeploymentSpec) -> DeploymentBase:
    if isinstance(config, AnthropicConfig):
        return AnthropicDeployment(config)
    return OpenAIDeployment(config)


class Switchboard:
    def __init__(
        self,
        deployments: Sequence[DeploymentSpec],
        selector: Callable[
            [str, list[DeploymentBase]], DeploymentBase
        ] = two_random_choices,
        failover_policy: AsyncRetrying = DEFAULT_FAILOVER_POLICY,
        ratelimit_window: float = 60.0,
        max_sessions: int = 1024,
    ) -> None:
        if not deployments:
            raise SwitchboardError("No deployments provided")

        self.deployments: dict[str, DeploymentBase] = {}
        # Routing is by model name alone, so a name on both providers would
        # make the serving surface ambiguous.
        owners: dict[str, tuple[str, str]] = {}
        for config in deployments:
            if config.name in self.deployments:
                raise SwitchboardError(f"Duplicate deployment name: {config.name}")

            provider = "anthropic" if isinstance(config, AnthropicConfig) else "openai"
            for model in config.models:
                owner = owners.setdefault(model.name, (provider, config.name))
                if owner[0] != provider:
                    raise SwitchboardError(
                        f"{model.name} is registered on both {owner[0]} "
                        f"({owner[1]}) and {provider} ({config.name}) deployments; "
                        "a model name must belong to exactly one provider"
                    )

            self.deployments[config.name] = _build_deployment(config)

        self.selector = selector
        self.failover_policy = failover_policy

        self.sessions = _LRUDict(max_size=max_sessions)
        self.ratelimit_reset_task: asyncio.Task | None = None

        self.ratelimit_window = ratelimit_window

    @cached_property
    def chat(self) -> _Chat:
        return _Chat(self)

    @cached_property
    def messages(self) -> _Messages:
        return _Messages(self)

    async def __aenter__(self) -> Switchboard:
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.stop()

    def start(self) -> None:
        async def periodic_reset():
            if not self.ratelimit_window:
                return

            while True:
                await asyncio.sleep(self.ratelimit_window)
                self.reset_usage()

        self.ratelimit_reset_task = asyncio.create_task(periodic_reset())

    async def stop(self) -> None:
        if self.ratelimit_reset_task:
            try:
                self.ratelimit_reset_task.cancel()
                await self.ratelimit_reset_task
            except asyncio.CancelledError:
                pass

    def reset_usage(self) -> None:
        for deployment in self.deployments.values():
            deployment.reset_usage()

    def stats(self) -> dict[str, dict[str, UtilStats]]:
        return {
            name: deployment.stats() for name, deployment in self.deployments.items()
        }

    def select_deployment(
        self, *, model: str, session_id: str | None = None
    ) -> DeploymentBase:
        """
        Select a deployment using the configured selection algorithm.
        If session_id is provided, try to use that specific deployment first.
        """
        if session_id and session_id in self.sessions:
            deployment = self.sessions[session_id]
            if deployment.is_healthy(model):
                return deployment

            m = deployment.models.get(model)
            logger.bind(util=vars(m.stats()) if m else None).warning(
                f"{model} is unhealthy on {deployment.name}, falling back to selection"
            )

        eligible_deployments = [
            d for d in self.deployments.values() if d.is_healthy(model)
        ]

        if not eligible_deployments:
            # No healthy deployments — fall back to any deployment that supports
            # this model (even if cooling down) rather than failing immediately.
            # This prevents cascade failures when a single deployment has a
            # transient error: without fallback, the 10s cooldown would reject
            # every request, turning one failure into dozens.
            fallback_deployments = [
                d for d in self.deployments.values() if model in d.models
            ]
            if not fallback_deployments:
                raise SwitchboardError(f"No deployments available for {model}")
            logger.warning(
                f"No healthy deployments for {model}, using best-effort fallback"
            )
            eligible_deployments = fallback_deployments

        healthy_deployments_gauge.set(len(eligible_deployments), {"model": model})

        if len(eligible_deployments) == 1:
            deployment = eligible_deployments[0]
        else:
            deployment = self.selector(model, eligible_deployments)

        logger.trace(f"Selected deployment: {deployment.name}")

        if session_id:
            self.sessions[session_id] = deployment

        return deployment

    def __repr__(self) -> str:
        return f"Switchboard({self.deployments})"


class _Chat:
    class Completions:
        def __init__(self, sb: Switchboard) -> None:
            self.sb = sb

        async def _dispatch(
            self,
            *,
            model: str,
            session_id: str | None,
            call: Callable[[OpenAIDeployment], Awaitable[_R]],
        ) -> _R:  # pyright: ignore[reportReturnType]
            with logger.contextualize(model=model, session_id=session_id):
                # failover_policy is copied so concurrent requests
                # dont share retry state
                async for attempt in self.sb.failover_policy.copy():
                    with attempt:
                        deployment = cast(
                            OpenAIDeployment,
                            self.sb.select_deployment(
                                model=model, session_id=session_id
                            ),
                        )
                        with logger.contextualize(deployment=deployment.name):
                            logger.trace("Sending request")
                            response = await call(deployment)
                        request_counter.add(
                            1, {"model": model, "deployment": deployment.name}
                        )
                        return response

        @overload
        async def create(
            self, *, session_id: str | None = None, stream: Literal[True], **kwargs
        ) -> AsyncStream[ChatCompletionChunk]: ...

        @overload
        async def create(
            self, *, session_id: str | None = None, **kwargs
        ) -> ChatCompletion: ...

        async def create(
            self,
            *,
            model: str,
            session_id: str | None = None,
            stream: bool = False,
            **kwargs,
        ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
            return await self._dispatch(
                model=model,
                session_id=session_id,
                call=lambda d: d.create(model=model, stream=stream, **kwargs),
            )

        async def parse(
            self,
            *,
            model: str,
            response_format: type[_T],
            session_id: str | None = None,
            **kwargs,
        ) -> ParsedChatCompletion[_T]:
            return await self._dispatch(
                model=model,
                session_id=session_id,
                call=lambda d: d.parse(
                    model=model, response_format=response_format, **kwargs
                ),
            )

    def __init__(self, switchboard: Switchboard) -> None:
        self.completions = _Chat.Completions(switchboard)


class _Messages:
    def __init__(self, sb: Switchboard) -> None:
        self.sb = sb

    # duplicated with impl in _Chat to avoid fighting the type checker
    async def _dispatch(
        self,
        *,
        model: str,
        session_id: str | None,
        call: Callable[[AnthropicDeployment], Awaitable[_R]],
    ) -> _R:  # pyright: ignore[reportReturnType]
        with logger.contextualize(model=model, session_id=session_id):
            # failover_policy is copied so concurrent requests
            # dont share retry state
            async for attempt in self.sb.failover_policy.copy():
                with attempt:
                    deployment = cast(
                        AnthropicDeployment,
                        self.sb.select_deployment(model=model, session_id=session_id),
                    )
                    with logger.contextualize(deployment=deployment.name):
                        logger.trace("Sending request")
                        response = await call(deployment)
                    request_counter.add(
                        1, {"model": model, "deployment": deployment.name}
                    )
                    return response

    @overload
    async def create(
        self, *, session_id: str | None = None, stream: Literal[True], **kwargs
    ) -> AsyncAnthropicStream[RawMessageStreamEvent]: ...

    @overload
    async def create(self, *, session_id: str | None = None, **kwargs) -> Message: ...

    async def create(
        self,
        *,
        model: str,
        session_id: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Message | AsyncAnthropicStream[RawMessageStreamEvent]:
        """
        Send a Messages API request to the selected deployment, with automatic failover.

        `max_tokens` is required by the Messages API and is passed through
        unchanged; switchboard does not supply a default.
        """
        return await self._dispatch(
            model=model,
            session_id=session_id,
            call=lambda d: d.messages(model=model, stream=stream, **kwargs),
        )

    async def parse(
        self,
        *,
        model: str,
        output_format: type[_T],
        session_id: str | None = None,
        **kwargs,
    ) -> ParsedMessage[_T]:
        return await self._dispatch(
            model=model,
            session_id=session_id,
            call=lambda d: d.parse(model=model, output_format=output_format, **kwargs),
        )


# borrowed from https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class _LRUDict(OrderedDict):
    def __init__(self, *args, max_size: int = 1024, **kwargs):
        assert max_size > 0
        self.max_size = max_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: DeploymentBase) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.max_size:  # pragma: no cover
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key: str) -> DeploymentBase:
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
