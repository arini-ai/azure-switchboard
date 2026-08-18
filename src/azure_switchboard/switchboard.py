from __future__ import annotations

import asyncio
import random
from collections import OrderedDict
from functools import cached_property
from typing import (
    Literal,
    Protocol,
    TypeVar,
    overload,
)

from collections.abc import Awaitable, Callable, Sequence

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

from .anthropic_deployment import AnthropicDeployment
from .exceptions import SwitchboardError
from .resource import Foundry, Resource
from .deployment import ModelDeployment, UtilStats
from .openai_deployment import OpenAIDeployment

_T = TypeVar("_T", bound=BaseModel)
_R = TypeVar("_R")
_M = TypeVar("_M", bound=ModelDeployment)

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


class Selector(Protocol):
    """Picks one of the healthy deployments of a model.

    A Protocol rather than a Callable alias so the type variable is scoped to
    the call: one selector serves both pools without widening either.
    """

    def __call__(self, options: list[_M], /) -> _M: ...


def two_random_choices(options: list[_M], /) -> _M:
    """Power of two random choices algorithm.

    Randomly select 2 deployments and return the one with lower util. Every
    option is a deployment of the same model, so the model name is not an
    input to the choice.
    """
    selected = random.sample(options, min(2, len(options)))
    return min(selected, key=lambda d: d.util)


DEFAULT_FAILOVER_POLICY = AsyncRetrying(
    stop=stop_after_attempt(2),
    retry=retry_if_not_exception_type(SwitchboardError),
    reraise=True,
)


class Switchboard:
    def __init__(
        self,
        foundries: Sequence[Foundry],
        selector: Selector = two_random_choices,
        failover_policy: AsyncRetrying = DEFAULT_FAILOVER_POLICY,
        ratelimit_window: float = 60.0,
        max_sessions: int = 1024,
        openai_fallback: bool = False,
        anthropic_fallback: bool = False,
    ) -> None:
        if not foundries and not (openai_fallback or anthropic_fallback):
            raise SwitchboardError("No foundries provided")

        self.foundries: dict[str, Foundry] = {}
        for foundry in foundries:
            if foundry.name in self.foundries:
                raise SwitchboardError(f"Duplicate foundry name: {foundry.name}")
            self.foundries[foundry.name] = foundry

            # Each surface partitions these into its own pool; reject anything
            # neither would claim here, rather than on first use.
            for model in foundry.models.values():
                if not isinstance(model, (OpenAIDeployment, AnthropicDeployment)):
                    raise SwitchboardError(
                        f"{foundry.name}: {model.name} is not an OpenAIDeployment "
                        "or AnthropicDeployment"
                    )

        self._openai_fallback_enabled = openai_fallback
        self._anthropic_fallback_enabled = anthropic_fallback

        self.selector = selector
        self.failover_policy = failover_policy

        self.sessions: _LRUDict = _LRUDict(max_size=max_sessions)
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

    def _all_resources(self) -> list[Resource]:
        # each surface owns its first-party resource, so usage against a
        # fallback shows up in stats and resets on the same schedule
        first_party = [
            self.chat.completions._first_party,
            self.messages._first_party,
        ]
        return list(self.foundries.values()) + [f for f in first_party if f]

    def reset_usage(self) -> None:
        for resource in self._all_resources():
            resource.reset_usage()

    def stats(self) -> dict[str, dict[str, UtilStats]]:
        return {f.name: f.stats() for f in self._all_resources()}

    def _select(
        self,
        pool: dict[str, list[_M]],
        *,
        model: str,
        session_id: str | None,
        fallback: Callable[[], _M | None],
    ) -> _M:
        """Pick a deployment for a model out of one API's pool.

        Generic over the pool, so both surfaces share this implementation while
        each keeps its own concrete deployment type.
        """
        candidates = pool.get(model, [])

        if session_id and (pinned := self.sessions.get(session_id)):
            # Affinity is to the resource, so a session that uses several models
            # keeps hitting the same one and its prompt cache stays warm.
            preferred = [
                m for m in candidates if m.resource is pinned and m.is_healthy()
            ]
            if preferred:
                return self.selector(preferred)
            logger.warning(f"{model} is unhealthy on {pinned.name}, reselecting")

        eligible = [m for m in candidates if m.is_healthy()]

        if not eligible:
            if (first_party := fallback()) and first_party.is_healthy():
                logger.warning(f"No healthy deployments for {model}, using first-party")
                return first_party
            raise SwitchboardError(f"No deployments available for {model}")

        healthy_deployments_gauge.set(len(eligible), {"model": model})

        selected = eligible[0] if len(eligible) == 1 else self.selector(eligible)
        logger.trace(f"Selected deployment: {selected.resource.name}/{selected.name}")

        if session_id:
            self.sessions[session_id] = selected.resource

        return selected

    async def _dispatch(
        self,
        pool: dict[str, list[_M]],
        *,
        model: str,
        session_id: str | None,
        fallback: Callable[[], _M | None],
        call: Callable[[_M], Awaitable[_R]],
    ) -> _R:  # pyright: ignore[reportReturnType]
        with logger.contextualize(model=model, session_id=session_id):
            # failover_policy is copied so concurrent requests
            # dont share retry state
            async for attempt in self.failover_policy.copy():
                with attempt:
                    deployment = self._select(
                        pool, model=model, session_id=session_id, fallback=fallback
                    )
                    with logger.contextualize(resource=deployment.resource.name):
                        logger.trace("Sending request")
                        response = await call(deployment)
                    request_counter.add(
                        1,
                        {"model": model, "resource": deployment.resource.name},
                    )
                    return response

    def __repr__(self) -> str:
        return f"Switchboard({self.foundries})"


class _Chat:
    class Completions:
        """Everything specific to Chat Completions: the deployments that speak
        it, and the fallback for when none of them is usable."""

        def __init__(self, sb: Switchboard) -> None:
            self.sb = sb

            self._pool: dict[str, list[OpenAIDeployment]] = {}
            for foundry in sb.foundries.values():
                for model in foundry.models.values():
                    if isinstance(model, OpenAIDeployment):
                        self._pool.setdefault(model.name, []).append(model)

            # Not part of the pool: it is what selection reaches for once the
            # pool has nothing healthy left.
            self._first_party = (
                Resource(name="first-party-openai")
                if sb._openai_fallback_enabled
                else None
            )
            self._fallbacks: dict[str, OpenAIDeployment] = {}

        def _fallback(self, model: str) -> OpenAIDeployment | None:
            """The first-party deployment of a model, created on first need.

            Any model name resolves, including one on no resource at all — the
            vendor's API is the authority on whether it exists.
            """
            if self._first_party is None:
                return None
            if model not in self._fallbacks:
                deployment = OpenAIDeployment(model)
                self._first_party.add(deployment)
                self._fallbacks[model] = deployment
            return self._fallbacks[model]

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
            return await self.sb._dispatch(
                self._pool,
                model=model,
                session_id=session_id,
                fallback=lambda: self._fallback(model),
                call=lambda d: d.create(stream=stream, **kwargs),
            )

        async def parse(
            self,
            *,
            model: str,
            response_format: type[_T],
            session_id: str | None = None,
            **kwargs,
        ) -> ParsedChatCompletion[_T]:
            return await self.sb._dispatch(
                self._pool,
                model=model,
                session_id=session_id,
                fallback=lambda: self._fallback(model),
                call=lambda d: d.parse(response_format=response_format, **kwargs),
            )

    def __init__(self, switchboard: Switchboard) -> None:
        self.completions = _Chat.Completions(switchboard)


class _Messages:
    """Everything specific to the Messages API: the deployments that speak it,
    and the fallback for when none of them is usable."""

    def __init__(self, sb: Switchboard) -> None:
        self.sb = sb

        self._pool: dict[str, list[AnthropicDeployment]] = {}
        for foundry in sb.foundries.values():
            for model in foundry.models.values():
                if isinstance(model, AnthropicDeployment):
                    self._pool.setdefault(model.name, []).append(model)

        self._first_party = (
            Resource(name="first-party-anthropic")
            if sb._anthropic_fallback_enabled
            else None
        )
        self._fallbacks: dict[str, AnthropicDeployment] = {}

    def _fallback(self, model: str) -> AnthropicDeployment | None:
        if self._first_party is None:
            return None
        if model not in self._fallbacks:
            deployment = AnthropicDeployment(model)
            self._first_party.add(deployment)
            self._fallbacks[model] = deployment
        return self._fallbacks[model]

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
        return await self.sb._dispatch(
            self._pool,
            model=model,
            session_id=session_id,
            fallback=lambda: self._fallback(model),
            call=lambda d: d.create(stream=stream, **kwargs),
        )

    async def parse(
        self,
        *,
        model: str,
        output_format: type[_T],
        session_id: str | None = None,
        **kwargs,
    ) -> ParsedMessage[_T]:
        return await self.sb._dispatch(
            self._pool,
            model=model,
            session_id=session_id,
            fallback=lambda: self._fallback(model),
            call=lambda d: d.parse(output_format=output_format, **kwargs),
        )


# borrowed from https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class _LRUDict(OrderedDict):
    def __init__(self, *args, max_size: int = 1024, **kwargs):
        assert max_size > 0
        self.max_size = max_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Resource) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.max_size:  # pragma: no cover
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key: str) -> Resource:
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
