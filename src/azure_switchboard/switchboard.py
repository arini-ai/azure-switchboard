from __future__ import annotations

import asyncio
import logging
import random
from collections import OrderedDict
from typing import Callable, Literal, Sequence, overload

from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from opentelemetry import metrics
from tenacity import (
    AsyncRetrying,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from azure_switchboard.model import UtilStats

from .deployment import (
    Deployment,
    _RuntimeDeployment,
)

logger = logging.getLogger(__name__)

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


class SwitchboardError(Exception):
    pass


def two_random_choices(model: str, options: list[_RuntimeDeployment]) -> _RuntimeDeployment:
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


class Switchboard:
    def __init__(
        self,
        deployments: Sequence[Deployment],
        selector: Callable[
            [str, list[_RuntimeDeployment]], _RuntimeDeployment
        ] = two_random_choices,
        failover_policy: AsyncRetrying = DEFAULT_FAILOVER_POLICY,
        ratelimit_window: float = 60.0,
        max_sessions: int = 1024,
    ) -> None:
        if not deployments:
            raise SwitchboardError("No deployments provided")

        self.deployments: dict[str, _RuntimeDeployment] = {}
        for deployment in deployments:
            if deployment.name in self.deployments:
                raise SwitchboardError(f"Duplicate deployment name: {deployment.name}")
            self.deployments[deployment.name] = _RuntimeDeployment(deployment)

        self.selector = selector
        self.failover_policy = failover_policy

        # LRUDict to expire old sessions
        self.sessions = _LRUDict(max_size=max_sessions)

        # reset usage every N seconds
        self.ratelimit_window = ratelimit_window

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
                logger.debug("Resetting usage counters")
                self.reset_usage()

        self.ratelimit_reset_task = asyncio.create_task(periodic_reset())

    async def stop(self):
        if self.ratelimit_reset_task:
            try:
                self.ratelimit_reset_task.cancel()
                await self.ratelimit_reset_task
            except asyncio.CancelledError:
                pass

    def reset_usage(self) -> None:
        for client in self.deployments.values():
            client.reset_usage()

    def stats(self) -> dict[str, dict[str, UtilStats]]:
        return {
            name: deployment.stats() for name, deployment in self.deployments.items()
        }

    def select_deployment(
        self, *, model: str, session_id: str | None = None
    ) -> _RuntimeDeployment:
        """
        Select a deployment using the power of two random choices algorithm.
        If session_id is provided, try to use that specific deployment first.
        """
        # Handle session-based routing first
        if session_id and session_id in self.sessions:
            client = self.sessions[session_id]
            if client.is_healthy(model):
                logger.debug(f"Reusing {client} for session {session_id}")
                return client

            logger.warning(f"{client} is unhealthy, falling back to selection")

        # Get eligible deployments for the requested model
        eligible_deployments = [
            d for d in self.deployments.values() if d.is_healthy(model)
        ]

        if not eligible_deployments:
            raise SwitchboardError(f"No eligible deployments available for {model}")

        # Record healthy deployments count metric
        healthy_deployments_gauge.set(len(eligible_deployments), {"model": model})

        if len(eligible_deployments) == 1:
            selection = eligible_deployments[0]
        else:
            selection = self.selector(model, eligible_deployments)

        logger.debug(f"Selected {selection}")

        if session_id:
            self.sessions[session_id] = selection

        return selection

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
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:  # pyright: ignore[reportReturnType]
        """
        Send a chat completion request to the selected deployment, with automatic failover.
        """
        async for attempt in self.failover_policy:
            with attempt:
                deployment = self.select_deployment(
                    model=model, session_id=session_id
                )
                logger.debug(f"Sending completion request to {deployment}")
                response = await deployment.create(
                    model=model, stream=stream, **kwargs
                )
                request_counter.add(
                    1, {"model": model, "deployment": deployment.config.name}
                )
                return response

    def __repr__(self) -> str:
        return f"Switchboard({self.deployments})"


# borrowed from https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class _LRUDict(OrderedDict):
    def __init__(self, *args, max_size: int = 1024, **kwargs):
        assert max_size > 0
        self.max_size = max_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: _RuntimeDeployment) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.max_size:  # pragma: no cover
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key: str) -> _RuntimeDeployment:
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
