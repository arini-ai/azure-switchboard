from __future__ import annotations

import asyncio
import logging
import random
from collections import OrderedDict
from typing import Callable, Literal, overload

from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

from .deployment import Deployment, DeploymentConfig, default_deployment_factory

logger = logging.getLogger(__name__)


class SwitchboardError(Exception):
    pass


class Switchboard:
    def __init__(
        self,
        deployments: list[DeploymentConfig],
        client_factory: Callable[[DeploymentConfig], Deployment] = (
            default_deployment_factory
        ),
        healthcheck_interval: int = 10,
        ratelimit_window: int = 60,  # Reset usage counters every minute
    ) -> None:
        self.deployments: dict[str, Deployment] = {
            deployment.name: client_factory(deployment) for deployment in deployments
        }

        self.healthcheck_interval = healthcheck_interval
        self.ratelimit_window = ratelimit_window

        self.fallback_policy = AsyncRetrying(
            stop=stop_after_attempt(2),
        )

        self._sessions = LRUDict(max_size=1024)

    def start(self) -> None:
        # Start background tasks if intervals are positive
        if self.ratelimit_window > 0:

            async def periodic_reset():
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

    def get_usage(self) -> dict[str, dict]:
        return {name: client.get_usage() for name, client in self.deployments.items()}

    def select_deployment(
        self, *, model: str, session_id: str | None = None
    ) -> Deployment:
        """
        Select a deployment using the power of two random choices algorithm.
        If session_id is provided, try to use that specific deployment first.
        """
        # Handle session-based routing first
        if session_id and session_id in self._sessions:
            client = self._sessions[session_id]
            if client.is_healthy(model):
                logger.debug(f"Reusing {client} for session {session_id}")
                return client

            logger.warning(f"{client} is unhealthy, falling back to selection")

        # Get healthy deployments for the requested model
        healthy_deployments = list(
            filter(lambda d: d.is_healthy(model), self.deployments.values())
        )
        if not healthy_deployments:
            raise SwitchboardError("No healthy deployments available")

        if len(healthy_deployments) == 1:
            return healthy_deployments[0]

        # Power of two random choices
        choices = random.sample(healthy_deployments, min(2, len(healthy_deployments)))

        # Select the client with the lower utilization for the model
        selected = min(choices, key=lambda d: d.util(model))
        logger.debug(f"Selected {selected}")

        if session_id:
            self._sessions[session_id] = selected

        return selected

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
        """
        Send a chat completion request to the selected deployment, with automatic fallback.
        """

        try:
            async for attempt in self.fallback_policy:
                with attempt:
                    client = self.select_deployment(model=model, session_id=session_id)
                    logger.debug(f"Sending completion request to {client}")
                    return await client.create(model=model, stream=stream, **kwargs)
        except RetryError as e:
            raise SwitchboardError("All attempts failed") from e
        except asyncio.CancelledError:  # pragma: no cover
            pass

        # we should never reach here
        raise SwitchboardError("Unexpected error")  # pragma: no cover

    def __repr__(self) -> str:
        return f"Switchboard({self.get_usage()})"


# borrowed from https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class LRUDict(OrderedDict):
    def __init__(self, *args, max_size: int = 1024, **kwargs):
        assert max_size > 0
        self.max_size = max_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Deployment) -> None:
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.max_size:  # pragma: no cover
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key: str) -> Deployment:
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
