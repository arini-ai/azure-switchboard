import asyncio
import logging
import random
from collections import OrderedDict
from typing import Callable, Dict, Literal, overload

from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

from .client import Client, Deployment, default_client_factory

logger = logging.getLogger(__name__)


class SwitchboardError(Exception):
    pass


class Switchboard:
    def __init__(
        self,
        deployments: list[Deployment],
        client_factory: Callable[[Deployment], Client] = default_client_factory,
        healthcheck_interval: int = 10,
        ratelimit_window: int = 60,  # Reset usage counters every minute
    ) -> None:
        self.deployments: Dict[str, Client] = {
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
        self.healthcheck_task = (
            asyncio.create_task(self.periodically_check_health())
            if self.healthcheck_interval > 0
            else None
        )

        self.ratelimit_reset_task = (
            asyncio.create_task(self.periodically_reset_usage())
            if self.ratelimit_window > 0
            else None
        )

    async def stop(self):
        for task in [self.healthcheck_task, self.ratelimit_reset_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def periodically_check_health(self):
        """Periodically check the health of all deployments"""

        async def _check_health(client: Client):
            # splay outbound requests by a little bit
            await asyncio.sleep(random.uniform(0, 1))
            await client.check_health()

        while True:
            await asyncio.sleep(self.healthcheck_interval)
            await asyncio.gather(
                *[_check_health(client) for client in self.deployments.values()]
            )

    async def periodically_reset_usage(self):
        """Periodically reset usage counters on all clients.

        This is pretty naive but it will suffice for now."""
        while True:
            await asyncio.sleep(self.ratelimit_window)
            logger.debug("Resetting usage counters")
            self.reset_usage()

    def reset_usage(self) -> None:
        for client in self.deployments.values():
            client.reset_counters()

    def get_usage(self) -> dict[str, dict]:
        return {
            name: client.get_counters() for name, client in self.deployments.items()
        }

    def select_deployment(self, session_id: str | None = None) -> Client:
        """
        Select a deployment using the power of two random choices algorithm.
        If session_id is provided, try to use that specific deployment first.
        """
        # Handle session-based routing first
        if session_id and session_id in self._sessions:
            client = self._sessions[session_id]
            if client.healthy:
                logger.debug(f"Using client {client} for session {session_id}")
                return client

            logger.warning(f"Client {client} is unhealthy, falling back to selection")

        # Get healthy deployments
        healthy_deployments = [c for c in self.deployments.values() if c.healthy]
        if not healthy_deployments:
            raise SwitchboardError("No healthy deployments available")

        if len(healthy_deployments) == 1:
            return healthy_deployments[0]

        # Power of two random choices
        choices = random.sample(healthy_deployments, min(2, len(healthy_deployments)))

        # Select the client with the lower weight (lower weight = better choice)
        selected = min(choices, key=lambda c: c.util)
        logger.debug(f"Selected deployment {selected} with weight {selected.util}")

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
                    client = self.select_deployment(session_id)
                    logger.debug(f"Sending completion request to {client}")
                    return await client.create(stream=stream, **kwargs)
        except RetryError as e:
            raise SwitchboardError("All attempts failed") from e
        except asyncio.CancelledError:
            pass

        # we should never reach here
        raise SwitchboardError("Unexpected error")

    async def stream(
        self, *, session_id: str | None = None, **kwargs
    ) -> AsyncStream[ChatCompletionChunk]:
        return await self.create(stream=True, session_id=session_id, **kwargs)

    def __repr__(self) -> str:
        return f"Switchboard({self.get_usage()})"


# borrowed from https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class LRUDict(OrderedDict):
    def __init__(self, *args, max_size: int = 1024, **kwargs):
        assert max_size > 0
        self.max_size = max_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.max_size:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
