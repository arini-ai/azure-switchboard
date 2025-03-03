#!/usr/bin/env python3

"""
Switchboard

Switchboard is a library for distributing LLM inference workloads across
multiple Azure OpenAI deployments.

"""

import asyncio
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import AnyHttpUrl, BaseModel, SecretStr
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger("switchboard")


class AzureDeployment(BaseModel):
    name: str
    api_base: AnyHttpUrl
    api_key: SecretStr
    api_version: str = "2024-10-21"
    max_retries: int = 3
    timeout: float = 60.0
    max_rpm: int | None = None
    max_tpm: int | None = None


class Deployment:
    def __init__(self, config: AzureDeployment, mock: bool = False) -> None:
        self.config = config

        self._client: AsyncAzureOpenAI | None = None
        self._retry_policy = AsyncRetrying(
            retry=retry_if_exception_type((TimeoutError,)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
        )
        self._is_mock = mock

        self._request_count = 0
        self._last_request_time = 0.0

        self.is_healthy = True
        self.last_error_time = 0.0

    async def _get_client(self) -> AsyncAzureOpenAI:
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.config.api_key.get_secret_value(),
                api_version=self.config.api_version,
                azure_endpoint=str(self.config.api_base),
            )
        return self._client

    async def _mock_completion_response(self, sleep_for: float) -> ChatCompletion:
        await asyncio.sleep(sleep_for)

        return ChatCompletion(
            id="mock_id",
            model="mock_model",
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="mock response",
                    ),
                )
            ],
        )

    async def create_chat_completion(self, *args, **kwargs):
        client = await self._get_client()
        try:
            async for attempt in self._retry_policy:
                with attempt:
                    self._request_count += 1
                    self._last_request_time = time.monotonic()

                    if not self._is_mock:
                        response = await client.chat.completions.create(*args, **kwargs)
                    else:
                        response = await self._mock_completion_response(0.1)

                    self._request_count -= 1
                    return response

        except RetryError as e:
            self.is_healthy = False
            self.last_error_time = time.monotonic()
            # The exception will be reraised after the final retry attempt
            # We log it, and then let Switchboard handle trying other deployments
            logger.error(f"Deployment {self.config.name} failed: {e}")
            raise

        except Exception as e:
            self.is_healthy = False
            self.last_error_time = time.monotonic()
            logger.error(f"Deployment {self.config.name} failed: {e}")
            raise

    async def ping(self, model: str = "gpt-4o-mini"):
        try:
            await self.create_chat_completion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
            )
            self.is_healthy = True  # If no exception, it's healthy
        except Exception as e:
            logger.error(f"Deployment {self.config.name} failed to ping: {e}")
            pass


class HealthChecker:
    def __init__(
        self,
        deployments: List[Deployment],
        interval: float = 10.0,
        cooldown_period: float = 60.0,
    ):
        self.deployments = deployments
        self.interval = interval
        self.cooldown_period = cooldown_period
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def run(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(self.interval)
            for deployment in self.deployments:
                if not deployment.is_healthy:
                    if (
                        time.monotonic() - deployment.last_error_time
                        >= self.cooldown_period
                    ):
                        logger.info(
                            f"Deployment {deployment.config.name} cooldown period expired, marking as healthy."
                        )
                        deployment.is_healthy = True  # Recover after cooldown
                else:
                    await deployment.ping()

    async def start(self):
        if self._task is not None:
            raise ValueError("HealthChecker already started")
        self._task = asyncio.create_task(self.run())

    async def stop(self):
        self._stop_event.set()
        if self._task is not None:
            await self._task


class Switchboard:
    def __init__(
        self,
        deployments: List[AzureDeployment],
        *,
        health_check_interval: float = 10.0,
        cooldown_period: float = 60.0,
        sticky_session_duration: float = 600.0,
    ):
        self.deployments = [Deployment(config) for config in deployments]
        self._health_checker = HealthChecker(
            self.deployments, health_check_interval, cooldown_period
        )
        self._session_map: Dict[str, str] = {}  # Simple session map for now
        self.sticky_session_duration = sticky_session_duration
        # TODO: Initialize _metrics

    async def start(self):
        await self._health_checker.start()

    async def close(self):
        await self._health_checker.stop()
        for deployment in self.deployments:
            if deployment._client:
                await deployment._client.close()

    async def _get_deployment(self, session_id: Optional[str] = None):
        if session_id and session_id in self._session_map:
            deployment_name = self._session_map[session_id]
            deployment = next(
                (d for d in self.deployments if d.config.name == deployment_name),
                None,
            )
            if deployment and deployment.is_healthy:
                return deployment

        # Power of two choices
        healthy_deployments = [d for d in self.deployments if d.is_healthy]
        if not healthy_deployments:
            raise LoadBalancerError("No healthy deployments available")

        if len(healthy_deployments) == 1:
            chosen_deployment = healthy_deployments[0]
        else:
            d1, d2 = random.sample(healthy_deployments, 2)
            chosen_deployment = (
                d1 if d1._request_count <= d2._request_count else d2
            )  # Simple selection

        if session_id:
            self._session_map[session_id] = chosen_deployment.config.name
        return chosen_deployment

    async def chat_completion(self, messages, model, session_id=None, **kwargs):
        # Find a deployment
        deployment = await self._get_deployment(session_id)
        # try/except/retry loop around the deployment call
        # for handling failover
        try:
            response = await deployment.create_chat_completion(
                messages=messages, model=model, **kwargs
            )
            return response
        except Exception:
            # if we get here, retries on the deployment failed
            logger.info("Switchboard level failover")
            # remove from session map
            if session_id and session_id in self._session_map:
                del self._session_map[session_id]
            # find all deployments that aren't the failed deployment
            other_deployments = [
                d for d in self.deployments if d.config.name != deployment.config.name
            ]
            # pick a healthy one at random
            healthy_deployments = [d for d in other_deployments if d.is_healthy]
            if not healthy_deployments:
                raise LoadBalancerError("No healthy deployments available")
            chosen_deployment = random.choice(healthy_deployments)
            # try again
            response = await chosen_deployment.create_chat_completion(
                messages=messages, model=model, **kwargs
            )
            return response


class LoadBalancerError(Exception):
    pass


@asynccontextmanager
async def get_switchboard(deployments, **kwargs):
    switchboard = Switchboard(deployments, **kwargs)
    await switchboard.start()
    try:
        yield switchboard
    finally:
        await switchboard.close()


def _env(key: str) -> str:
    value = os.environ.get(key)
    assert value, f"missing {key}"
    return value


async def main():
    deployments = [
        AzureDeployment(
            name="default",
            api_base=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_env("AZURE_OPENAI_API_KEY"),
        )
    ]

    config = {
        "health_check_interval": 1,
        "cooldown_period": 5,
    }
    async with get_switchboard(deployments=deployments, **config) as switchboard:
        # quick self-test
        response = await switchboard.chat_completion(
            messages=[{"role": "user", "content": "Hello, world!"}],
            model="gpt-4o-mini",
            session_id="test_session",
        )
        logger.info(response.choices[0].message.content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
