import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional

import opentelemetry.trace as trace
from openai import AsyncAzureOpenAI
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, Field, HttpUrl

# Tracer for OpenTelemetry
tracer = trace.get_tracer("switchboard")

test_client = AsyncAzureOpenAI(
    azure_endpoint="https://hireai-openai-east.openai.azure.com/",
    api_key="72532109eae542c399b88dd5c017c6ef",
    api_version="2024-10-21",
)


class AzureDeployment(BaseModel):
    """Azure deployment configuration"""

    name: str
    api_base: HttpUrl
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 60.0
    max_tpm: Annotated[int, Field(description="Tokens per minute limit")] = 0
    max_rpm: Annotated[int, Field(description="Requests per minute limit")] = 0
    healthcheck_interval: int = 30


@dataclass
class Deployment:
    """Runtime state of a deployment"""

    config: AzureDeployment
    client: AsyncAzureOpenAI
    tpm_usage: int = 0
    rpm_usage: int = 0
    last_reset: float = field(default_factory=time.time)
    healthy: bool = True

    # @property
    # def token_utilization(self) -> float:
    #     """Return token utilization as a value between 0.0 and 1.0"""
    #     self._maybe_reset_counters()
    #     return self.tpm_usage / self.config.max_tpm if self.config.max_tpm > 0 else 1.0

    # @property
    # def request_utilization(self) -> float:
    #     """Return request utilization as a value between 0.0 and 1.0"""
    #     self._maybe_reset_counters()
    #     return self.rpm_usage / self.config.max_rpm if self.config.max_rpm > 0 else 1.0

    # def _maybe_reset_counters(self):
    #     """Reset counters if more than a minute has passed"""
    #     now = time.time()
    #     if now - self.last_reset >= 60:
    #         self.tpm_usage = 0
    #         self.rpm_usage = 0
    #         self.last_reset = now


def default_client_factory(config: AzureDeployment) -> AsyncAzureOpenAI:
    """Default factory for creating AsyncAzureOpenAI clients."""

    return AsyncAzureOpenAI(
        azure_endpoint=str(config.api_base),
        api_key=config.api_key,
        api_version=config.api_version,
    )


class Switchboard:
    def __init__(
        self,
        deployments: List[AzureDeployment],
        client_factory: Callable[
            [AzureDeployment], AsyncAzureOpenAI
        ] = default_client_factory,
        health_check_interval: int = 30,
    ):
        self.client_factory = client_factory
        self.deployments = [
            Deployment(config=d, client=client_factory(d)) for d in deployments
        ]
        self.health_check_interval = health_check_interval
        self._health_check_task = None

        # self.start_health_checks()

    def start_health_checks(self):
        """Start background health check task"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    def stop_health_checks(self):
        """Stop the health check task"""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task
            self._health_check_task = None

    async def _health_check_loop(self):
        """Periodically check health of all deployments"""
        while True:
            with tracer.start_as_current_span(
                "health_check_loop", kind=SpanKind.INTERNAL
            ):
                for deployment in self.deployments:
                    try:
                        # Simple ping to check if deployment is responsive
                        with tracer.start_as_current_span(
                            "health_check",
                            attributes={"deployment": deployment.config.name},
                        ):
                            await deployment.client.models.list()
                            deployment.healthy = True
                    except Exception as e:
                        deployment.healthy = False
                        trace.get_current_span().record_exception(e)

            await asyncio.sleep(self.health_check_interval)

    def _select_deployment(self) -> Deployment:
        """
        Select a deployment using power of two random choices algorithm,
        picking the one with lower utilization.
        """
        with tracer.start_as_current_span("select_deployment", kind=SpanKind.INTERNAL):
            healthy_deployments = [d for d in self.deployments if d.healthy]
            if not healthy_deployments:
                raise RuntimeError("No healthy deployments available")

            if len(healthy_deployments) == 1:
                return healthy_deployments[0]

            # Power of two random choices
            d1, d2 = random.sample(
                healthy_deployments, min(2, len(healthy_deployments))
            )

            # Use the maximum of token and request utilization as the load metric
            d1_load = max(
                d1.tpm_usage / d1.config.max_tpm, d1.rpm_usage / d1.config.max_rpm
            )
            d2_load = max(
                d2.tpm_usage / d2.config.max_tpm, d2.rpm_usage / d2.config.max_rpm
            )

            span = trace.get_current_span()
            span.set_attribute("d1_load", d1_load)
            span.set_attribute("d2_load", d2_load)

            # Return the deployment with lower load
            return d1 if d1_load <= d2_load else d2

    def __getattr__(self, name: str):
        """Dynamic dispatch to the selected deployment client"""

        deployment = self._select_deployment()
        print(f"Using {deployment}")
        return getattr(deployment.client, name)

    async def chat_completion(self, **kwargs):
        """Chat completion"""
        deployment = self._select_deployment()
        if kwargs.get("stream", False):
            response = await deployment.client.chat.completions.create(**kwargs)
        else:
            response = await deployment.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
