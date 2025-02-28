import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import opentelemetry.trace as trace
from openai import AsyncAzureOpenAI
from opentelemetry.trace import SpanKind
from pydantic import BaseModel, Field, HttpUrl

# Tracer for OpenTelemetry
tracer = trace.get_tracer("switchboard")


class AzureDeployment(BaseModel):
    """Azure deployment configuration"""

    name: str
    api_base: HttpUrl
    api_key: str
    api_version: str = "2024-10-21"
    timeout: float = 60.0
    max_tpm: int = Field(..., description="Tokens per minute limit")
    max_rpm: int = Field(..., description="Requests per minute limit")


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
        client_factory: Callable[[AzureDeployment], Any] = default_client_factory,
        health_check_interval: int = 30,
        auto_start_health_checks: bool = True,
    ):
        self.client_factory = client_factory
        self.deployments = [
            Deployment(config=d, client=client_factory(d)) for d in deployments
        ]
        self.health_check_interval = health_check_interval
        self._health_check_task = None

        if auto_start_health_checks:
            self.start_health_checks()

    def start_health_checks(self):
        """Start background health check task"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    def stop_health_checks(self):
        """Stop the health check task"""
        if self._health_check_task:
            self._health_check_task.cancel()
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
                            attributes={
                                "deployment": deployment.config.deployment_name
                            },
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
            d1_load = max(d1.token_utilization, d1.request_utilization)
            d2_load = max(d2.token_utilization, d2.request_utilization)

            span = trace.get_current_span()
            span.set_attribute("d1_load", d1_load)
            span.set_attribute("d2_load", d2_load)

            # Return the deployment with lower load
            return d1 if d1_load <= d2_load else d2

    def __getattr__(self, name: str):
        """Dynamic dispatch to the selected deployment client"""

        async def method(*args, **kwargs):
            with tracer.start_as_current_span(
                f"azure_openai.{name}", kind=SpanKind.CLIENT
            ) as span:
                # Select deployment before each call
                deployment = self._select_deployment()

                span.set_attribute("deployment", deployment.config.deployment_name)
                span.set_attribute("endpoint", deployment.config.endpoint)

                # Get the method from the client
                client_method = getattr(deployment.client, name)

                # Track the request
                deployment.rpm_usage += 1

                try:
                    # Call the method
                    result = await client_method(*args, **kwargs)

                    # Handle token tracking for chat.completions.create
                    if name == "chat.completions.create":
                        # Extract token counts from the response
                        input_tokens = result.usage.prompt_tokens
                        output_tokens = result.usage.completion_tokens
                        total_tokens = result.usage.total_tokens

                        # Update metrics
                        deployment.tpm_usage += total_tokens

                        # Record telemetry
                        span.set_attribute("input_tokens", input_tokens)
                        span.set_attribute("output_tokens", output_tokens)
                        span.set_attribute("total_tokens", total_tokens)

                    return result
                except Exception as e:
                    # Mark deployment as unhealthy on error
                    deployment.healthy = False
                    span.record_exception(e)
                    raise

        return method
