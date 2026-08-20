"""Traces and metrics for routing decisions.

Switchboard's job is choosing where a request goes, so what is worth recording
is the choice: which resource served it, how loaded it was, what failed, and
what that failure took out of rotation. The vendor SDKs' own instrumentation
covers the wire call underneath.

A deployment's name is the model name and is unique within a resource, so
`{model, resource}` identifies one deployment and no other dimension is needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from opentelemetry import metrics, trace
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.trace import Span, Status, StatusCode

if TYPE_CHECKING:
    from .deployment import ModelDeployment
    from .switchboard import Switchboard

tracer = trace.get_tracer("azure_switchboard")
meter = metrics.get_meter("azure_switchboard")

requests = meter.create_counter(
    name="switchboard.requests",
    description="Requests dispatched to a deployment, by outcome",
    unit="{request}",
)
tokens = meter.create_counter(
    name="switchboard.tokens",
    description="Tokens charged against a deployment's quota, by kind",
    unit="{token}",
)
failovers = meter.create_counter(
    name="switchboard.failovers",
    description="Retries onto another deployment, by what the last one raised",
    unit="{failover}",
)
cooldowns = meter.create_counter(
    name="switchboard.cooldowns",
    description="Deployments or resources taken out of rotation, by cause",
    unit="{cooldown}",
)
duration = meter.create_histogram(
    name="switchboard.request.duration",
    description="Time one attempt spent in the upstream call",
    unit="s",
)


# The gauges read the switchboard that is currently serving. One switchboard
# lives as long as the app around it, so this is a reference to a live object
# rather than a registry needing cleanup -- and the OTel API has no way to
# unregister an observable gauge's callback anyway.
#
# A second switchboard takes the gauges over rather than being added to them:
# summing two pools would report one pool's spare capacity as the other's
# health, and a pool with nothing healthy left is the reading that matters.
_tracked: Switchboard | None = None


def track(switchboard: Switchboard) -> None:
    """Point the observable gauges at a switchboard."""
    global _tracked
    _tracked = switchboard


def _deployments() -> Iterable[tuple[str, ModelDeployment]]:
    if _tracked is None:
        return
    for resource in _tracked._all_resources():
        for deployment in resource.models.values():
            yield resource.name, deployment


def _observe_utilization(options: CallbackOptions) -> Iterable[Observation]:
    for resource, deployment in _deployments():
        yield Observation(
            deployment.load, {"model": deployment.name, "resource": resource}
        )


def _observe_healthy(options: CallbackOptions) -> Iterable[Observation]:
    # counted rather than yielded per deployment: zero healthy deployments of a
    # model is the value worth alerting on, and it is only visible in the sum
    counts: dict[str, int] = {}
    for _, deployment in _deployments():
        counts[deployment.name] = (
            counts.get(deployment.name, 0) + deployment.is_healthy()
        )
    for model, healthy in counts.items():
        yield Observation(healthy, {"model": model})


meter.create_observable_gauge(
    name="switchboard.utilization",
    description="Fraction of a deployment's tighter quota in use; 1 while cooling",
    unit="1",
    callbacks=[_observe_utilization],
)
meter.create_observable_gauge(
    name="switchboard.deployments.healthy",
    description="Deployments of a model currently eligible for selection",
    unit="{deployment}",
    callbacks=[_observe_healthy],
)


def start_stream_span(deployment: ModelDeployment) -> Span:
    """Open a span for a stream that outlives the call which opened it.

    A stream's tokens are only known once the last chunk lands, well after
    dispatch returned and the attempt span closed. Started here, while the
    attempt span is still current, so it is parented to it; ended by whoever
    drains the stream, so its duration is time to last chunk.
    """
    return tracer.start_span(
        "stream",
        attributes={
            "switchboard.resource": deployment.resource.name,
            "switchboard.model": deployment.name,
        },
    )


def end_stream_span(span: Span, exc: BaseException | None = None) -> None:
    """Close a stream span, recording whatever ended it."""
    if exc is not None:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.end()


def record_usage(
    deployment: ModelDeployment,
    span: Span | None = None,
    *,
    input: int | None = None,
    output: int | None = None,
    cached: int | None = None,
    reasoning: int | None = None,
) -> None:
    """Record what a response cost, on both the span and the token counter.

    Both providers report cached and reasoning tokens under their own field
    names; deployments extract, this normalizes. `cached` and `reasoning` are
    subsets of input and output respectively, so they are their own series
    rather than addends.

    `span` is explicit because a stream's usage lands long after the span that
    opened it stopped being current.
    """
    span = span or trace.get_current_span()
    attributes = {"model": deployment.name, "resource": deployment.resource.name}

    for kind, count in (
        ("input", input),
        ("output", output),
        ("cached", cached),
        ("reasoning", reasoning),
    ):
        if count:
            tokens.add(count, {**attributes, "kind": kind})

    if cached:
        span.set_attribute("gen_ai.usage.cached_tokens", cached)
    if reasoning:
        span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning)
