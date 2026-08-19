"""What a routing decision looks like from the outside.

The instruments in azure_switchboard.telemetry are created at import time
against the API's proxy provider, so the SDK providers here are installed once
per process and the exporters drained between tests.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response
from openai import APIConnectionError, RateLimitError
from openai.resources.chat.completions import AsyncCompletions
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from azure_switchboard import Foundry, OpenAIDeployment, Switchboard, SwitchboardError

from .conftest import (
    COMPLETION_BODY,
    COMPLETION_PARAMS,
    COMPLETION_RESPONSE,
    chat_completion_mock,
)

_exporter = InMemorySpanExporter()
_reader = InMemoryMetricReader()

# A provider can only be installed once per process, and xdist gives each
# worker its own, so this runs at import rather than in a fixture.
trace.set_tracer_provider(
    TracerProvider(active_span_processor=SimpleSpanProcessor(_exporter))
)
metrics.set_meter_provider(MeterProvider(metric_readers=[_reader]))


@pytest.fixture(autouse=True)
def spans() -> InMemorySpanExporter:
    _exporter.clear()
    return _exporter


def named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def points(instrument: str, **match) -> list:
    """Data points of an instrument whose attributes include `match`."""
    found = []
    for resource in _reader.get_metrics_data().resource_metrics:  # pyright: ignore[reportOptionalMemberAccess]
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name != instrument:
                    continue
                for point in metric.data.data_points:
                    attrs = dict(point.attributes or {})
                    if all(attrs.get(k) == v for k, v in match.items()):
                        found.append(point)
    return found


def total(instrument: str, **match) -> float:
    return sum(p.value for p in points(instrument, **match))


def rate_limited_once():
    """The first deployment called returns a 429, the next one answers.

    Patched at the SDK boundary rather than on OpenAIDeployment.create, so the
    deployment's own error handling -- the thing that records the cooldown --
    actually runs.
    """
    error = RateLimitError(
        "rate limited",
        response=Response(
            status_code=429,
            request=Request("POST", "https://test.openai.azure.com/"),
        ),
        body=None,
    )
    # explicitly async: the SDK wraps create() in a sync decorator, so
    # patch.object would otherwise hand back a MagicMock nothing can await
    return patch.object(
        AsyncCompletions,
        "create",
        new=AsyncMock(side_effect=[error, COMPLETION_RESPONSE]),
    )


class TestDispatchSpans:
    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_a_call_is_one_span_with_one_attempt_under_it(
        self, switchboard: Switchboard, spans: InMemorySpanExporter
    ):
        await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        (call,) = named(spans, "chat gpt-4o-mini")
        (attempt,) = named(spans, "attempt")

        assert call.attributes["gen_ai.request.model"] == "gpt-4o-mini"  # pyright: ignore[reportOptionalSubscript]
        assert attempt.parent is not None
        assert attempt.parent.span_id == call.context.span_id  # pyright: ignore[reportOptionalMemberAccess]
        assert attempt.attributes["switchboard.attempt"] == 0  # pyright: ignore[reportOptionalSubscript]
        assert attempt.attributes["switchboard.resource"] in switchboard.resources  # pyright: ignore[reportOptionalSubscript]

    async def test_a_session_id_is_recorded_on_the_call(
        self, switchboard: Switchboard, spans: InMemorySpanExporter
    ):
        with patch.object(
            OpenAIDeployment, "create", side_effect=chat_completion_mock()
        ):
            await switchboard.chat.completions.create(
                session_id="abc", **COMPLETION_PARAMS
            )

        (call,) = named(spans, "chat gpt-4o-mini")
        assert call.attributes["switchboard.session_id"] == "abc"  # pyright: ignore[reportOptionalSubscript]

    async def test_a_failover_is_two_attempts_and_the_failure_is_on_the_first(
        self, switchboard: Switchboard, spans: InMemorySpanExporter
    ):
        with rate_limited_once():
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        (call,) = named(spans, "chat gpt-4o-mini")
        first, second = sorted(
            named(spans, "attempt"),
            key=lambda s: s.attributes["switchboard.attempt"],  # pyright: ignore[reportOptionalSubscript]
        )

        assert first.status.status_code is StatusCode.ERROR
        assert first.events[0].name == "exception"
        assert second.status.status_code is not StatusCode.ERROR
        # the call as a whole succeeded, so only the attempt is marked failed
        assert call.status.status_code is not StatusCode.ERROR

    async def test_an_unroutable_model_never_opens_an_attempt(
        self, switchboard: Switchboard, spans: InMemorySpanExporter
    ):
        with pytest.raises(SwitchboardError):
            await switchboard.chat.completions.create(
                model="nonexistent", messages=COMPLETION_PARAMS["messages"]
            )

        assert named(spans, "chat nonexistent")
        assert not named(spans, "attempt")


class TestStreamSpans:
    async def test_the_stream_span_outlives_the_attempt_that_opened_it(
        self, deployment: OpenAIDeployment, spans: InMemorySpanExporter
    ):
        """Usage lands on the last chunk, long after create() returned. A span
        that closed with the attempt could not record it."""
        with patch.object(
            deployment.client.chat.completions,
            "create",
            side_effect=chat_completion_mock(),
        ):
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            assert not named(spans, "stream"), "ended before the first chunk"

            async for _ in stream:
                pass

        (span,) = named(spans, "stream")
        assert span.attributes["switchboard.model"] == "gpt-4o-mini"  # pyright: ignore[reportOptionalSubscript]
        # only reachable because the span was still open when the usage chunk landed
        assert span.attributes["gen_ai.usage.cached_tokens"] > 0  # pyright: ignore[reportOptionalSubscript, reportOperatorIssue]

    async def test_the_stream_span_hangs_off_the_attempt_that_opened_it(
        self, switchboard: Switchboard, spans: InMemorySpanExporter
    ):
        with patch.object(
            AsyncCompletions,
            "create",
            new=AsyncMock(side_effect=chat_completion_mock()),
        ):
            stream = await switchboard.chat.completions.create(
                stream=True, **COMPLETION_PARAMS
            )
            async for _ in stream:
                pass

        (attempt,) = named(spans, "attempt")
        (stream_span,) = named(spans, "stream")
        assert stream_span.parent is not None
        assert stream_span.parent.span_id == attempt.context.span_id  # pyright: ignore[reportOptionalMemberAccess]
        # and it is still the longer of the two: the attempt only opened it
        assert stream_span.end_time > attempt.end_time  # pyright: ignore[reportOptionalOperand]

    async def test_a_stream_that_fails_mid_body_closes_its_span_as_an_error(
        self, deployment: OpenAIDeployment, spans: InMemorySpanExporter
    ):
        error = APIConnectionError(request=Request("POST", "https://test/"))

        async def failing():
            raise error
            yield  # pragma: no cover

        with patch.object(
            deployment.client.chat.completions,
            "create",
            new=AsyncMock(return_value=failing()),
        ):
            stream = await deployment.create(stream=True, **COMPLETION_BODY)
            with pytest.raises(APIConnectionError):
                async for _ in stream:
                    pass

        (span,) = named(spans, "stream")
        assert span.status.status_code is StatusCode.ERROR


class TestMetrics:
    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_a_served_request_is_counted_against_its_resource(
        self, switchboard: Switchboard
    ):
        before = total("switchboard.requests", model="gpt-4o-mini", outcome="success")
        await switchboard.chat.completions.create(**COMPLETION_PARAMS)
        after = total("switchboard.requests", model="gpt-4o-mini", outcome="success")

        assert after == before + 1
        assert points("switchboard.request.duration", outcome="success")

    async def test_a_rate_limit_counts_a_failover_and_a_cooldown(
        self, switchboard: Switchboard
    ):
        # every dimension the assertion filters on has to be in the baseline
        # too, or unrelated series leak into the comparison
        failed_over = total(
            "switchboard.failovers", model="gpt-4o-mini", reason="ratelimit"
        )
        cooled = total("switchboard.cooldowns", reason="ratelimit")

        with rate_limited_once():
            await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        assert (
            total("switchboard.failovers", model="gpt-4o-mini", reason="ratelimit")
            == failed_over + 1
        )
        assert total("switchboard.cooldowns", reason="ratelimit") == cooled + 1

    @pytest.mark.mock_models("gpt-4o-mini")
    async def test_tokens_are_counted_by_kind(self, switchboard: Switchboard):
        before = total("switchboard.tokens", model="gpt-4o-mini", kind="input")
        await switchboard.chat.completions.create(**COMPLETION_PARAMS)

        assert total("switchboard.tokens", model="gpt-4o-mini", kind="input") > before
        # cached is its own series, not an addend of input
        assert total("switchboard.tokens", model="gpt-4o-mini", kind="cached") >= 5


class TestObservableGauges:
    """Utilization is state, so it is read at collection time rather than
    written from the request path."""

    async def test_utilization_is_reported_per_deployment(self):
        resource = Foundry(
            name="gauge-util",
            api_key="k",
            models=[OpenAIDeployment(name="gauge-model", tpm=1000, rpm=100)],
        )
        async with Switchboard([resource], ratelimit_window=0):
            resource.models["gauge-model"].spend_tokens(250)

            (point,) = points("switchboard.utilization", resource="gauge-util")
            assert point.value == 0.25

    async def test_a_cooling_resource_reports_no_healthy_deployments(self):
        resource = Foundry(
            name="gauge-health",
            api_key="k",
            models=[OpenAIDeployment(name="gauge-health-model", tpm=1000, rpm=100)],
        )
        async with Switchboard([resource], ratelimit_window=0):
            assert (
                total("switchboard.deployments.healthy", model="gauge-health-model")
                == 1
            )

            resource.mark_down()
            assert (
                total("switchboard.deployments.healthy", model="gauge-health-model")
                == 0
            )
            assert total("switchboard.utilization", model="gauge-health-model") == 1

    async def test_a_discarded_switchboard_stops_being_reported(self):
        resource = Foundry(
            name="gauge-gone",
            api_key="k",
            models=[OpenAIDeployment(name="gauge-gone-model", tpm=1000, rpm=100)],
        )
        sb = Switchboard([resource], ratelimit_window=0)
        assert points("switchboard.utilization", resource="gauge-gone")

        del sb
        import gc

        gc.collect()
        assert not points("switchboard.utilization", resource="gauge-gone")
