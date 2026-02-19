# Structured Outputs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `parse()` method to `Switchboard` and `DeploymentState` that wraps `client.beta.chat.completions.parse()` with load-balancing, failover, and usage tracking.

**Architecture:** Parallel `parse()` methods at both layers (DeploymentState and Switchboard), mirroring the existing `create()` pattern. No refactoring of existing code. TypeVar for generic return type.

**Tech Stack:** openai SDK (ParsedChatCompletion, beta.chat.completions.parse), pydantic BaseModel, tenacity for failover, respx/unittest.mock for tests.

**Design doc:** `docs/plans/2026-02-19-structured-outputs-design.md`

---

### Task 1: Add test fixtures for structured outputs

**Files:**

- Modify: `tests/conftest.py`

**Step 1: Add ParsedChatCompletion test fixtures to conftest.py**

Add imports and a test Pydantic model, a mock `ParsedChatCompletion` response, and a helper mock function at the end of `conftest.py`:

```python
# Add to imports at top:
from openai.types.chat import ParsedChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChoice, ParsedChatCompletionMessage
from pydantic import BaseModel as PydanticBaseModel

# Add test model and fixtures at bottom:

class WeatherResult(PydanticBaseModel):
    """Test Pydantic model for structured outputs."""
    city: str
    temperature: float
    unit: str

PARSED_COMPLETION_PARAMS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "response_format": WeatherResult,
}

PARSED_RESPONSE = ParsedChatCompletion[WeatherResult](
    id="chatcmpl-parsed-test",
    choices=[
        ParsedChoice[WeatherResult](
            finish_reason="stop",
            index=0,
            message=ParsedChatCompletionMessage[WeatherResult](
                content='{"city": "Paris", "temperature": 18.5, "unit": "celsius"}',
                role="assistant",
                parsed=WeatherResult(city="Paris", temperature=18.5, unit="celsius"),
                refusal=None,
            ),
        )
    ],
    created=1741124380,
    model="gpt-4o-mini",
    object="chat.completion",
    usage=CompletionUsage(
        completion_tokens=15,
        prompt_tokens=12,
        total_tokens=27,
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
    ),
)
```

**Step 2: Run existing tests to verify nothing is broken**

Run: `uv run pytest tests/ -v --timeout=10`
Expected: All existing tests PASS.

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add structured outputs fixtures for parse() tests"
```

---

### Task 2: Add `DeploymentState.parse()` — failing tests first

**Files:**

- Create: `tests/test_parse.py`

**Step 1: Write failing tests for `DeploymentState.parse()`**

Create `tests/test_parse.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response
from openai import RateLimitError

from azure_switchboard import SwitchboardError
from azure_switchboard.deployment import DeploymentState

from .conftest import (
    PARSED_COMPLETION_PARAMS,
    PARSED_RESPONSE,
    WeatherResult,
)


class TestDeploymentParse:
    """DeploymentState.parse() tests — mirrors TestDeployment completion tests."""

    async def test_parse_returns_parsed_completion(self, deployment: DeploymentState):
        """Test basic parse returns ParsedChatCompletion with correct parsed model."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ) as mock:
            response = await deployment.parse(**PARSED_COMPLETION_PARAMS)

            mock.assert_called_once()
            assert response == PARSED_RESPONSE
            assert response.choices[0].message.parsed is not None
            assert isinstance(response.choices[0].message.parsed, WeatherResult)
            assert response.choices[0].message.parsed.city == "Paris"

    async def test_parse_tracks_usage(self, deployment: DeploymentState):
        """Test that parse() updates TPM/RPM counters like create() does."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ):
            await deployment.parse(**PARSED_COMPLETION_PARAMS)

            model = deployment.model("gpt-4o-mini")
            usage = model.stats()
            assert usage.tpm.startswith(str(PARSED_RESPONSE.usage.total_tokens))
            assert usage.rpm.startswith("1")

    async def test_parse_invalid_model(self, deployment: DeploymentState):
        """Test that an unconfigured model raises SwitchboardError."""
        with pytest.raises(SwitchboardError, match="gpt-fake not configured"):
            await deployment.parse(
                model="gpt-fake",
                messages=[],
                response_format=WeatherResult,
            )

    async def test_parse_rate_limit_marks_down(self, deployment: DeploymentState):
        """Test that RateLimitError marks model down and raises SwitchboardError."""
        rate_limit_error = RateLimitError(
            "rate limited",
            response=Response(
                status_code=429,
                request=Request(
                    "POST",
                    "https://test.openai.azure.com/openai/v1/chat/completions",
                ),
            ),
            body={"error": {"message": "rate limited"}},
        )

        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=rate_limit_error),
        ):
            with pytest.raises(
                SwitchboardError,
                match="Rate limit exceeded in deployment parse",
            ):
                await deployment.parse(**PARSED_COMPLETION_PARAMS)

            assert not deployment.model("gpt-4o-mini").is_healthy()

    async def test_parse_exception_marks_down(self, deployment: DeploymentState):
        """Test that generic exceptions mark model down and raise RuntimeError."""
        with patch.object(
            deployment.client.beta.chat.completions,
            "parse",
            new=AsyncMock(side_effect=Exception("upstream error")),
        ):
            with pytest.raises(RuntimeError, match="Error in deployment parse"):
                await deployment.parse(**PARSED_COMPLETION_PARAMS)

            assert not deployment.model("gpt-4o-mini").is_healthy()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parse.py -v --timeout=10`
Expected: FAIL — `AttributeError: 'DeploymentState' object has no attribute 'parse'`

**Step 3: Commit failing tests**

```bash
git add tests/test_parse.py
git commit -m "test: add failing tests for DeploymentState.parse()"
```

---

### Task 3: Implement `DeploymentState.parse()`

**Files:**

- Modify: `src/azure_switchboard/deployment.py`

**Step 1: Add TypeVar and ParsedChatCompletion import**

Add to the imports at the top of `deployment.py`:

```python
# Add to existing typing import line:
from typing import Literal, TypeVar, cast, overload

# Add ParsedChatCompletion to the openai.types.chat import line:
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion

# Add after imports, before class definitions:
from pydantic import BaseModel as _BaseModel  # rename to avoid clash with existing BaseModel usage
_T = TypeVar("_T", bound=_BaseModel)
```

Note: `deployment.py` already imports `BaseModel` from pydantic (line 12) and uses it for the `Deployment` config class. The TypeVar needs to be bound to pydantic's `BaseModel`. Since `Deployment` already inherits from `BaseModel`, no new import is needed — just add the TypeVar using the existing `BaseModel` import:

```python
_T = TypeVar("_T", bound=BaseModel)
```

**Step 2: Add parse() method to DeploymentState**

Add after `create()` method (after line 167), before `_estimate_token_usage()`:

```python
    async def parse(
        self,
        *,
        model: str,
        response_format: type[_T],
        **kwargs,
    ) -> ParsedChatCompletion[_T]:
        """
        Send a structured output parse request to this client.
        Tracks usage metrics for load balancing.
        """

        if model not in self.models:
            raise SwitchboardError(f"{model} not configured for deployment")

        _preflight_estimate = self._estimate_token_usage(kwargs)
        self.models[model].spend_tokens(_preflight_estimate)
        self.models[model].spend_request()

        kwargs["timeout"] = kwargs.get("timeout", self.config.timeout)
        try:
            logger.trace("Creating parsed completion")
            response = await self.client.beta.chat.completions.parse(
                model=model, response_format=response_format, **kwargs
            )

            if response.usage:
                self.models[model].spend_tokens(
                    response.usage.total_tokens - _preflight_estimate
                )
                self._set_span_attributes(response.usage)

            return response
        except RateLimitError as e:
            logger.warning("Hit rate limits")
            self.models[model].mark_down()
            raise SwitchboardError(
                "Rate limit exceeded in deployment parse"
            ) from e
        except Exception as e:
            logger.exception("Marking down model for parse error")
            self.models[model].mark_down()
            raise RuntimeError("Error in deployment parse") from e
```

**Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_parse.py -v --timeout=10`
Expected: All 5 tests PASS.

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=10`
Expected: All tests PASS (existing + new).

**Step 5: Lint**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: Clean.

**Step 6: Commit**

```bash
git add src/azure_switchboard/deployment.py
git commit -m "feat: add DeploymentState.parse() for structured outputs"
```

---

### Task 4: Add `Switchboard.parse()` — failing tests first

**Files:**

- Modify: `tests/test_parse.py`

**Step 1: Add failing Switchboard.parse() tests**

Append to `tests/test_parse.py`:

```python
from unittest.mock import patch

from azure_switchboard import Switchboard

from .conftest import (
    PARSED_COMPLETION_PARAMS,
    PARSED_RESPONSE,
    WeatherResult,
    azure_config,
)


class TestSwitchboardParse:
    """Switchboard.parse() tests — mirrors TestSwitchboard completion tests."""

    async def test_parse(self, switchboard: Switchboard):
        """Test parse through switchboard with load balancing."""
        with patch(
            "azure_switchboard.deployment.DeploymentState.parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ) as mock:
            response = await switchboard.parse(**PARSED_COMPLETION_PARAMS)

            mock.assert_called_once()
            assert response == PARSED_RESPONSE
            assert response.choices[0].message.parsed.city == "Paris"

    async def test_parse_session_affinity(self, switchboard: Switchboard):
        """Test that session_id routes to same deployment."""
        with patch(
            "azure_switchboard.deployment.DeploymentState.parse",
            new=AsyncMock(return_value=PARSED_RESPONSE),
        ):
            await switchboard.parse(session_id="test-session", **PARSED_COMPLETION_PARAMS)
            deployment_1 = switchboard.sessions["test-session"]

            await switchboard.parse(session_id="test-session", **PARSED_COMPLETION_PARAMS)
            deployment_2 = switchboard.sessions["test-session"]

            assert deployment_1.name == deployment_2.name

    async def test_parse_failover(self, switchboard: Switchboard):
        """Test that parse fails over to another deployment on error."""
        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first attempt fails")
            return PARSED_RESPONSE

        with patch(
            "azure_switchboard.deployment.DeploymentState.parse",
            new=AsyncMock(side_effect=failing_then_success),
        ):
            response = await switchboard.parse(**PARSED_COMPLETION_PARAMS)
            assert response == PARSED_RESPONSE
            assert call_count == 2

    async def test_parse_invalid_model(self, switchboard: Switchboard):
        """Test that an invalid model raises SwitchboardError."""
        with pytest.raises(
            SwitchboardError,
            match="No eligible deployments available for invalid-model",
        ):
            await switchboard.parse(
                model="invalid-model",
                messages=[],
                response_format=WeatherResult,
            )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parse.py::TestSwitchboardParse -v --timeout=10`
Expected: FAIL — `AttributeError: 'Switchboard' object has no attribute 'parse'`

**Step 3: Commit failing tests**

```bash
git add tests/test_parse.py
git commit -m "test: add failing tests for Switchboard.parse()"
```

---

### Task 5: Implement `Switchboard.parse()`

**Files:**

- Modify: `src/azure_switchboard/switchboard.py`

**Step 1: Add TypeVar and ParsedChatCompletion import**

Add to imports at top of `switchboard.py`:

```python
# Add to existing typing import:
from typing import Callable, Literal, Sequence, TypeVar, overload

# Add ParsedChatCompletion to existing openai.types.chat import:
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion

# Add pydantic import:
from pydantic import BaseModel

# Add TypeVar after imports:
_T = TypeVar("_T", bound=BaseModel)
```

**Step 2: Add parse() method to Switchboard class**

Add after the `create()` method (after line 207), before `__repr__`:

```python
    async def parse(
        self,
        *,
        model: str,
        response_format: type[_T],
        session_id: str | None = None,
        **kwargs,
    ) -> ParsedChatCompletion[_T]:
        """
        Send a structured output parse request to the selected deployment, with automatic failover.
        """
        with logger.contextualize(model=model, session_id=session_id):
            async for attempt in self.failover_policy:
                with attempt:
                    deployment = self.select_deployment(
                        model=model, session_id=session_id
                    )
                    with logger.contextualize(deployment=deployment.name):
                        logger.trace("Sending parse request")
                        response = await deployment.parse(
                            model=model,
                            response_format=response_format,
                            **kwargs,
                        )
                    request_counter.add(
                        1, {"model": model, "deployment": deployment.name}
                    )
                    return response
```

**Step 3: Run parse tests**

Run: `uv run pytest tests/test_parse.py -v --timeout=10`
Expected: All tests PASS (deployment + switchboard).

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=10`
Expected: All tests PASS.

**Step 5: Lint**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: Clean.

**Step 6: Commit**

```bash
git add src/azure_switchboard/switchboard.py
git commit -m "feat: add Switchboard.parse() for structured outputs"
```

---

### Task 6: Update exports

**Files:**

- Modify: `src/azure_switchboard/__init__.py`

**Step 1: Add ParsedChatCompletion re-export**

Update `__init__.py` to re-export `ParsedChatCompletion`:

```python
from loguru import logger as _logger

from openai.types.chat import ParsedChatCompletion

from .deployment import Deployment
from .exceptions import SwitchboardError
from .model import Model
from .switchboard import Switchboard

# As a library, do not configure sinks or emit logs by default.
# Applications can opt in explicitly.
_logger.disable("azure_switchboard")


__all__ = [
    "Deployment",
    "Model",
    "ParsedChatCompletion",
    "SwitchboardError",
    "Switchboard",
]
```

**Step 2: Write test for the export**

Check if `tests/test_init.py` exists and add a test. If it already tests exports, add `ParsedChatCompletion` to the expected set.

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=10`
Expected: All tests PASS.

**Step 4: Lint**

Run: `uv run ruff check . --fix && uv run ruff format .`
Expected: Clean.

**Step 5: Commit**

```bash
git add src/azure_switchboard/__init__.py tests/test_init.py
git commit -m "feat: export ParsedChatCompletion from azure_switchboard"
```

---

### Task 7: Final verification

**Step 1: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --timeout=10`
Expected: All tests PASS.

**Step 2: Run linting**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean.

**Step 3: Verify imports work end-to-end**

Run: `uv run python -c "from azure_switchboard import Switchboard, ParsedChatCompletion; print('OK')"`
Expected: `OK`
