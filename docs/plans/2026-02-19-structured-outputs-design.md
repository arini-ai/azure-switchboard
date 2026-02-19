# Structured Outputs via `parse()` — Design

## Summary

Add a `parse()` method to `Switchboard` and `DeploymentState` that mirrors the OpenAI SDK's `client.beta.chat.completions.parse()` API. This gives users typed structured outputs with full load-balancing, failover, session affinity, and usage tracking — the same guarantees as `create()`.

## Motivation

The OpenAI SDK's `beta.chat.completions.parse()` method accepts a Pydantic `BaseModel` as `response_format` and returns a `ParsedChatCompletion[T]` with automatic JSON schema generation and response parsing. This is the standard way to get structured outputs from OpenAI models.

While switchboard's `create()` already forwards `**kwargs` (so raw `response_format` dicts work), there is no way to use the typed `parse()` API with its generic return type and SDK-level validation through switchboard today.

## Approach

**Parallel methods** — add `parse()` alongside `create()` at both layers, following the same pattern. No refactoring of existing code.

## Design

### Call flow

```
switchboard.parse(model=..., response_format=MyModel, ...)
  └─ failover loop (tenacity AsyncRetrying, same as create)
       └─ select_deployment(model=..., session_id=...)
            └─ deployment.parse(model=..., response_format=MyModel, ...)
                 └─ client.beta.chat.completions.parse(...)
                      └─ returns ParsedChatCompletion[MyModel]
```

### `DeploymentState.parse()` — deployment.py

```python
async def parse(
    self,
    *,
    model: str,
    response_format: type[T],
    **kwargs,
) -> ParsedChatCompletion[T]:
```

Implementation mirrors the non-streaming path of `create()`:

1. Validate model is configured, raise `SwitchboardError` if not
2. Estimate input tokens, pre-spend for concurrent utilization tracking
3. Set default timeout from deployment config
4. Call `self.client.beta.chat.completions.parse(model=model, response_format=response_format, **kwargs)`
5. Track actual usage (adjust for preflight estimate), set OTel span attributes
6. Handle `RateLimitError` → `mark_down()` + raise `SwitchboardError`
7. Handle other exceptions → `mark_down()` + raise `RuntimeError`
8. Return `ParsedChatCompletion[T]`

No streaming support — `parse()` is non-streaming only.

### `Switchboard.parse()` — switchboard.py

```python
async def parse(
    self,
    *,
    model: str,
    response_format: type[T],
    session_id: str | None = None,
    **kwargs,
) -> ParsedChatCompletion[T]:
```

Implementation mirrors `create()` non-streaming path: failover loop → select deployment → delegate to `deployment.parse()` → count request metric → return.

### Type parameter

Python >=3.10 compatible using `TypeVar`:

```python
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
```

Gives full type inference: `switchboard.parse(response_format=CalendarEvent)` → `ParsedChatCompletion[CalendarEvent]`.

### Exports

Re-export `ParsedChatCompletion` from `__init__.py` for user convenience.

### Return type

`ParsedChatCompletion[T]` — the full OpenAI response object. Maintains the "API-compatible drop-in replacement" contract. Users access:

- `.choices[0].message.parsed` → `T | None` (the parsed model)
- `.usage` → token counts
- `.choices[0].message.refusal` → refusal reason if applicable
- `.choices[0].finish_reason` → "stop", "length", etc.

## Testing

- Mock at HTTP level via `respx` (same `/chat/completions` endpoint)
- Test parse returns correct Pydantic model via `.choices[0].message.parsed`
- Test usage tracking (TPM/RPM counters update correctly)
- Test failover (rate limit → switches deployment)
- Test model cooldown on errors

## Non-goals

- Streaming parse (can be added later if needed)
- Simplified `T | None` return type wrapper
- Refactoring `create()` to share logic with `parse()`

## Dependencies

- `openai>=1.62.0` (already satisfied — `beta.chat.completions.parse()` available)
- No new runtime dependencies
