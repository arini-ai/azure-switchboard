from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.completion_usage import CompletionUsage

MOCK_STREAM_CHUNKS = [
    ChatCompletionChunk(
        id="test_chunk_1",
        choices=[
            Choice(
                delta=ChoiceDelta(content="Hello", role="assistant"),
                finish_reason=None,
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=None,
    ),
    ChatCompletionChunk(
        id="test_chunk_2",
        choices=[
            Choice(
                delta=ChoiceDelta(content=", "),
                finish_reason=None,
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=None,
    ),
    ChatCompletionChunk(
        id="test_chunk_3",
        choices=[
            Choice(
                delta=ChoiceDelta(content="world!"),
                finish_reason=None,
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=None,
    ),
    ChatCompletionChunk(
        id="test_chunk_4",
        choices=[
            Choice(
                delta=ChoiceDelta(),
                finish_reason="stop",
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
        usage=CompletionUsage(
            completion_tokens=5,
            prompt_tokens=15,
            total_tokens=20,
        ),
    ),
]


MOCK_COMPLETION_JSON = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": None,
            "message": {
                "content": "Hello! How can I assist you today?",
                "refusal": None,
                "role": "assistant",
            },
        }
    ],
    "created": 1741124380,
    "id": "chatcmpl-test",
    "model": "gpt-4o-mini",
    "object": "chat.completion",
    "service_tier": "default",
    "system_fingerprint": "fp_06737a9306",
    "usage": {
        "completion_tokens": 10,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens": 8,
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        "total_tokens": 18,
    },
}

MOCK_COMPLETION = ChatCompletion.model_validate(MOCK_COMPLETION_JSON)
