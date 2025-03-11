from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from azure_switchboard import Deployment, ModelState


def _make_model_config() -> dict[str, ModelState]:
    return {
        "gpt-4o-mini": ModelState(
            tpm=10000,
            rpm=60,
        ),
        "gpt-4o": ModelState(
            tpm=10000,
            rpm=60,
        ),
    }


TEST_DEPLOYMENT_1 = Deployment(
    name="test1",
    api_base="https://test1.openai.azure.com/",
    api_key="test1",
    models=_make_model_config(),
)

TEST_DEPLOYMENT_2 = Deployment(
    name="test2",
    api_base="https://test2.openai.azure.com/",
    api_key="test2",
    models=_make_model_config(),
)

TEST_DEPLOYMENT_3 = Deployment(
    name="test3",
    api_base="https://test3.openai.azure.com/",
    api_key="test3",
    models=_make_model_config(),
)


MOCK_STREAM_CHUNKS = [
    ChatCompletionChunk(
        id="test_chunk_1",
        choices=[
            ChunkChoice(
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
            ChunkChoice(
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
            ChunkChoice(
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
            ChunkChoice(
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

MOCK_COMPLETION = ChatCompletion(
    id="test",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(
                content="Hello, world!",
                role="assistant",
            ),
        )
    ],
    created=1234567890,
    model="gpt-4o-mini",
    object="chat.completion",
    usage=CompletionUsage(
        completion_tokens=3,
        prompt_tokens=8,
        total_tokens=11,
    ),
)
