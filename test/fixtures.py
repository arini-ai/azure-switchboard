# Create mock streaming chunks for testing
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from switchboard import Deployment

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
        completion_tokens=10,
        prompt_tokens=20,
        total_tokens=30,
    ),
)

TEST_DEPLOYMENT_3 = Deployment(
    name="test3",
    api_base="https://test3.openai.azure.com/",
    api_key="test3",
    tpm_ratelimit=1000,
    rpm_ratelimit=6,
)

TEST_DEPLOYMENT_2 = Deployment(
    name="test2",
    api_base="https://test2.openai.azure.com/",
    api_key="test2",
    tpm_ratelimit=1000,
    rpm_ratelimit=6,
)

TEST_DEPLOYMENT_1 = Deployment(
    name="test1",
    api_base="https://test1.openai.azure.com/",
    api_key="test1",
    tpm_ratelimit=1000,
    rpm_ratelimit=6,
)

BASIC_CHAT_COMPLETION_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}
