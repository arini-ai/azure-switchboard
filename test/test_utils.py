from switchboard import Deployment

TEST_DEPLOYMENT_1 = Deployment(
    name="test1",
    api_base="https://test1.openai.azure.com/",
    api_key="test1",
    tpm_ratelimit=100,
    rpm_ratelimit=100,
)
TEST_DEPLOYMENT_2 = Deployment(
    name="test2",
    api_base="https://test2.openai.azure.com/",
    api_key="test2",
    tpm_ratelimit=200,
    rpm_ratelimit=200,
)
TEST_DEPLOYMENT_3 = Deployment(
    name="test3",
    api_base="https://test3.openai.azure.com/",
    api_key="test3",
    tpm_ratelimit=300,
    rpm_ratelimit=300,
)

TEST_DEPLOYMENTS = [TEST_DEPLOYMENT_1, TEST_DEPLOYMENT_2, TEST_DEPLOYMENT_3]

BASIC_CHAT_COMPLETION_ARGS = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}
