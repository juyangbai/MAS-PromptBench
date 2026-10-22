import pytest

RUNNER_ENV = (
    "VLLM_BASE_URL",
    "MODEL_ID",
    "OPENAI_API_KEY",
    "TASK_MODEL_TEMPERATURE",
    "TASK_MODEL_TOP_P",
    "TASK_MODEL_MAX_TOKENS",
    "REQUEST_SEED",
    "INDEPENDENT_N_AGENTS",
    "DECENTRALIZED_N_AGENTS",
    "DECENTRALIZED_N_ROUNDS",
    "TOOLHOP_INDEPENDENT_N_AGENTS",
    "TOOLHOP_DECENTRALIZED_N_AGENTS",
    "TOOLHOP_DECENTRALIZED_N_ROUNDS",
    "APIBANK_INDEPENDENT_N_AGENTS",
    "APIBANK_DECENTRALIZED_N_AGENTS",
    "APIBANK_DECENTRALIZED_N_ROUNDS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """No runner variable set."""
    for name in RUNNER_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def custom_env(clean_env):
    """Every endpoint / decoding variable set to a non-default value."""
    values = {
        "VLLM_BASE_URL": "http://127.0.0.1:9999/v1",
        "MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",
        "OPENAI_API_KEY": "key-123",
        "TASK_MODEL_TEMPERATURE": "0.2",
        "TASK_MODEL_TOP_P": "0.95",
        "TASK_MODEL_MAX_TOKENS": "1024",
        "REQUEST_SEED": "7",
    }
    for name, value in values.items():
        clean_env.setenv(name, value)
    return clean_env
