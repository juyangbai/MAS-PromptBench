import pytest

from core import settings

EXTRA_BODY = {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}}


def test_defaults(clean_env):
    assert settings.base_url() == "http://localhost:8000/v1"
    assert settings.model_id() == "Qwen/Qwen3.5-9B"
    assert settings.api_key() == "EMPTY"
    assert settings.api_key(blank_as_unset=True) == "EMPTY"
    assert settings.temperature() == 0.0
    assert settings.top_p() == 0.9
    assert settings.max_tokens() == 32768
    assert settings.request_seed() == 0


def test_values_are_read_at_call_time(clean_env):
    clean_env.setenv("TASK_MODEL_TEMPERATURE", "0.2")
    assert settings.temperature() == 0.2
    clean_env.setenv("TASK_MODEL_TEMPERATURE", "0.0")
    assert settings.temperature() == 0.0
    clean_env.setenv("MODEL_ID", "m")
    clean_env.setenv("VLLM_BASE_URL", "http://h/v1")
    assert (settings.model_id(), settings.base_url()) == ("m", "http://h/v1")


def test_blank_api_key(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "")
    assert settings.api_key() == ""
    assert settings.api_key(blank_as_unset=True) == "EMPTY"
    clean_env.setenv("OPENAI_API_KEY", "k")
    assert settings.api_key() == settings.api_key(blank_as_unset=True) == "k"


def test_decoding_defaults(clean_env):
    decoding = settings.decoding()
    assert decoding == {"temperature": 0.0, "top_p": 0.9, "seed": 0, "max_tokens": 32768, "extra_body": EXTRA_BODY}
    assert list(decoding) == ["temperature", "top_p", "seed", "max_tokens", "extra_body"]


def test_decoding_from_env(custom_env):
    assert settings.decoding() == {
        "temperature": 0.2,
        "top_p": 0.95,
        "seed": 7,
        "max_tokens": 1024,
        "extra_body": EXTRA_BODY,
    }


def test_decoding_explicit_seed_and_no_cap(custom_env):
    assert settings.decoding(seed=0)["seed"] == 0
    assert settings.decoding(seed=3)["seed"] == 3
    assert "max_tokens" not in settings.decoding(include_max_tokens=False)


def test_extra_body_is_fresh(clean_env):
    first = settings.extra_body()
    first["chat_template_kwargs"]["enable_thinking"] = True
    assert settings.extra_body() == EXTRA_BODY


@pytest.mark.parametrize(
    ("func", "name", "default"),
    [
        (settings.independent_n_agents, "INDEPENDENT_N_AGENTS", 4),
        (settings.decentralized_n_agents, "DECENTRALIZED_N_AGENTS", 4),
        (settings.decentralized_n_rounds, "DECENTRALIZED_N_ROUNDS", 2),
    ],
)
def test_team_shape(clean_env, func, name, default):
    assert func() == default
    assert func(8) == 8
    clean_env.setenv(name, "6")
    assert func() == func(8) == 6
    assert func(dataset="toolhop") == 6
    clean_env.setenv(f"TOOLHOP_{name}", "3")
    assert func(dataset="toolhop") == 3
    assert func(dataset="apibank") == 6
    assert func() == 6
    clean_env.delenv(name)
    clean_env.setenv(f"APIBANK_{name}", "5")
    assert func(10, dataset="apibank") == 5
    assert func(10) == 10
