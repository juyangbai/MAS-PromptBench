"""Shared real-topology module adapter for non-BFCL datasets."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.paths import PROMPTS_DIR
from optimizers.bridge.adapters.gpqa_common import coerce_instance, default_chat_model
from optimizers.bridge.lm import TASK_MODEL, next_task_endpoint, task_max_tokens, task_request_seed, task_temperature
from optimizers.bridge.output_contracts import append_output_contract

_MODULE_LOCKS: dict[str, threading.Lock] = {}
_MODULE_LOCKS_GUARD = threading.Lock()


def module_lock(module_name: str) -> threading.Lock:
    with _MODULE_LOCKS_GUARD:
        if module_name not in _MODULE_LOCKS:
            _MODULE_LOCKS[module_name] = threading.Lock()
        return _MODULE_LOCKS[module_name]


def prompt_path(topology: str, dataset: str, role: str) -> Path:
    return PROMPTS_DIR / topology / dataset / f"{role}.txt"


def load_prompts(
    dataset: str,
    topology: str,
    roles: list[str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    overrides = overrides or {}
    return {role: overrides.get(role, prompt_path(topology, dataset, role).read_text().strip()) for role in roles}


def call_with_supported_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **supported)


@contextmanager
def hide_broken_matplotlib_metadata():
    """Treat a broken matplotlib dist-info as an absent optional dependency.

    Some LangChain imports pull in transformers, which probes optional package
    versions. With malformed matplotlib metadata,
    ``importlib.metadata.version("matplotlib")`` raises ``TypeError`` during
    topology import. The runners do not need matplotlib, so it is reported as
    "not installed" while real runner modules are imported.
    """

    originals: list[tuple[Any, Any]] = []

    def patch_version(metadata_module: Any) -> None:
        original = metadata_module.version

        def safe_version(distribution_name: str):
            try:
                return original(distribution_name)
            except TypeError:
                if str(distribution_name).lower() == "matplotlib":
                    raise metadata_module.PackageNotFoundError(distribution_name) from None
                raise

        metadata_module.version = safe_version
        originals.append((metadata_module, original))

    patch_version(importlib.metadata)
    try:
        import importlib_metadata  # type: ignore

        if importlib_metadata is not importlib.metadata:
            patch_version(importlib_metadata)
    except Exception:
        pass

    try:
        yield
    finally:
        for metadata_module, original in reversed(originals):
            metadata_module.version = original


def import_real_module(module_name: str):
    """Import a real runner module with environment guards."""

    import importlib

    with hide_broken_matplotlib_metadata():
        return importlib.import_module(module_name)


def import_isolated_real_module(module_name: str, **params: Any):
    """Import a fresh copy of a real runner module for threaded execution.

    Real topology modules keep prompts, model URLs, and team sizes in module
    globals. Optimizers patch those globals per program candidate. Loading a unique
    module object per eval call lets threaded baseline/validation runs avoid
    the module-level lock without cross-contaminating prompts.
    ``params`` are preset in the copy before it runs (a runner parameter such as
    ``COMMUNICATION_FORMAT``, see :mod:`core.variant`).
    """

    import importlib

    with hide_broken_matplotlib_metadata():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None or spec.loader is None:
            return importlib.import_module(module_name)
        unique_name = f"{module_name}__gepa_{threading.get_ident()}_{uuid.uuid4().hex}"
        isolated_spec = importlib.util.spec_from_file_location(unique_name, spec.origin)
        if isolated_spec is None or isolated_spec.loader is None:
            return importlib.import_module(module_name)
        module = importlib.util.module_from_spec(isolated_spec)
        module.__dict__.update(params)
        sys.modules[unique_name] = module
        isolated_spec.loader.exec_module(module)
        return module


def fenced_code(code: str | None) -> str:
    return f"```python\n{code or ''}\n```"


def fenced_diff(patch: str | None) -> str:
    return f"```diff\n{patch or ''}\n```"


def create_tiny_git_repo() -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "README.md").write_text("Temporary SWE workspace for compile-time prompt optimization.\n")
    (root / "buggy.py").write_text("def buggy():\n    return 'replace me if needed'\n")
    subprocess.run(["git", "init"], cwd=root, capture_output=True, text=True, check=False)
    subprocess.run(["git", "add", "."], cwd=root, capture_output=True, text=True, check=False)
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-m", "init"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return tmp


@dataclass(frozen=True)
class RoleNudge:
    """An instruction appended to the prompt of the matching roles of a dataset."""

    datasets: frozenset[str]
    roles: frozenset[str]
    text: str
    prompt_topologies: frozenset[str] | None = None  # None: every prompt topology

    def applies(self, dataset: str, prompt_topology: str, role: str) -> bool:
        """Whether this nudge belongs in ``role``'s prompt."""
        if self.prompt_topologies is not None and prompt_topology not in self.prompt_topologies:
            return False
        return dataset in self.datasets and role in self.roles


_CENTRALIZED_PROMPTS = frozenset({"centralized", "centralized_autogen"})
# Robustness instructions of the code and patch tasks, in the order they are appended.
ROLE_NUDGES = (
    RoleNudge(
        datasets=frozenset({"apps", "lcb"}),
        roles=frozenset({"manager"}),
        prompt_topologies=_CENTRALIZED_PROMPTS,
        text=(
            "\n\nCompile-time robustness: if delegation/tool calls are not "
            "strictly necessary, solve directly and emit the final fenced "
            "```python``` solution. If you do call a tool, keep arguments "
            "short valid JSON strings."
        ),
    ),
    RoleNudge(
        datasets=frozenset({"apps", "lcb"}),
        roles=frozenset({"solver", "coder", "debugger", "debater"}),
        text=(
            "\n\nCode-first requirement: do not write a long explanation. "
            "Your final answer must start with ```python and contain the "
            "complete submitted solution before any prose."
        ),
    ),
    RoleNudge(
        datasets=frozenset({"swe"}),
        roles=frozenset({"solver", "patcher", "debater", "manager", "patcher_worker"}),
        text=(
            "\n\nPatch-first requirement: do not write a long explanation. "
            "Your final answer must start with ```diff and contain the "
            "complete unified diff patch before any prose."
        ),
    ),
    RoleNudge(
        datasets=frozenset({"swe"}),
        roles=frozenset({"manager"}),
        prompt_topologies=_CENTRALIZED_PROMPTS,
        text=(
            "\n\nTopology tool rule: as manager, do not call shell_exec, "
            "str_replace, or file_write yourself. Delegate repository "
            "navigation to navigator_worker, targeted edits to "
            "patcher_worker, and shell checks to tester_worker. The final "
            "manager response must contain the selected unified diff in "
            "one fenced ```diff``` block."
        ),
    ),
)


class ModuleAdapterBase:
    """Prompt-mutable adapter that delegates execution to one real module."""

    dataset: str
    topology: str
    framework: str
    prompt_topology: str
    roles_: list[str]
    module_name: str

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        n_agents: int | None = None,
        n_rounds: int | None = None,
    ):
        self._prompts = load_prompts(self.dataset, self.prompt_topology, self.roles_, prompts)
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def roles(self) -> list[str]:
        return list(self.roles_)

    def get_prompt(self, role: str) -> str:
        self._check_role(role)
        return self._prompts[role]

    def set_prompt(self, role: str, text: str) -> None:
        self._check_role(role)
        self._prompts[role] = text

    def reset(self) -> None:
        return None

    def __getstate__(self):
        return self.__dict__.copy()

    def describe_runtime(self, example: Any | None = None) -> dict:
        instance = coerce_instance(example) if example is not None else {}
        return {
            "topology": self.topology,
            "dataset": self.dataset,
            "framework": self.framework,
            "roles": self.roles(),
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "module": self.module_name,
            "example_id": instance.get("id"),
        }

    def load_module(self):
        """A fresh copy of the runner module for one call (see ``import_isolated_real_module``)."""
        return import_isolated_real_module(self.module_name)

    @contextmanager
    def patched_module(self, module):
        restore = {}

        def patch(name: str, value: Any) -> None:
            restore[name] = getattr(module, name, None)
            setattr(module, name, value)

        if hasattr(module, "_load_prompt"):
            patch("_load_prompt", lambda role: self.prompt_for_role(module, role))
        if hasattr(module, "SYSTEM_PROMPT"):
            role = "debater" if "debater" in self._prompts else self.roles_[0]
            patch("SYSTEM_PROMPT", self.prompt_for_role(module, role))
        if hasattr(module, "_build_llm"):
            patch("_build_llm", self._crewai_llm if self.framework == "crewai" else self._langchain_llm)
        if hasattr(module, "_build_client"):
            patch("_build_client", self._openai_client if self.framework == "openai" else self._autogen_client)
        if hasattr(module, "build_agent"):
            original_build_agent = module.build_agent

            def _build_agent_with_recursion_headroom(*args, **kwargs):
                return _AgentRecursionHeadroom(original_build_agent(*args, **kwargs))

            patch("build_agent", _build_agent_with_recursion_headroom)
        if hasattr(module, "VLLM_BASE_URL"):
            patch("VLLM_BASE_URL", next_task_endpoint())
        if hasattr(module, "MODEL_ID"):
            patch("MODEL_ID", os.environ.get("MODEL_ID", TASK_MODEL))
        if hasattr(module, "N_AGENTS") and self.n_agents is not None:
            patch("N_AGENTS", self.n_agents)
        if hasattr(module, "N_ROUNDS") and self.n_rounds is not None:
            patch("N_ROUNDS", self.n_rounds)
        if hasattr(module, "_RECURSION_LIMIT"):
            patch("_RECURSION_LIMIT", max(int(getattr(module, "_RECURSION_LIMIT", 0) or 0), 90))
        try:
            yield
        finally:
            for name, value in restore.items():
                setattr(module, name, value)

    def prompt_for_role(self, module, role: str) -> str:
        """The prompt ``role`` executes with: its text, the module's output-format nudge and appendix,
        the :data:`ROLE_NUDGES` that apply to it, then the protected output contract.

        Each addition is appended only when the text does not already contain it.
        """
        text = self._prompts[role]
        for name in ("_OUTPUT_FORMAT_NUDGE", "_OUTPUT_FORMAT_APPENDIX"):
            addition = getattr(module, name, "")
            if addition and addition not in text:
                text += addition
        for nudge in ROLE_NUDGES:
            if nudge.applies(self.dataset, self.prompt_topology, role) and nudge.text not in text:
                text += nudge.text
        return append_output_contract(text, self.dataset, self.prompt_topology, role)

    @staticmethod
    def _langchain_llm():
        return default_chat_model(0, max_tokens=task_max_tokens())

    @staticmethod
    def _crewai_llm():
        from crewai import LLM

        return LLM(
            model=f"openai/{os.environ.get('MODEL_ID', TASK_MODEL)}",
            base_url=next_task_endpoint(),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            temperature=task_temperature(),
            top_p=0.9,
            seed=task_request_seed(),
            max_tokens=task_max_tokens(),
            additional_drop_params=[],
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    @staticmethod
    def _openai_client():
        from openai import OpenAI

        return OpenAI(
            base_url=next_task_endpoint(),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=600.0,
            max_retries=5,
        )

    @staticmethod
    def _autogen_client():
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=os.environ.get("MODEL_ID", TASK_MODEL),
            base_url=next_task_endpoint(),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "qwen",
                "structured_output": False,
            },
            temperature=task_temperature(),
            top_p=0.9,
            seed=task_request_seed(),
            max_tokens=task_max_tokens(),
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    def format_role_trace(self, role: str, output: Any) -> str:
        self._check_role(role)
        if not isinstance(output, dict):
            return str(output)
        runner_output = output.get("runner_output") or {}
        return "\n".join(
            [
                f"role={role}",
                f"winner={output.get('winner')}",
                f"selected_answer={output.get('answer')}",
                self.role_detail(role, runner_output),
            ]
        )

    def role_detail(self, role: str, out: dict) -> str:
        if "by_stage" in out:
            return f"{role}_text={str((out.get('by_stage') or {}).get(role, ''))[:1200]}"
        for key in ("per_agent", "per_peer"):
            if key in out:
                return f"{role}_{key}={str(out.get(key))[:1200]}"
        messages = out.get("messages") or []
        role_msgs = [msg.get("content", "") for msg in messages if isinstance(msg, dict) and msg.get("source") == role]
        return f"{role}_last_message={str(role_msgs[-1] if role_msgs else out.get('raw', ''))[:1200]}"

    def _check_role(self, role: str) -> None:
        if role not in self.roles_:
            raise KeyError(f"Unknown role {role!r}; expected one of {self.roles_}")


class _AgentRecursionHeadroom:
    """Raise hardcoded ReAct recursion limits without changing agent behavior."""

    def __init__(self, agent):
        self._agent = agent

    def __getattr__(self, name: str):
        return getattr(self._agent, name)

    @staticmethod
    def _config(config):
        updated = dict(config or {})
        updated["recursion_limit"] = max(int(updated.get("recursion_limit", 0) or 0), 80)
        return updated

    def invoke(self, *args, **kwargs):
        kwargs["config"] = self._config(kwargs.get("config"))
        return self._agent.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        kwargs["config"] = self._config(kwargs.get("config"))
        return await self._agent.ainvoke(*args, **kwargs)
