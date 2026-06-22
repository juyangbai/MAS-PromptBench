"""Shared real-topology module adapter for non-BFCL datasets."""
from __future__ import annotations

import inspect
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from real_runner_mipro.adapters.gpqa_common import coerce_instance, default_chat_model
from real_runner_mipro.lm import TASK_MODEL, next_task_endpoint
from real_runner_mipro.output_contracts import append_output_contract


MODULE_MAX_TOKENS = int(os.environ.get("REAL_RUNNER_TASK_MAX_TOKENS", "4096"))

_MODULE_LOCKS: dict[str, threading.Lock] = {}
_MODULE_LOCKS_GUARD = threading.Lock()


def module_lock(module_name: str) -> threading.Lock:
    with _MODULE_LOCKS_GUARD:
        if module_name not in _MODULE_LOCKS:
            _MODULE_LOCKS[module_name] = threading.Lock()
        return _MODULE_LOCKS[module_name]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ensure_repo_root_on_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def prompt_path(topology: str, dataset: str, role: str) -> Path:
    return repo_root() / "configs" / "prompts" / topology / dataset / f"{role}.txt"


def load_prompts(
    dataset: str,
    topology: str,
    roles: list[str],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    overrides = overrides or {}
    return {
        role: overrides.get(role, prompt_path(topology, dataset, role).read_text().strip())
        for role in roles
    }


def call_with_supported_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **supported)


@contextmanager
def hide_broken_matplotlib_metadata():
    """Treat the broken local matplotlib dist-info as an absent optional dep.

    Some LangChain imports pull in transformers, which probes optional package
    versions.  The current conda env has malformed matplotlib metadata, causing
    ``importlib.metadata.version("matplotlib")`` to raise ``TypeError`` during
    topology import.  We do not need matplotlib for these runners, so expose it
    as "not installed" while importing real runner modules.
    """

    originals: list[tuple[Any, Any]] = []

    def patch_version(metadata_module: Any) -> None:
        original = metadata_module.version

        def safe_version(distribution_name: str):
            try:
                return original(distribution_name)
            except TypeError:
                if str(distribution_name).lower() == "matplotlib":
                    raise metadata_module.PackageNotFoundError(distribution_name)
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
    """Import a real runner module with codex-side environment guards."""

    import importlib

    ensure_repo_root_on_path()
    with hide_broken_matplotlib_metadata():
        return importlib.import_module(module_name)


def import_isolated_real_module(module_name: str):
    """Import a fresh copy of a real runner module for threaded execution.

    Real topology modules keep prompts, model URLs, and team sizes in module
    globals. MIPRO patches those globals per program candidate. Loading a
    unique module object per eval call lets threaded optimization avoid
    module-level lock contention without cross-contaminating prompts.
    """

    import importlib

    ensure_repo_root_on_path()
    with hide_broken_matplotlib_metadata():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None or spec.loader is None:
            return importlib.import_module(module_name)
        unique_name = f"{module_name}__mipro_{threading.get_ident()}_{uuid.uuid4().hex}"
        isolated_spec = importlib.util.spec_from_file_location(unique_name, spec.origin)
        if isolated_spec is None or isolated_spec.loader is None:
            return importlib.import_module(module_name)
        module = importlib.util.module_from_spec(isolated_spec)
        sys.modules[unique_name] = module
        isolated_spec.loader.exec_module(module)
        return module


def fenced_json(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n```"


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
        ["git", "-c", "user.email=codex@example.com", "-c", "user.name=Codex", "commit", "-m", "init"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return tmp


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
        text = self._prompts[role]
        nudge = getattr(module, "_OUTPUT_FORMAT_NUDGE", "")
        appendix = getattr(module, "_OUTPUT_FORMAT_APPENDIX", "")
        if nudge and nudge not in text:
            text += nudge
        if appendix and appendix not in text:
            text += appendix
        if (
            self.dataset in {"apps", "lcb"}
            and self.prompt_topology in {"centralized", "centralized_autogen"}
            and role == "manager"
        ):
            direct_nudge = (
                "\n\nCompile-time robustness: if delegation/tool calls are not "
                "strictly necessary, solve directly and emit the final fenced "
                "```python``` solution. If you do call a tool, keep arguments "
                "short valid JSON strings."
            )
            if direct_nudge not in text:
                text += direct_nudge
        if self.dataset in {"apps", "lcb"} and role in {"solver", "coder", "debugger", "debater"}:
            code_first_nudge = (
                "\n\nCode-first requirement: do not write a long explanation. "
                "Your final answer must start with ```python and contain the "
                "complete submitted solution before any prose."
            )
            if code_first_nudge not in text:
                text += code_first_nudge
        if self.dataset == "swe" and role in {"solver", "patcher", "debater", "manager", "patcher_worker"}:
            patch_first_nudge = (
                "\n\nPatch-first requirement: do not write a long explanation. "
                "Your final answer must start with ```diff and contain the "
                "complete unified diff patch before any prose."
            )
            if patch_first_nudge not in text:
                text += patch_first_nudge
        if (
            self.dataset == "swe"
            and self.prompt_topology in {"centralized", "centralized_autogen"}
            and role == "manager"
        ):
            swe_manager_nudge = (
                "\n\nTopology tool rule: as manager, do not call shell_exec, "
                "str_replace, or file_write yourself. Delegate repository "
                "navigation to navigator_worker, targeted edits to "
                "patcher_worker, and shell checks to tester_worker. The final "
                "manager response must contain the selected unified diff in "
                "one fenced ```diff``` block."
            )
            if swe_manager_nudge not in text:
                text += swe_manager_nudge
        return append_output_contract(text, self.dataset, self.prompt_topology, role)

    @staticmethod
    def _langchain_llm():
        return default_chat_model(0, max_tokens=MODULE_MAX_TOKENS)

    @staticmethod
    def _crewai_llm():
        from crewai import LLM

        return LLM(
            model=f"openai/{os.environ.get('MODEL_ID', TASK_MODEL)}",
            base_url=next_task_endpoint(),
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            temperature=0.2,
            top_p=0.9,
            seed=0,
            max_tokens=MODULE_MAX_TOKENS,
            additional_drop_params=[],
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    @staticmethod
    def _openai_client():
        from openai import OpenAI

        return OpenAI(base_url=next_task_endpoint(), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=300.0, max_retries=5)

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
            temperature=0.2,
            top_p=0.9,
            seed=0,
            max_tokens=MODULE_MAX_TOKENS,
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
        role_msgs = [
            msg.get("content", "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("source") == role
        ]
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
