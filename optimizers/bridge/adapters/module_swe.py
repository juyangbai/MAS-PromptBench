"""SWE adapters backed by the real topology modules."""

from __future__ import annotations

import inspect
import os
import tempfile
from typing import Any

from optimizers.bridge.adapters.gpqa_common import coerce_instance
from optimizers.bridge.adapters.module_common import (
    ModuleAdapterBase,
    create_tiny_git_repo,
    fenced_diff,
    import_real_module,
    module_lock,
)


class ModuleSWEAdapter(ModuleAdapterBase):
    dataset = "swe"

    def run_example(self, example: Any) -> dict:
        instance = coerce_instance(example)
        module = import_real_module(self.module_name)
        temp_repos = []
        try:
            with module_lock(self.module_name), self.patched_module(module):
                out = self._solve(module, instance, temp_repos)
        finally:
            from core import swe_sandbox

            for tmp in temp_repos:
                swe_sandbox.unregister_worktree(tmp.name)
                tmp.cleanup()
        patch = out.get("patch") or ""
        raw = out.get("raw") or self._fallback_raw(out)
        return {
            "model_output": [],
            "answer": patch,
            "answer_text": fenced_diff(patch) if patch else raw,
            "patch": patch,
            "winner": out.get("winner"),
            "buckets": {},
            "raw": raw,
            "runner_output": out,
        }

    def _solve(self, module, instance: dict, temp_repos: list) -> dict:
        sig = inspect.signature(module.solve)
        if hasattr(module, "_set_repo_dir"):
            tmp = self._repo_for_instance(instance, suffix="main", module=module)
            temp_repos.append(tmp)
            self._register_sandbox(module, tmp.name, instance)
            module._set_repo_dir(tmp.name)

        if "problem_statement" in sig.parameters:
            return module.solve(
                instance["problem_statement"],
                instance_id=instance.get("instance_id"),
                hints_text=instance.get("hints_text"),
            )

        kwargs = {}
        if "eval_mode" in sig.parameters:
            kwargs["eval_mode"] = "none"
        if "peer_workdirs" in sig.parameters:
            peer_dirs = []
            for _ in range(self.n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4"))):
                tmp = self._repo_for_instance(instance, suffix=f"peer{len(peer_dirs)}", module=module)
                temp_repos.append(tmp)
                self._register_sandbox(module, tmp.name, instance)
                peer_dirs.append(tmp.name)
            kwargs["peer_workdirs"] = peer_dirs
        return module.solve(instance, **kwargs)

    @staticmethod
    def _register_sandbox(module, workdir: str, instance: dict) -> None:
        """Bind a worktree to its instance image so shell_exec runs sandboxed."""
        from core import swe_sandbox

        ensure_sif = module._ensure_sif
        instance_id = instance.get("instance_id") or instance.get("id")
        if instance_id:
            swe_sandbox.register_worktree(workdir, lambda: ensure_sif(instance_id))

    def _repo_for_instance(self, instance: dict, suffix: str, module) -> tempfile.TemporaryDirectory:
        """Private copy of the instance repository, reset to its base commit.

        Reuses a checkout under SWE_WORK_ROOT / ~/swe_work* when one exists and
        otherwise clones the repository once into SWE_WORK_ROOT (default
        ~/swe_work). The toy repository is only for offline tests
        (ALLOW_TOY_SWE_WORKDIR=1); real rollouts never fall back to it.
        """
        instance_id = instance.get("instance_id") or instance.get("id")
        source = self._existing_workdir(instance, suffix) or self._clone_base_workdir(instance, module)
        if source is None:
            if os.environ.get("ALLOW_TOY_SWE_WORKDIR", "0") == "1":
                return create_tiny_git_repo()
            raise RuntimeError(f"missing_real_swe_workdir: {instance_id}")
        tmp = tempfile.TemporaryDirectory()
        try:
            subprocess_result = __import__("subprocess").run(
                ["cp", "-a", f"{source}/.", tmp.name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if subprocess_result.returncode != 0:
                raise RuntimeError(subprocess_result.stderr.strip() or subprocess_result.stdout.strip())
            head = __import__("subprocess").run(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=tmp.name,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if head.returncode != 0:
                raise RuntimeError(head.stderr.strip() or "copied workdir has no HEAD")
            base_commit = instance.get("base_commit")
            if base_commit:
                for command in (["git", "reset", "--hard", "--quiet", base_commit], ["git", "clean", "-fdxq"]):
                    result = __import__("subprocess").run(
                        command, cwd=tmp.name, capture_output=True, text=True, timeout=120
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr.strip() or f"{command[1]} failed")
        except Exception:
            tmp.cleanup()
            raise
        return tmp

    @staticmethod
    def _clone_base_workdir(instance: dict, module):
        """Clone the instance repository at its base commit into the work root."""
        from pathlib import Path

        instance_id = instance.get("instance_id") or instance.get("id")
        repo, base_commit = instance.get("repo"), instance.get("base_commit")
        if not (instance_id and repo and base_commit):
            return None
        target = Path(os.environ.get("SWE_WORK_ROOT") or Path.home() / "swe_work") / str(instance_id)
        if not (target / ".git").is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            if module.clone_and_checkout(repo, base_commit, target):
                return None
        return target

    def _existing_workdir(self, instance: dict, suffix: str):
        from pathlib import Path

        instance_id = instance.get("instance_id") or instance.get("id")
        if not instance_id:
            return None
        roots = []
        override = os.environ.get("SWE_WORK_ROOT")
        if override:
            roots.append(Path(override))
        topo_root = self.topology
        for marker in ("_autogen", "_openai", "_crewai"):
            topo_root = topo_root.replace(marker, "")
        home = Path.home()
        roots.append(home / "swe_work")
        roots.append(home / f"swe_work_{topo_root}")
        roots.extend(sorted(home.glob("swe_work*")))
        names = [str(instance_id)]
        names.extend(f"{instance_id}_a{i}" for i in range(self.n_agents or 4))
        for root in roots:
            if not root.exists():
                continue
            for name in names:
                candidate = root / name
                if (candidate / ".git").is_dir():
                    return candidate
            for candidate in sorted(root.glob(f"{instance_id}*")):
                if (candidate / ".git").is_dir():
                    return candidate
        return None

    def _fallback_raw(self, out: dict) -> str:
        if "by_stage" in out:
            stages = out.get("by_stage") or {}
            return str(stages.get(self.roles_[-1]) or stages.get("patcher") or "")
        if "per_agent" in out:
            return str((out.get("per_agent") or [{}])[0].get("raw", ""))
        if "per_peer" in out:
            return str((out.get("per_peer") or [{}])[0].get("raw", ""))
        messages = out.get("messages") or []
        return str(messages[-1].get("content") if messages and isinstance(messages[-1], dict) else "")


class SingleSWEAdapter(ModuleSWEAdapter):
    topology = "single"
    framework = "langgraph"
    prompt_topology = "single"
    roles_ = ["solver"]
    module_name = "topologies.single.swe.langgraph_swe"


class IndependentSWEAdapter(ModuleSWEAdapter):
    topology = "independent"
    framework = "langgraph"
    prompt_topology = "independent"
    roles_ = ["patcher"]
    module_name = "topologies.independent.swe.langgraph_swe"

    def __init__(self, *args, n_agents: int | None = None, **kwargs):
        super().__init__(*args, n_agents=n_agents or int(os.environ.get("INDEPENDENT_N_AGENTS", "4")), **kwargs)


class DecentralizedSWEAdapter(ModuleSWEAdapter):
    topology = "decentralized"
    framework = "langgraph"
    prompt_topology = "decentralized"
    roles_ = ["debater"]
    module_name = "topologies.decentralized.langgraph.swe.langgraph_swe"

    def __init__(self, *args, n_agents: int | None = None, n_rounds: int | None = None, **kwargs):
        super().__init__(
            *args,
            n_agents=n_agents or int(os.environ.get("DECENTRALIZED_N_AGENTS", "4")),
            n_rounds=n_rounds or int(os.environ.get("DECENTRALIZED_N_ROUNDS", "2")),
            **kwargs,
        )


class DecentralizedOpenAIAgentsSWEAdapter(DecentralizedSWEAdapter):
    topology = "decentralized_openai_agents"
    framework = "openai_agents"
    module_name = "topologies.decentralized.openai_agents.swe.openai_agents_swe"


class SequentialSWEAdapter(ModuleSWEAdapter):
    topology = "sequential"
    framework = "langgraph"
    prompt_topology = "sequential"
    roles_ = ["investigator", "planner", "patcher", "tester"]
    module_name = "topologies.sequential.langgraph.swe.langgraph_swe"


class SequentialCrewAISWEAdapter(SequentialSWEAdapter):
    topology = "sequential_crewai"
    framework = "crewai"
    module_name = "topologies.sequential.crewai.swe.crewai_swe"


class CentralizedSWEAdapter(ModuleSWEAdapter):
    topology = "centralized"
    framework = "langgraph"
    prompt_topology = "centralized"
    roles_ = ["manager", "navigator_worker", "patcher_worker", "tester_worker"]
    module_name = "topologies.centralized.langgraph.swe.langgraph_swe"


class CentralizedAutoGenSWEAdapter(CentralizedSWEAdapter):
    topology = "centralized_autogen"
    framework = "autogen"
    module_name = "topologies.centralized.autogen.swe.autogen_swe"
