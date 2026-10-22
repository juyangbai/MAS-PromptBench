"""End-to-end optimizer method cells (``methods/<method>/<dataset>/<registry key>``).

Each cell runs one protocol job through ``optimizers.protocol.run.main`` on a
tiny budget and on tiny train/validation/test subsets of the dataset's fixed
splits: optimization, final validation, the locked selection and the paired
test. Every task-model and reflection-model request reaches the fake server
and is recorded; the snapshot adds the job's artifacts (``job.json``,
``optimization.json``, the optimizer result, records, bundles, evaluations,
``selection.json``, ``test.json``, ``result.json``) and a short summary.

Determinism:

* the scripted task model answers an item correctly when the item is
  even-numbered or the request carries :data:`MARK`; the scripted reflection
  model recognizes every method's proposal prompt and answers with a parseable
  proposal that adds :data:`MARK`. Replies never depend on the request seed, so
  a race between concurrent identical prompts cannot change an outcome;
* ``time`` and ``datetime`` seen by the repository's own modules (protocol,
  methods, bridge, runners) are frozen: latencies, elapsed times and the
  runners' ``solve_s`` are 0 and TAVO's meta-knowledge stamp is fixed, which
  makes every content hash in the artifacts reproducible;
* LangChain stamps every message with random ids (uuid4 / uuid7 run ids),
  which the protocol hashes into records and some methods quote in reflection
  prompts; the cell's adapter class returns its output with those ids
  replaced, in order of appearance, by fixed UUIDs (:func:`canonical_ids`).
  The rollout itself is unchanged;
* MASPOB's GNN runs single-threaded and deterministic on CPU with a harness
  GATv2 layer (:func:`install_gatv2_stand_in`), because ``torch_geometric`` is
  an optional dependency that is not installed here;
* cells whose requests are concurrent (independent/decentralized topologies,
  MAPRO's judge fan-out, MASPO's paired proposals) record requests in
  canonical order and sort the request telemetry lists named in the cell;
  replica topologies record each distinct request once (the OpenAI SDK
  re-sends some async requests on connections pooled by the previous
  rollout's event loop, and which one depends on timing);
* MASPOB's pool cache identity hashes the fake server's random port; that
  digest is masked;
* reflection prompts are counted with the reflection model's tokenizer (a
  golden prerequisite, see ``prereqs``) and task-model prompts always with
  ``context_fit``'s character fallback (:func:`pin_task_model_token_counter`),
  whether or not the task model's tokenizer is cached.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import io
import json
import os
import re
import sys
import time as _time
import uuid
from pathlib import Path
from typing import Any

from tests.golden import responder, scrub

MARK = "GOLDEN-HINT"
TASK_MODEL = "Qwen/Qwen3.5-9B"
FROZEN_MONOTONIC = 1000.0
FROZEN_NOW = _datetime.datetime(2026, 1, 1, 0, 0, 0)
# Packages whose clock is frozen: the protocol and the methods hash run records,
# and the runners put their own wall-clock durations (``solve_s``) into them.
FROZEN_PACKAGES = ("optimizers", "core", "topologies", "teamsizes", "communications")
WRONG_FINAL = "I could not determine the answer."
TINY_SPLITS = {"train": 3, "validation": 2, "test": 2}

# (method, dataset, registry key, optimizer seed, budget, optimizer kwargs, unordered telemetry)
# The methods that accept only exact grid cells (``cells.BASELINE_METHODS``) run on
# Table-6 cells (hotpotqa / lcb / bfcl, LangGraph, team size 4): hotpotqa is their
# text task and bfcl their tool task.
METHOD_CELLS: tuple[tuple[str, str, str, int, int, dict, tuple[str, ...]], ...] = (
    ("identity", "math", "single", 0, 4, {}, ()),
    ("identity", "toolhop", "single", 1, 4, {}, ()),
    ("gepa", "math", "single", 0, 30, {}, ()),
    ("gepa", "toolhop", "single", 2, 8, {}, ()),
    ("mipro", "math", "single", 1, 40, {}, ()),
    ("mipro", "toolhop", "single", 0, 8, {}, ()),
    ("hivemind", "hotpotqa", "centralized", 0, 10, {}, ()),
    ("hivemind", "bfcl", "independent", 1, 18, {}, ()),
    ("mamut_gepa", "hotpotqa", "sequential", 2, 16, {}, ()),
    ("mamut_gepa", "bfcl", "centralized", 0, 10, {}, ()),
    ("mapro", "hotpotqa", "centralized", 1, 9, {"num_threads": 1}, ("reflection", "task_judge_and_probe")),
    ("mapro", "bfcl", "sequential", 0, 9, {"num_threads": 1}, ("reflection", "task_judge_and_probe")),
    (
        "maspo",
        "hotpotqa",
        "sequential",
        0,
        12,
        {"num_threads": 1, "reflection_inflight": 1, "minibatch": 2},
        ("reflection",),
    ),
    (
        "maspo",
        "bfcl",
        "centralized",
        2,
        12,
        {"num_threads": 1, "reflection_inflight": 1, "minibatch": 2},
        ("reflection",),
    ),
    ("maspob", "hotpotqa", "sequential", 0, 10, {"reflection_inflight": 1}, ()),
    ("maspob", "bfcl", "independent", 1, 10, {"reflection_inflight": 1}, ()),
    ("tavo", "hotpotqa", "centralized", 0, 12, {"reflection_inflight": 1}, ()),
    ("tavo", "bfcl", "sequential", 1, 12, {"reflection_inflight": 1}, ()),
)
# Methods whose own request fan-out is concurrent (canonical request order).
CONCURRENT_METHODS = ("mapro", "maspo")


def method_cells() -> list[dict]:
    cells = []
    for method, dataset, key, seed, budget, kwargs, unordered in METHOD_CELLS:
        topology = key.split("_", 1)[0]
        replicas = topology in ("independent", "decentralized")
        cells.append(
            {
                "id": f"methods/{method}/{dataset}/{key}",
                "area": "methods",
                "kind": "method",
                "method": method,
                "dataset": dataset,
                "key": key,
                "topology": topology,
                "seed": seed,
                "budget": budget,
                "optimizer_kwargs": dict(kwargs),
                "splits": dict(TINY_SPLITS),
                "unordered_telemetry": list(unordered),
                "endpoint_digest": method == "maspob",
                "concurrent": replicas or method in CONCURRENT_METHODS,
                # Replicas run as async tasks; a new event loop per rollout makes the SDK
                # retry some requests on stale pooled connections, so which identical
                # request is recorded twice depends on timing.
                "dedupe_requests": replicas,
            }
        )
    return cells


# ----------------------------------------------------------------------------- frozen clock
class _FrozenTime:
    """``time`` module whose clocks stand still (``sleep`` still sleeps)."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def time(self) -> float:
        return FROZEN_NOW.timestamp()

    def monotonic(self) -> float:
        return FROZEN_MONOTONIC

    def perf_counter(self) -> float:
        return FROZEN_MONOTONIC

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _FrozenDateTime(_datetime.datetime):
    @classmethod
    def now(cls, tz=None):  # stdlib signature
        return cls(*FROZEN_NOW.timetuple()[:6], tzinfo=tz)

    @classmethod
    def today(cls):  # stdlib signature
        return cls(*FROZEN_NOW.timetuple()[:6])


class _FrozenDateTimeModule:
    datetime = _FrozenDateTime

    def __getattr__(self, name: str) -> Any:
        return getattr(_datetime, name)


_FROZEN_TIME = _FrozenTime(_time)
_FROZEN_DATETIME = _FrozenDateTimeModule()


def _freeze_module(module: Any) -> None:
    namespace = getattr(module, "__dict__", {})
    for name, value in list(namespace.items()):
        if value is _time:
            namespace[name] = _FROZEN_TIME
        elif value is _datetime:
            namespace[name] = _FROZEN_DATETIME
        elif value is _datetime.datetime:
            namespace[name] = _FrozenDateTime


class _ClockFreezer(importlib.abc.MetaPathFinder):
    """Freeze the clock of every repository module as it is imported."""

    def find_spec(self, name, path, target=None):
        if not _frozen_package(name):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None or not hasattr(loader, "exec_module"):
            return spec
        original = loader.exec_module

        def exec_module(module, _original=original):
            _original(module)
            _freeze_module(module)

        loader.exec_module = exec_module
        return spec


def _frozen_package(name: str) -> bool:
    return any(name == package or name.startswith(package + ".") for package in FROZEN_PACKAGES)


def freeze_clock() -> None:
    for name, module in list(sys.modules.items()):
        if _frozen_package(name):
            _freeze_module(module)
    if not any(isinstance(finder, _ClockFreezer) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ClockFreezer())


# ----------------------------------------------------------------------------- message ids
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ID_KEYS = frozenset({"id", "run_id", "parent_run_id", "tool_call_id", "trace_id", "message_id"})


def canonical_ids(value: Any, mapping: dict[str, str] | None = None) -> Any:
    """Copy of an adapter output whose random UUIDs under id keys are numbered by first appearance."""
    mapping = {} if mapping is None else mapping

    def fixed(match: re.Match) -> str:
        found = match.group(0).lower()
        if found not in mapping:
            mapping[found] = str(uuid.UUID(int=len(mapping) + 1))
        return mapping[found]

    if isinstance(value, dict):
        return {
            key: (_UUID.sub(fixed, item) if key in ID_KEYS and isinstance(item, str) else canonical_ids(item, mapping))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonical_ids(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(canonical_ids(item, mapping) for item in value)
    fields = getattr(type(value), "model_fields", None)
    if isinstance(fields, dict) and callable(getattr(value, "model_copy", None)):
        # pydantic objects such as LangChain messages: a copy with fixed ids
        update = {}
        for name in fields:
            item = getattr(value, name, None)
            update[name] = (
                _UUID.sub(fixed, item) if name in ID_KEYS and isinstance(item, str) else canonical_ids(item, mapping)
            )
        return value.model_copy(update=update)
    return value


def canonicalize_adapter_ids(dataset: str, key: str) -> None:
    """Make the cell's adapter class return :func:`canonical_ids` of its output."""
    from optimizers.bridge.registry import get_adapter_class

    adapter_class = get_adapter_class(dataset, key)
    original = adapter_class.run_example

    def run_example(self, example, _original=original):
        return canonical_ids(_original(self, example))

    adapter_class.run_example = run_example


# ----------------------------------------------------------------------------- prompt token counting
def pin_task_model_token_counter() -> None:
    """Make ``context_fit`` count task-model prompts with its character fallback."""
    from optimizers.protocol.methods import context_fit

    original = context_fit._token_counter

    def token_counter(model: str | None) -> Any:
        name = str(model or "").removeprefix("openai/")
        return None if name == TASK_MODEL else original(model)

    context_fit._token_counter = token_counter


# ----------------------------------------------------------------------------- MASPOB GNN layer
def install_gatv2_stand_in() -> None:
    """Deterministic dense GATv2 layer registered as ``torch_geometric.nn.GATv2Conv``.

    Same constructor, ``reset_parameters`` and ``forward(x, edge_index)``
    contract as the PyG layer (self loops added, LeakyReLU 0.2 attention,
    softmax over incoming edges, attention dropout, concat or head mean).
    """
    import types

    import torch
    from torch import nn

    class GATv2Conv(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            bias=True,
            **_,
        ):
            super().__init__()
            self.heads, self.out_channels, self.concat = int(heads), int(out_channels), bool(concat)
            self.negative_slope, self.dropout, self.add_self_loops = (
                float(negative_slope),
                float(dropout),
                add_self_loops,
            )
            self.lin_l = nn.Linear(in_channels, self.heads * self.out_channels, bias=bias)
            self.lin_r = nn.Linear(in_channels, self.heads * self.out_channels, bias=bias)
            self.att = nn.Parameter(torch.empty(1, self.heads, self.out_channels))
            self.bias = nn.Parameter(torch.empty(self.heads * self.out_channels if concat else self.out_channels))
            self.reset_parameters()

        def reset_parameters(self):
            for lin in (self.lin_l, self.lin_r):
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            nn.init.xavier_uniform_(self.att)
            nn.init.zeros_(self.bias)

        def forward(self, x, edge_index):
            count, heads, channels = x.size(0), self.heads, self.out_channels
            left = self.lin_l(x).view(count, heads, channels)
            right = self.lin_r(x).view(count, heads, channels)
            adjacency = torch.zeros(count, count, dtype=torch.bool)
            adjacency[edge_index[1], edge_index[0]] = True
            if self.add_self_loops:
                adjacency |= torch.eye(count, dtype=torch.bool)
            pair = nn.functional.leaky_relu(right.unsqueeze(1) + left.unsqueeze(0), self.negative_slope)
            scores = (pair * self.att.unsqueeze(0)).sum(-1)
            scores = scores.masked_fill(~adjacency.unsqueeze(-1), float("-inf"))
            alpha = torch.softmax(scores, dim=1)
            alpha = nn.functional.dropout(alpha, p=self.dropout, training=self.training)
            out = torch.einsum("dsh,shc->dhc", alpha, left)
            out = out.reshape(count, heads * channels) if self.concat else out.mean(dim=1)
            return out + self.bias

    package = types.ModuleType("torch_geometric")
    package.__path__ = []
    layers = types.ModuleType("torch_geometric.nn")
    layers.GATv2Conv = GATv2Conv
    package.nn = layers
    sys.modules["torch_geometric"] = package
    sys.modules["torch_geometric.nn"] = layers
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------------- task data and script
def tiny_task_data(dataset: str, sizes: dict[str, int]) -> Any:
    """The first rows of each fixed split, loaded through the bridge dataset."""
    from optimizers.bridge.datasets.split_utils import fixed_split_ids
    from optimizers.protocol.runner import TaskData

    module = importlib.import_module(f"optimizers.bridge.datasets.{dataset}")
    ids = fixed_split_ids(dataset)
    chosen = {split: [str(item) for item in ids[split]][: sizes[split]] for split in ("train", "validation", "test")}
    return TaskData(dataset, chosen, module.load_all())


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _question_texts(value: Any, depth: int = 0) -> list[str]:
    if depth > 6:
        return []
    if isinstance(value, str):
        return [value] if len(value.strip()) >= 12 else []
    if isinstance(value, dict):
        found = []
        for key in ("question", "problem", "query", "prompt", "input", "instruction", "content"):
            if key in value:
                found.extend(_question_texts(value[key], depth + 1))
        return found
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _question_texts(item, depth + 1)]
    return []


def instance_script(dataset: str, rows: list[tuple[str, Any]]) -> dict:
    """Per-item scripts: the needle that identifies the item in a request and its correct final answer."""
    from tests.golden import worker

    instances = []
    for index, (item_id, native) in enumerate(rows):
        inst = worker.example_instance(native)
        task = inst.get("task_instance") if isinstance(inst.get("task_instance"), dict) else inst
        texts = _question_texts(task) or _question_texts(inst)
        needle = _normalized(texts[0])[:40] if texts else _normalized(item_id)
        instances.append({"id": item_id, "index": index, "needle": needle, "script": worker.script_for(dataset, inst)})
    return {"mark": MARK, "instances": instances}


# ----------------------------------------------------------------------------- scripted models
def _text(content: Any) -> str:
    return responder._text(content)


def _messages(body: dict) -> list[dict]:
    return [m for m in (body.get("messages") or []) if isinstance(m, dict)]


def _task_reply(body: dict, script: dict) -> dict:
    messages = _messages(body)
    all_text = "\n".join(_text(m.get("content")) for m in messages)
    system = "\n".join(_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    if "strict reward model" in system or "*reward model*" in all_text:
        digest = int(hashlib.sha1(all_text.encode("utf-8")).hexdigest()[:4], 16)
        bonus = 0.5 if MARK in all_text else 0.0
        return {"content": f"{min(0.99, 0.3 + bonus + (digest % 20) / 100):.2f}"}
    normalized = _normalized(all_text)
    chosen = next(
        (inst for inst in script.get("instances", ()) if inst["needle"] and inst["needle"] in normalized), None
    )
    if chosen is None:
        return responder.respond(body, {"final": WRONG_FINAL})
    correct = chosen["index"] % 2 == 0 or MARK in all_text
    return responder.respond(body, chosen["script"] if correct else {"final": WRONG_FINAL})


def _between(text: str, start: str, end: str) -> str | None:
    head = text.split(start, 1)
    if len(head) != 2:
        return None
    return head[1].split(end, 1)[0]


def _short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _dspy_fields(messages: list[dict]) -> str:
    """DSPy adapter reply filling every output field (chat or JSON format).

    A context-fitted system message may have lost its field list; the output
    fields are then the ``[[ ## name ## ]]`` / JSON template fields that are
    not inputs of the user message.
    """
    system = _text(messages[0].get("content"))
    inputs = dict(
        re.findall(
            r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond with|\Z)",
            _text(messages[-1].get("content")),
            re.DOTALL,
        )
    )
    json_mode = "Outputs will be a JSON object" in system
    fields = re.search(r"Your output fields are:\n(.*?)\nAll interactions will be structured", system, re.DOTALL)
    if fields:
        names = re.findall(r"`(\w+)`", fields.group(1))
    elif json_mode:
        names = re.findall(r'"(\w+)": "\{\w+\}"', system)
    else:
        names = [name for name in re.findall(r"\[\[ ## (\w+) ## \]\]\n\{\w+\}", system) if name not in inputs]
    values = {}
    for name in dict.fromkeys(names):
        if name == "proposed_instruction":
            values[name] = f"{inputs.get('basic_instruction', '').strip()}\n{MARK}: verify the final answer."
        else:
            values[name] = f"scripted {name}"
    if json_mode:
        return json.dumps(values)
    return "".join(f"[[ ## {name} ## ]]\n{value}\n\n" for name, value in values.items()) + "[[ ## completed ## ]]"


def _tavo_reply(prompt: str, head: str) -> str | None:
    if head.startswith("Please evaluate the quality and progress of the current iteration"):
        return json.dumps(
            {
                "iteration_progress": {"score": 6, "analysis": "advanced", "key_achievements": []},
                "iteration_summary": {"overall_score": 6, "areas_for_improvement": ["verify earlier"]},
            }
        )
    if head.startswith("Please analyze the following Agent's performance"):
        role = re.search(r"- Agent ID: (\S+)", prompt)
        role = role.group(1) if role else "agent"
        payload = {
            "overall_score": 7,
            "strengths": [f"{role} kept the handoff short"],
            "weaknesses": [f"{role} did not verify the final answer"],
            "prompt_suggestions": {
                "result_oriented_improvements": f"{MARK}: tie every step to the final answer",
                "effectiveness_enhancements": "N/A",
                "quality_focus_additions": "Check the answer format before replying",
                "collaboration_optimizations": "N/A",
            },
            "specific_prompt_modifications": {
                "add_instructions": [f"{MARK}: verify the final answer"],
                "remove_content": [],
                "restructure_suggestions": [],
            },
        }
        return "```json\n" + json.dumps(payload) + "\n```\nThese recommendations target the result."
    if head.startswith("You are given a list of short rules/suggestions"):
        items = list(dict.fromkeys(re.findall(r"^\d+\. (.+)$", prompt, re.M)))
        return json.dumps({"top": [{"text": item, "support_count": 1, "importance": 0} for item in items[:2]]})
    if head.startswith("You are to synthesize a reusable TRUCE prompt-rule overlay"):
        return f"1. Re-read the question and check the final answer ({MARK}) before replying.\n2. Keep handoffs short."
    if head.startswith("Aggregate the following trajectory-aware prompt edits"):
        return json.dumps(
            {
                "specific_prompt_modifications": {
                    "add_instructions": [f"{MARK}: aggregated rule"],
                    "remove_content": [],
                    "restructure_suggestions": [],
                },
                "prompt_suggestions": {"result_oriented_improvements": "Check the final answer"},
                "strengths": [],
                "weaknesses": [],
            }
        )
    if head.startswith("Please optimize the Agent's System Prompt"):
        role = re.search(r"\*\*Agent ID:\*\* (\S+)", prompt)
        return f"Refined prompt for {role.group(1) if role else 'agent'}. {MARK}: verify the final answer."
    return None


def _reflection_reply(body: dict) -> str:
    messages = _messages(body)
    if messages and "Your output fields are:" in _text(messages[0].get("content")):
        return _dspy_fields(messages)
    prompt = "\n".join(_text(m.get("content")) for m in messages if m.get("role") == "user")
    head = prompt.lstrip()
    if "===BEGIN NEW INSTRUCTION===" in prompt:
        current = re.search(r"```\n(.*?)\n```", prompt, re.DOTALL)
        text = current.group(1) if current else ""
        return f"===BEGIN NEW INSTRUCTION===\n{text}\n{MARK}: verify the final answer.\n===END NEW INSTRUCTION==="
    if "===BEGIN LESSONS===" in prompt:
        return f"===BEGIN LESSONS===\n- {MARK}: re-check the final answer before replying.\n===END LESSONS==="
    if "STYLE TO FOLLOW:" in prompt and "ORIGINAL instruction:" in prompt:
        seed = _between(prompt, "ORIGINAL instruction:\n---\n", "\n---\n\nSTYLE TO FOLLOW:") or ""
        style = _between(prompt, "STYLE TO FOLLOW:", "RULES:") or prompt
        tag = _short(style)
        hint = f"{MARK}: " if int(tag, 16) % 2 == 0 else ""
        return f"{seed.strip()}\n\n{hint}Follow style {tag} when you reply."
    tavo = _tavo_reply(prompt, head)
    if tavo is not None:
        return tavo
    if "BLAME <parent_id>" in prompt:
        parents = re.findall(r"--- parent (\S+)", prompt)
        return "\n".join(f"BLAME {parent}: its output omitted a needed detail" for parent in parents)
    if "alternative versions" in prompt and "VARIANT" in prompt:
        base = _between(prompt, 'Base role prompt:\n"""', '"""') or ""
        count = int(re.search(r"Produce (\d+) alternative", prompt).group(1))
        return "\n".join(f"VARIANT {i}:\n{base.strip()}\n{MARK} variant {i}." for i in range(1, count + 1))
    if "Revision attempt" in prompt:
        current = _between(prompt, 'Current role prompt:\n"""', '"""') or ""
        nonce = re.search(r"Revision attempt (\S+) ", prompt)
        return f"{current.strip()}\n{MARK} revision {nonce.group(1) if nonce else 0}."
    if 'Respond ONLY with "A" or "B"' in prompt:
        output_a = _between(prompt, "Output A:", "Output B:") or ""
        output_b = _between(prompt, "Output B:", "\n\n") or ""
        return "A" if output_a.count(MARK) > output_b.count(MARK) else "B"
    if "<prompt>" in prompt and "Reference prompt" in prompt:
        reference = (
            _between(prompt, "<reference_prompt>\n", "\n</reference_prompt>")
            or _between(prompt, "Reference prompt:\n```\n", "\n```")
            or ""
        )
        return (
            f"<analyse>The outputs skip a final check.</analyse>\n<modification>Add a final check.</modification>\n"
            f"<prompt>{reference.strip()}\n{MARK}: check {_short(prompt)}.</prompt>"
        )
    current = re.search(r"```\n(.*?)\n```", prompt, re.DOTALL)
    if current:
        return f"```\n{current.group(1)}\n{MARK}: verify the final answer.\n```"
    return f"{MARK}: verify the final answer."


def respond(body: dict, script: dict) -> dict:
    """Reply of the scripted task or reflection model (a pure function of body and script)."""
    if body.get("model") == TASK_MODEL:
        return _task_reply(body, script)
    return {"content": _reflection_reply(body)}


# ----------------------------------------------------------------------------- artifacts
SKIPPED_DIRS = ("dspy_cache",)


def _read(path: Path) -> Any:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def _sort_requests(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("requests"), list):
        return {**value, "requests": sorted(value["requests"], key=scrub.canonical)}
    return value


def _replace_text(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {key: _replace_text(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_text(item, old, new) for item in value]
    return value


def stabilize(rel: str, payload: Any, cell: dict, endpoint_digest: str) -> Any:
    """Stable projection of one artifact for the cell's declared nondeterminism.

    * ``unordered_telemetry``: request-telemetry lists that concurrent native
      calls append in completion order are sorted (their content is
      deterministic, only the append order is not);
    * ``endpoint_digest``: MASPOB's prompt-pool cache identity stores the
      SHA-256 of the reflection endpoint URL, whose port is the fake server's
      random port; the digest is masked.

    The optimizer result's ``artifact_sha256`` hashes the unstable form and is dropped.
    """
    unordered, digest = cell["unordered_telemetry"], cell["endpoint_digest"]
    if digest:
        payload = _replace_text(payload, endpoint_digest, "<endpoint-sha256>")
    if not (unordered or digest) or not rel.endswith("optimizer_result.json") or not isinstance(payload, dict):
        return payload
    payload = dict(payload)
    payload.pop("artifact_sha256", None)
    metadata = dict(payload.get("metadata") or {})
    for key in unordered:
        if key in metadata:
            metadata[key] = _sort_requests(metadata[key])
    payload["metadata"] = metadata
    return payload


def collect_artifacts(out: Path, cell: dict) -> dict:
    """Every JSON/JSONL artifact of the job (stable projection) and the names of the other files."""
    endpoint = os.environ.get("REFLECTION_MODEL_BASE_URL", "")
    endpoint_digest = hashlib.sha256(endpoint.strip().encode("utf-8")).hexdigest()
    files: dict[str, Any] = {}
    others: list[str] = []
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out).as_posix()
        if not path.is_file() or any(part in SKIPPED_DIRS for part in rel.split("/")):
            continue
        if path.suffix in (".json", ".jsonl"):
            payload = stabilize(rel, _read(path), cell, endpoint_digest)
            files[rel] = scrub.to_data(payload, long_limit=scrub.LONG_STRING_LIMIT)
        else:
            others.append(rel)
    return {"json": files, "other_files": others}


def summarize(out: Path) -> dict:
    summary: dict[str, Any] = {}
    for name in ("optimization.json", "selection.json", "result.json"):
        path = out / name
        if path.exists():
            summary[name] = json.loads(path.read_text(encoding="utf-8"))
    optimization = summary.get("optimization.json") or {}
    selection = summary.get("selection.json") or {}
    result = summary.get("result.json") or {}
    incumbent = (optimization.get("incumbent_bundle") or {}).get("roles")
    return scrub.to_data(
        {
            "optimization_status": optimization.get("status"),
            "failure_kind": optimization.get("failure_kind"),
            "error": optimization.get("error"),
            "stop_reason": optimization.get("stop_reason"),
            "budget": optimization.get("budget"),
            "incumbent_roles": incumbent,
            "selected_candidate": selection.get("selected_candidate"),
            "fallback_reason": selection.get("fallback_reason"),
            "baseline_validation_score": selection.get("baseline_validation_score"),
            "incumbent_validation_score": selection.get("incumbent_validation_score"),
            "deployed_bundle_sha256": (selection.get("deployed_bundle") or {}).get("bundle_sha256"),
            "test": result.get("test"),
        }
    )


# ----------------------------------------------------------------------------- driver
def drive_method(cell: dict, tmp: Path, script: dict) -> dict:
    """Run one protocol job for the cell and return its snapshot payload."""
    freeze_clock()
    canonicalize_adapter_ids(cell["dataset"], cell["key"])
    pin_task_model_token_counter()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if cell["method"] == "maspob":
        install_gatv2_stand_in()
    from optimizers.protocol import run

    def load_task_data(dataset: str) -> Any:
        data = tiny_task_data(dataset, cell["splits"])
        rows = [
            (item, data.native(item)) for split in ("train", "validation", "test") for item in data.split_ids[split]
        ]
        script.update(instance_script(dataset, rows))
        return data

    out = tmp / "job"
    argv = [
        "--method",
        cell["method"],
        "--dataset",
        cell["dataset"],
        "--topology",
        cell["key"],
        "--model",
        "qwen",
        "--seed",
        str(cell["seed"]),
        "--budget",
        str(cell["budget"]),
        "--out",
        str(out),
    ]
    hooks = run.JobHooks(load_task_data=load_task_data, optimizer_kwargs=dict(cell["optimizer_kwargs"]))
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = run.main(argv, hooks=hooks)
    printed = [line for line in stdout.getvalue().splitlines() if line.startswith("{")]
    return {
        "argv": scrub.to_data(argv),
        "exit_code": exit_code,
        "cli_summary": scrub.to_data(json.loads(printed[-1]) if printed else None),
        "summary": summarize(out),
        "artifacts": collect_artifacts(out, cell),
    }
