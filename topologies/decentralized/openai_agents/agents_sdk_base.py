"""Decentralized debate engine on the OpenAI Agents SDK.

N peers answer the same task over R synchronous rounds. Round 0 is
independent; in every later round each peer receives the original task, its
own previous answer and the other peers' answers from the immediately
preceding completed round (a snapshot barrier: no peer reads a round that is
still in progress). Every peer turn is one Agents SDK run with the dataset's
function tools and no handoffs. The final output is the most common
whitespace-normalized final-round output; ties go to the lowest peer index.

openai-agents 0.22.0 requires openai>=3, which conflicts with the main
environment (openai<3), so the SDK is installed into its own directory
($OPENAI_AGENTS_PATH, default <repo>/vendor/openai_agents) and that directory
goes first on PYTHONPATH for the whole process: its copies of openai, anyio,
typing_extensions, pydantic and the other shared dependencies then win every
import, whatever the import order. The entry points of openai_agents cells
call :func:`reexec_with_sdk_first` before they run, and
:func:`load_agents_sdk` refuses a process that mixes the two environments.
"""

# Config
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import pkgutil
import site
import sys
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from core.paths import REPO_ROOT as _REPO_ROOT

VENDOR_SDK_DIR = _REPO_ROOT / "vendor" / "openai_agents"
SDK_REQUIREMENTS = "requirements-openai-agents.txt"
FRAMEWORK = "openai_agents"
DEFAULT_MAX_TURNS = 8

_INSTALL_HINT = (
    "OpenAI Agents SDK is not installed. It needs openai>=3, which conflicts with "
    "the main environment, so install it into its own directory from the "
    f"repository root:\n  pip install --target vendor/openai_agents -r {SDK_REQUIREMENTS}\n"
    "or set OPENAI_AGENTS_PATH to an existing installation."
)


# SDK directory and process path
def sdk_dir() -> Path | None:
    """The isolated SDK installation ($OPENAI_AGENTS_PATH, else
    <repo>/vendor/openai_agents), resolved; None when it is not a directory."""
    configured = os.environ.get("OPENAI_AGENTS_PATH")
    path = Path(configured).expanduser() if configured else VENDOR_SDK_DIR
    return path.resolve() if path.is_dir() else None


def sdk_first_on_path(path: Path) -> bool:
    """True when ``path`` is on ``sys.path`` ahead of every site-packages
    directory, as it is when it is the first PYTHONPATH entry."""
    target = os.path.realpath(path)
    site_dirs = {os.path.realpath(entry) for entry in (*site.getsitepackages(), site.getusersitepackages())}
    for entry in sys.path:
        real = os.path.realpath(entry or os.curdir)
        if real == target:
            return True
        if real in site_dirs:
            return False
    return False


def reexec_with_sdk_first() -> None:
    """Restart this process with the SDK directory first on PYTHONPATH.

    For the ``__main__`` block of an entry point that runs openai_agents
    cells, before it does anything else. Returns when no restart is needed or
    possible: there is no SDK directory, it is already ahead of
    site-packages, or PYTHONPATH already starts with it but the interpreter
    ignores PYTHONPATH (``-E``, ``-I``), which :func:`load_agents_sdk` then
    reports. Otherwise replaces the process (``os.execve``, same interpreter,
    command line and environment plus the PYTHONPATH entry) and does not
    return; the new process imports every package the SDK directory provides
    from it.
    """
    path = sdk_dir()
    if path is None or sdk_first_on_path(path):
        return
    pythonpath = os.environ.get("PYTHONPATH", "")
    head = pythonpath.split(os.pathsep, 1)[0]
    if head and os.path.realpath(head) == str(path):
        return
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(path), pythonpath)))}
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)


def _foreign_modules(path: Path) -> list[str]:
    """Top-level modules that ``path`` provides but this process imported
    from somewhere else."""
    foreign = []
    for info in pkgutil.iter_modules([str(path)]):
        origin = getattr(sys.modules.get(info.name), "__file__", None)
        if origin and not Path(origin).resolve().is_relative_to(path):
            foreign.append(info.name)
    return foreign


def _check_process_path(path: Path) -> None:
    """Raise RuntimeError unless ``path`` serves this process every package
    it provides: it is ahead of site-packages and none of its packages was
    imported from elsewhere."""
    if not sdk_first_on_path(path):
        raise RuntimeError(
            f"OpenAI Agents SDK: {path} is not first on PYTHONPATH. It must be first for "
            "the whole process, so that the SDK's openai>=3 and its dependencies replace "
            "the main environment's copies. The decentralized/openai_agents runners and "
            "`python -m optimizers.protocol.run` restart themselves that way; elsewhere, "
            f"start Python with PYTHONPATH={path}{os.pathsep}$PYTHONPATH."
        )
    foreign = _foreign_modules(path)
    if foreign:
        raise RuntimeError(
            f"OpenAI Agents SDK: {path} is first on sys.path, but this process had already "
            f"imported {', '.join(sorted(foreign))} from elsewhere, so it would mix two "
            "environments. Put the directory first on PYTHONPATH when the process starts "
            "instead of adding it to sys.path later."
        )


# SDK import
_SDK: SimpleNamespace | None = None
_SDK_LOCK = threading.Lock()


def _import_sdk() -> SimpleNamespace:
    # openai imports its chat resources on first use; import them here, under
    # the lock, before peers on concurrent threads touch `client.chat`.
    import openai.resources.chat.completions  # noqa: F401
    from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    return SimpleNamespace(
        Agent=Agent,
        FunctionTool=FunctionTool,
        ModelSettings=ModelSettings,
        RunConfig=RunConfig,
        Runner=Runner,
        OpenAIChatCompletionsModel=OpenAIChatCompletionsModel,
        AsyncOpenAI=AsyncOpenAI,
    )


def load_agents_sdk() -> SimpleNamespace:
    """Import the SDK once and return its entry points.

    The SDK comes from the process's own path, which is never changed here:
    from the SDK directory, which must be ahead of site-packages without any
    of its packages imported from elsewhere, or from the main environment
    when there is no SDK directory. Raises RuntimeError with the reason and
    the remedy when that does not hold or the import fails.
    """
    global _SDK
    with _SDK_LOCK:
        if _SDK is None:
            path = sdk_dir()
            if path is not None:
                _check_process_path(path)
            try:
                _SDK = _import_sdk()
            except ImportError as exc:
                if path is None:
                    raise RuntimeError(_INSTALL_HINT) from exc
                raise RuntimeError(
                    f"OpenAI Agents SDK: the installation in {path} does not import ({exc}). "
                    "Empty the directory and reinstall the pinned set from the repository root:\n"
                    f"  pip install --target {path} -r {SDK_REQUIREMENTS}"
                ) from exc
        return _SDK


def require_agents_sdk() -> None:
    """Import the SDK or exit with the reason it is unavailable (status 1).

    The runners call it once their command line is parsed (:func:`core.cli.main`'s
    ``preflight``), so an unusable SDK stops the run before the first row instead of
    becoming every row's error.
    """
    try:
        load_agents_sdk()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


# Records
@dataclass(frozen=True)
class Usage:
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model_calls=self.model_calls + other.model_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def telemetry(self) -> dict[str, int]:
        """The 5-key shape used by `core.telemetry.normalize`."""
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "n_llm_calls": self.model_calls,
            "n_tool_calls": self.tool_calls,
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], Any]


@dataclass
class AgentTurnResult:
    output: str
    usage: Usage = field(default_factory=Usage)
    items: list[Mapping[str, Any]] = field(default_factory=list)
    tool_events: list[Mapping[str, Any]] = field(default_factory=list)


class TurnInvoker(Protocol):
    def invoke(
        self,
        *,
        name: str,
        instructions: str,
        input_text: str,
        request_seed: int,
        tools: tuple[ToolSpec, ...],
    ) -> AgentTurnResult: ...


# Active peer
# Tool handlers that act on per-peer state (e.g. SWE worktrees) look up the
# peer whose turn is running. The runner sets it around each SDK run; the
# SDK's event-loop tasks inherit it from the calling context.
_ACTIVE_AGENT_NAME: ContextVar[str | None] = ContextVar("decentralized_openai_agents_active_peer", default=None)


def active_agent_name() -> str | None:
    """Return the peer whose tool handler is currently executing."""
    return _ACTIVE_AGENT_NAME.get()


# Invoker
def _close_client(client: Any) -> None:
    """Close the per-turn async client on the loop `Runner.run_sync` used."""
    close = getattr(client, "close", None)
    if close is None:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running() or loop.is_closed():
            return
        loop.run_until_complete(close())
    except Exception:
        pass


class OpenAIAgentsSDKInvoker:
    """openai-agents 0.22 adapter for OpenAI-compatible (vLLM) endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 32768,
        thinking: bool = False,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.base_url = base_url
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.max_turns = max_turns

    @staticmethod
    def _serialize_item(item: Any) -> Mapping[str, Any]:
        if hasattr(item, "to_input_item"):
            try:
                value = item.to_input_item()
                if isinstance(value, Mapping):
                    return dict(value)
            except Exception:
                pass
        raw = getattr(item, "raw_item", None)
        if hasattr(raw, "model_dump"):
            return raw.model_dump(mode="json")
        if hasattr(item, "__dict__"):
            return {key: repr(value) for key, value in vars(item).items()}
        return {"repr": repr(item)}

    @staticmethod
    def _tools(specs: tuple[ToolSpec, ...], FunctionTool) -> list[Any]:
        tools = []
        for spec in specs:

            async def call(context, arguments: str, bound: ToolSpec = spec):
                parsed = json.loads(arguments or "{}")
                value = bound.handler(parsed)
                if inspect.isawaitable(value):
                    value = await value
                if isinstance(value, str):
                    return value
                return json.dumps(value, ensure_ascii=False, default=str)

            tools.append(
                FunctionTool(
                    name=spec.name,
                    description=spec.description,
                    params_json_schema=dict(spec.parameters),
                    on_invoke_tool=call,
                    strict_json_schema=False,
                )
            )
        return tools

    def invoke(
        self,
        *,
        name: str,
        instructions: str,
        input_text: str,
        request_seed: int,
        tools: tuple[ToolSpec, ...] = (),
    ) -> AgentTurnResult:
        sdk = load_agents_sdk()
        client = sdk.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
        )
        model = sdk.OpenAIChatCompletionsModel(model=self.model_id, openai_client=client)
        settings = sdk.ModelSettings(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            include_usage=True,
            extra_args={"seed": request_seed},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        agent = sdk.Agent(
            name=name,
            instructions=instructions,
            model=model,
            model_settings=settings,
            tools=self._tools(tools, sdk.FunctionTool),
        )
        try:
            result = sdk.Runner.run_sync(
                agent,
                input_text,
                max_turns=self.max_turns,
                run_config=sdk.RunConfig(
                    tracing_disabled=True,
                    workflow_name="decentralized debate",
                ),
            )
        except Exception as exc:
            details = getattr(exc, "run_data", None)
            if details is not None:
                exc._partial_turn = self._turn_result(details, output="")
            raise
        finally:
            _close_client(client)
        return self._turn_result(result, output=str(result.final_output or ""))

    def _turn_result(self, result: Any, *, output: str) -> AgentTurnResult:
        """Serialize complete or SDK-reported partial state without new calls."""
        usage = Usage()
        for response in result.raw_responses:
            item = response.usage
            usage += Usage(
                model_calls=int(getattr(item, "requests", 0) or 0),
                input_tokens=int(getattr(item, "input_tokens", 0) or 0),
                output_tokens=int(getattr(item, "output_tokens", 0) or 0),
                total_tokens=int(getattr(item, "total_tokens", 0) or 0),
            )
        # The SDK appends a turn's ModelResponse to `raw_responses` only after
        # that turn's tools have run, so a run that fails inside its first tool
        # call reports no responses there although the model was called.
        # `context_wrapper.usage` is updated as each response completes; prefer
        # it when it reports more completed requests. Both are SDK-reported.
        wrapper_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if wrapper_usage is not None:
            requests = int(getattr(wrapper_usage, "requests", 0) or 0)
            if requests > usage.model_calls:
                usage = Usage(
                    model_calls=requests,
                    input_tokens=int(getattr(wrapper_usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(wrapper_usage, "output_tokens", 0) or 0),
                    total_tokens=int(getattr(wrapper_usage, "total_tokens", 0) or 0),
                )
        serialized = [self._serialize_item(item) for item in result.new_items]

        def is_tool_event(item: Mapping[str, Any]) -> bool:
            item_type = str(item.get("type", "")).lower()
            role = str(item.get("role", "")).lower()
            return "tool" in item_type or "tool" in role or item_type in {"function_call", "function_call_output"}

        tool_events = [item for item in serialized if is_tool_event(item)]
        tool_call_count = sum(
            str(item.get("type", "")).lower() in {"function_call", "tool_call"} for item in serialized
        )
        usage = Usage(
            model_calls=usage.model_calls,
            tool_calls=tool_call_count,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
        return AgentTurnResult(
            output=output,
            usage=usage,
            items=serialized,
            tool_events=tool_events,
        )


def build_task_invoker(*, base_url: str, model_id: str) -> OpenAIAgentsSDKInvoker:
    """Invoker with the evaluation decoding policy, read from the environment
    at call time (temperature 0.0 unless TASK_MODEL_TEMPERATURE overrides it;
    thinking disabled)."""
    return OpenAIAgentsSDKInvoker(
        base_url=base_url,
        model_id=model_id,
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        temperature=float(os.environ.get("TASK_MODEL_TEMPERATURE", "0.0")),
        top_p=float(os.environ.get("TASK_MODEL_TOP_P", "0.9")),
        max_tokens=int(os.environ.get("TASK_MODEL_MAX_TOKENS", "32768")),
        thinking=False,
    )


# Seeds
def base_request_seed() -> int:
    return int(os.environ.get("REQUEST_SEED", "0"))


def peer_request_seed(base_seed: int, example_id: str, peer: int, round_index: int) -> int:
    """Stable per-(row, peer, round) request seed; retries reuse it."""
    payload = json.dumps(
        {
            "request_seed": int(base_seed),
            "example_id": str(example_id),
            "peer": f"peer_{int(peer)}",
            "round": int(round_index),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


def content_example_id(prefix: str, text: str) -> str:
    """Row id derived from the task text (md5 prefix, as the loaders use)."""
    return f"{prefix}_" + hashlib.md5((text or "").strip().encode("utf-8")).hexdigest()[:10]


# Debate runner
@dataclass
class DebateRecord:
    example_id: str
    status: str  # "success" | "semantic_failure" | "infrastructure_failure"
    final_output: str | None
    selected_peer: int | None
    votes: dict[str, int]
    peer_final_outputs: list[str]
    messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    usage: Usage
    latency_seconds: float
    model_id: str
    team_size: int
    rounds: int
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def telemetry(self) -> dict[str, int]:
        return self.usage.telemetry()

    def turns(self, round_index: int | None = None) -> list[dict[str, Any]]:
        """Completed (non-partial) peer turns, optionally for one round."""
        return [
            turn
            for turn in self.messages
            if not turn.get("partial") and (round_index is None or turn.get("round") == round_index)
        ]


def _normalized(output: str) -> str:
    return " ".join(output.strip().split())


class DecentralizedAgentsRunner:
    """N peers over R synchronous rounds; no SDK handoffs."""

    framework = FRAMEWORK
    topology = "decentralized"

    def __init__(
        self,
        invoker: TurnInvoker,
        *,
        team_size: int = 4,
        rounds: int = 2,
        tools: tuple[ToolSpec, ...] = (),
    ) -> None:
        if int(team_size) <= 0:
            raise ValueError("team_size must be positive")
        if int(rounds) <= 0:
            raise ValueError("rounds must be positive")
        self.invoker = invoker
        self.team_size = int(team_size)
        self.rounds = int(rounds)
        self.tools = tuple(tools)

    @staticmethod
    def _instructions(roles: Mapping[str, str], peer: int) -> str:
        values = list(roles.values())
        if len(values) == 1:
            return values[0]
        key = f"peer_{peer}"
        if key in roles:
            return roles[key]
        return values[peer % len(values)]

    @staticmethod
    def _round_input(question: str, own_history: list[str], peer_snapshot: tuple[str, ...]) -> str:
        if not own_history:
            return question
        peers = "\n\n".join(f"Peer {index + 1}:\n{answer}" for index, answer in enumerate(peer_snapshot))
        return (
            f"Original task:\n{question}\n\n"
            f"Your previous answer:\n{own_history[-1]}\n\n"
            "Other peers from the immediately preceding completed round:\n"
            f"{peers}\n\nReview the evidence and return your revised final answer."
        )

    @staticmethod
    def _votes(outputs: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for output in outputs:
            normalized = _normalized(output)
            counts[normalized] = counts.get(normalized, 0) + 1
        return counts

    @classmethod
    def _aggregate(cls, outputs: tuple[str, ...]) -> str:
        counts = cls._votes(outputs)
        best = max(counts.values())
        for output in outputs:
            if counts[_normalized(output)] == best:
                return output
        raise AssertionError("unreachable")

    @staticmethod
    def _selected_peer(outputs: Sequence[str], final_output: str) -> int:
        normalized = _normalized(final_output)
        return next(
            (index for index, output in enumerate(outputs) if _normalized(output) == normalized),
            0,
        )

    def run(
        self,
        *,
        example_id: str,
        question: str,
        roles: Mapping[str, str],
        seed_for: Callable[[int, int], int],
    ) -> DebateRecord:
        started = time.monotonic()
        histories: list[list[str]] = [[] for _ in range(self.team_size)]
        messages: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        usage = Usage()
        previous: tuple[str, ...] = ()
        round_index = peer = seed = None
        input_text = ""
        model_id = getattr(self.invoker, "model_id", "")

        try:
            for round_index in range(self.rounds):
                snapshot = tuple(previous)
                current: list[str] = []
                for peer in range(self.team_size):
                    others = tuple(value for index, value in enumerate(snapshot) if index != peer)
                    input_text = self._round_input(question, histories[peer], others)
                    seed = seed_for(peer, round_index)
                    agent_name = f"peer_{peer}"
                    token = _ACTIVE_AGENT_NAME.set(agent_name)
                    turn_started = time.monotonic()
                    try:
                        result = self.invoker.invoke(
                            name=agent_name,
                            instructions=self._instructions(roles, peer),
                            input_text=input_text,
                            request_seed=seed,
                            tools=self.tools,
                        )
                    finally:
                        _ACTIVE_AGENT_NAME.reset(token)
                    histories[peer].append(result.output)
                    current.append(result.output)
                    usage += result.usage
                    messages.append(
                        {
                            "round": round_index,
                            "peer": peer,
                            "request_seed": seed,
                            "input": input_text,
                            "output": result.output,
                            "usage": asdict(result.usage),
                            "latency_s": round(time.monotonic() - turn_started, 3),
                            "sdk_items": result.items,
                        }
                    )
                    tool_events.extend(
                        {"round": round_index, "peer": peer, **dict(event)} for event in result.tool_events
                    )
                previous = tuple(current)
            final_output = self._aggregate(previous)
            return DebateRecord(
                example_id=str(example_id),
                status="success",
                final_output=final_output,
                selected_peer=self._selected_peer(previous, final_output),
                votes=self._votes(previous),
                peer_final_outputs=list(previous),
                messages=messages,
                tool_events=tool_events,
                usage=usage,
                latency_seconds=time.monotonic() - started,
                model_id=model_id,
                team_size=self.team_size,
                rounds=self.rounds,
                metadata={
                    "framework": self.framework,
                    "team_size": self.team_size,
                    "rounds": self.rounds,
                    "round_barrier": "synchronous_previous_round_snapshot",
                    "remote_tracing": False,
                },
            )
        except Exception as exc:
            partial = getattr(exc, "_partial_turn", None)
            if isinstance(partial, AgentTurnResult):
                usage += partial.usage
                messages.append(
                    {
                        "round": round_index,
                        "peer": peer,
                        "request_seed": seed,
                        "input": input_text,
                        "output": partial.output,
                        "usage": asdict(partial.usage),
                        "sdk_items": partial.items,
                        "partial": True,
                        "error_type": type(exc).__name__,
                    }
                )
                tool_events.extend({"round": round_index, "peer": peer, **dict(event)} for event in partial.tool_events)
            partial_calls = partial.usage.model_calls if isinstance(partial, AgentTurnResult) else 0
            bounded_failure = type(exc).__name__ == "MaxTurnsExceeded" and partial_calls > 0
            # A tool the agent itself called wrongly (bad arguments, malformed
            # JSON, a handler that raises) is the agent's own wrong answer:
            # the row is scored as a failure instead of being retried, but only
            # once a model response was observed.
            tool_misuse = _agent_tool_misuse(exc) and (usage.model_calls > 0 or partial_calls > 0)
            bounded_failure = bounded_failure or tool_misuse
            return DebateRecord(
                example_id=str(example_id),
                status="semantic_failure" if bounded_failure else "infrastructure_failure",
                final_output=None,
                selected_peer=None,
                votes={},
                peer_final_outputs=[],
                messages=messages,
                tool_events=tool_events,
                usage=usage,
                latency_seconds=time.monotonic() - started,
                model_id=model_id,
                team_size=self.team_size,
                rounds=self.rounds,
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "framework": self.framework,
                    "team_size": self.team_size,
                    "rounds": self.rounds,
                    "failure_type": type(exc).__name__,
                    "bounded_failure": bounded_failure,
                    "agent_tool_misuse": str(exc)[:300] if tool_misuse else None,
                    "partial_sdk_state_retained": isinstance(partial, AgentTurnResult),
                    "usage_scope": "SDK-reported completed responses",
                },
                exception=exc,
            )


def _agent_tool_misuse(exc: BaseException) -> bool:
    """True when the SDK failed because the agent misused one of its tools.

    openai-agents raises ``ModelBehaviorError`` for malformed tool-call JSON and
    ``UserError("Error running tool <name>: ...")`` when a function tool cannot
    run with the arguments the model supplied.
    """
    name = type(exc).__name__
    if name == "ModelBehaviorError":
        return True
    return name == "UserError" and str(exc).startswith("Error running tool")


class PreObservationFailure(RuntimeError):
    """The debate failed before any model response was observed."""


def raise_if_pre_observation_failure(record: DebateRecord) -> None:
    """Raise for infrastructure failures that never reached the model (the
    caller may retry them); failures after an observation are returned."""
    if record.status == "infrastructure_failure" and record.usage.model_calls == 0:
        raise PreObservationFailure(
            record.error or "OpenAI Agents SDK failed before an observation"
        ) from record.exception


def run_decentralized_debate(
    *,
    invoker: TurnInvoker,
    example_id: str,
    question: str,
    roles: Mapping[str, str],
    tools: tuple[ToolSpec, ...] = (),
    n_agents: int,
    n_rounds: int,
    base_seed: int | None = None,
) -> DebateRecord:
    """Run one N-peer x R-round debate with per-(peer, round) request seeds."""
    base = base_request_seed() if base_seed is None else int(base_seed)
    runner = DecentralizedAgentsRunner(invoker, team_size=n_agents, rounds=n_rounds, tools=tools)
    return runner.run(
        example_id=str(example_id),
        question=question,
        roles=roles,
        seed_for=lambda peer, round_index: peer_request_seed(base, example_id, peer, round_index),
    )


# Tool schemas
_JSON_TYPE_ALIASES = {
    "dict": "object",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
}


def json_schema(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a tool parameter schema (Python type names -> JSON Schema)."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            out = {str(key): normalize(child) for key, child in item.items()}
            if "type" in out:
                out["type"] = _JSON_TYPE_ALIASES.get(str(out["type"]).lower(), out["type"])
            return out
        if isinstance(item, list):
            return [normalize(child) for child in item]
        return item

    normalized = normalize(dict(parameters or {}))
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    normalized.setdefault("additionalProperties", False)
    return normalized


_REQUIRED = object()


def required(kind: type) -> tuple[type, Any]:
    return (kind, _REQUIRED)


def coerce_tool_arguments(
    arguments: Mapping[str, Any],
    fields: Mapping[str, tuple[type, Any]],
) -> dict[str, Any]:
    """Validate model-supplied tool arguments like a typed tool signature.

    Unknown keys are dropped, required keys must be present, and values must
    match the declared `str`/`int` type (integral floats and numeric strings
    are accepted for `int`). A violation raises, which the SDK reports as an
    error running the tool.
    """
    if not isinstance(arguments, Mapping):
        raise TypeError(f"tool arguments must be a JSON object, got {type(arguments).__name__}")
    values: dict[str, Any] = {}
    for name, (kind, default) in fields.items():
        if name not in arguments:
            if default is _REQUIRED:
                raise TypeError(f"missing required argument {name!r}")
            values[name] = default
            continue
        value = arguments[name]
        if kind is int:
            if isinstance(value, bool):
                raise TypeError(f"argument {name!r} must be an integer")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
                value = int(value.strip())
            if not isinstance(value, int):
                raise TypeError(f"argument {name!r} must be an integer")
        elif kind is str and not isinstance(value, str):
            raise TypeError(f"argument {name!r} must be a string")
        values[name] = value
    return values


# Transcript views
def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text", part.get("refusal"))
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return "" if content is None else str(content)


def chat_view(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Render serialized SDK items as chat-completions style messages."""
    messages: list[dict[str, Any]] = []
    for item in items or []:
        kind = str(item.get("type", ""))
        role = item.get("role")
        if kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, default=str)
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": output})
        elif kind == "message" or role == "assistant":
            messages.append({"role": role or "assistant", "content": _message_text(item.get("content"))})
        else:
            messages.append(dict(item))
    return messages


def peer_contexts(record: DebateRecord, roles: Mapping[str, str]) -> list[list[dict[str, Any]]]:
    """One chat-style context per peer: system prompt, then for every round
    the round input followed by that turn's SDK items."""
    contexts = [
        [{"role": "system", "content": DecentralizedAgentsRunner._instructions(roles, peer)}]
        for peer in range(record.team_size)
    ]
    for turn in record.messages:
        peer = turn.get("peer")
        if not isinstance(peer, int) or not 0 <= peer < len(contexts):
            continue
        contexts[peer].append({"role": "user", "content": turn.get("input", "")})
        contexts[peer].extend(chat_view(turn.get("sdk_items") or []))
    return contexts
