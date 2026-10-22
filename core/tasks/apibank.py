"""API-Bank: curated next-API-call tasks, upstream API replay, scoring, prompts and batch records.

A task is the dialogue before one API call of API-Bank (DAMO-ConvAI) Level 1, 2
or 3; the model must emit that call as ``[ApiName(key=value, ...)]``. The
benchmark slice is the curated manifest ``benchmarks/apibank/apibank_eval_ids.json``
(``APIBANK_LEVEL`` / ``--level`` selects one level's manifest). A prediction is
correct when replaying it through API-Bank's own API classes, after the
dialogue's earlier calls, passes that API's correctness check; ToolSearcher
calls are scored against a frozen search catalog (``APIBANK_TOOLSEARCHER_SCORER``
selects the official, upstream or keyword scorer).

An agent turn is one request to the raw OpenAI client: the system prompt is the
role prompt plus the API-Bank hard rule, the user message the API descriptions
and the dialogue (the Agents SDK runner also gets the APIs as function tools).
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import threading
import time
import types
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import agent_runs, cli
from core.batch import attempt
from core.llm import completion_kwargs, openai_client
from core.output_contracts import append_output_contract
from core.paths import PROMPTS_DIR, REPO_ROOT
from core.telemetry import normalize, openai_sdk_accumulate, sum_telemetry

DATASET = "apibank"

# Benchmark slices (levels) and sources
_COMBINED_LEVEL = "all"
_LEVEL_KEYS = ("1", "2", "3")
_DEFAULT_LEVEL_LIMITS = {"1": 100, "2": 100, "3": 245}
_DEFAULT_CURATED_PATHS = {
    _COMBINED_LEVEL: REPO_ROOT / "benchmarks" / "apibank" / "apibank_eval_ids.json",
    "1": REPO_ROOT / "benchmarks" / "apibank" / "apibank_level1_curated.json",
    "2": REPO_ROOT / "benchmarks" / "apibank" / "apibank_level2_curated.json",
    "3": REPO_ROOT / "benchmarks" / "apibank" / "apibank_level3_curated.json",
}
_DEFAULT_UPSTREAM_ROOT = REPO_ROOT / "benchmarks" / "apibank" / "apibank_upstream" / "api-bank"
_CACHE_UPSTREAM_ROOT = REPO_ROOT / ".cache" / "apibank_smoke_damo" / "api-bank"

_LEVEL_DIRS = {
    _COMBINED_LEVEL: "level-1+level-2+level-3",
    "1": "lv1-lv2-samples/level-1-given-desc",
    "2": "lv1-lv2-samples/level-2-toolsearcher",
    "3": "test-data/level-3.json",
}
_LEVEL_NAMES = {
    _COMBINED_LEVEL: "API-Bank curated",
    "1": "API-Bank Level-1 API-call curated",
    "2": "API-Bank Level-2 ToolSearcher/API-call curated",
    "3": "API-Bank Level-3 API-call curated",
}
_LEVEL_ALIASES = {
    "all": _COMBINED_LEVEL,
    "combined": _COMBINED_LEVEL,
    "api-bank": _COMBINED_LEVEL,
    "apibank": _COMBINED_LEVEL,
    "1": "1",
    "l1": "1",
    "level1": "1",
    "level-1": "1",
    "level_1": "1",
    "level-1-given-desc": "1",
    "level_1_given_desc": "1",
    "2": "2",
    "l2": "2",
    "level2": "2",
    "level-2": "2",
    "level_2": "2",
    "level-2-toolsearcher": "2",
    "level_2_toolsearcher": "2",
    "3": "3",
    "l3": "3",
    "level3": "3",
    "level-3": "3",
    "level_3": "3",
}
_SPECIAL_SKIP_APIS = {"SearchEngine", "Translate", "ToolSearcher"}
_NONDETERMINISTIC_SKIP_APIS = {"SearchEngine", "Translate"}
_ANSWER_CALL_RE = re.compile(r"\[(?P<name>[A-Za-z_]\w*)\((?P<args>.*?)\)\]", re.DOTALL)
_ANSWER_CALL_START_RE = re.compile(r"\[[A-Za-z_]\w*\(")
_IMPORT_LOCK = threading.RLock()
_TOOLSEARCHER_SCORER_ALIASES = {
    "official": "official",
    "official_frozen": "official",
    "frozen": "official",
    "upstream": "upstream",
    "raw": "upstream",
    "keyword": "keyword",
}
_FROZEN_TOOLSEARCHER_CACHE: dict[str, dict[str, Any]] = {}


def normalize_level(level: str | int | None = None) -> str:
    raw = str(level or os.environ.get("APIBANK_LEVEL", _COMBINED_LEVEL)).strip().lower()
    key = _LEVEL_ALIASES.get(raw)
    if key is None:
        raise ValueError(f"unsupported API-Bank level {level!r}; expected all, 1, 2, or 3")
    return key


APIBANK_LEVEL = normalize_level(os.environ.get("APIBANK_LEVEL", _COMBINED_LEVEL))
BENCHMARK_NAME = _LEVEL_NAMES[APIBANK_LEVEL]


def benchmark_name(level: str | int | None = None) -> str:
    return _LEVEL_NAMES[normalize_level(level)]


def level_dir(level: str | int | None = None) -> str:
    return _LEVEL_DIRS[normalize_level(level)]


def default_toolsearcher_scorer(level: str | int | None = None) -> str:
    return "official" if normalize_level(level) in {"2", "3", _COMBINED_LEVEL} else "keyword"


def normalize_toolsearcher_scorer(
    value: str | None = None,
    level: str | int | None = None,
) -> str:
    scorer = (
        (value or os.environ.get("APIBANK_TOOLSEARCHER_SCORER") or default_toolsearcher_scorer(level)).strip().lower()
    )
    normalized = _TOOLSEARCHER_SCORER_ALIASES.get(scorer)
    if normalized is None:
        raise ValueError(
            f"unsupported APIBANK_TOOLSEARCHER_SCORER={scorer!r}; expected 'official', 'upstream', or 'keyword'"
        )
    return normalized


def toolsearcher_scorer(level: str | int | None = None) -> str:
    return normalize_toolsearcher_scorer(level=level)


def apibank_root() -> Path:
    """Return the API-Bank upstream root, allowing env override."""
    candidates = [
        os.environ.get("APIBANK_ROOT"),
        str(_DEFAULT_UPSTREAM_ROOT),
        str(_CACHE_UPSTREAM_ROOT),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if (root / "apis").is_dir() and (root / "init_database").is_dir():
            return root
    raise FileNotFoundError(
        "API-Bank source not found. Set APIBANK_ROOT to an API-Bank checkout (ships under benchmarks/apibank/apibank_upstream)."
    )


def curated_path(level: str | int | None = None) -> Path:
    override = os.environ.get("APIBANK_CURATED_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_CURATED_PATHS[normalize_level(level)].resolve()


# Upstream API-Bank code: API classes imported on demand, stubs for optional dependencies
def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _load_module_from_file(name: str, path: Path):
    with _IMPORT_LOCK:
        existing = sys.modules.get(name)
        if getattr(existing, "_apibank_loaded", False):
            return existing
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module._apibank_loaded = True
        return module


def _install_sklearn_metrics_stub_if_needed() -> None:
    """Avoid a broken local sklearn wheel blocking sentence_transformers import.

    The official API-Bank ToolSearcher imports ``sentence_transformers``. In
    the shared ``mas-promptbench`` environment, the installed scikit-learn wheel can be
    ABI-incompatible with numpy. ToolSearcher only needs ``SentenceTransformer``
    and ``util.cos_sim``; the sklearn imports are incidental. This stub supplies
    the small metrics surface imported by transformers/sentence_transformers so
    the official ToolSearcher code can run without mutating the conda env.
    """
    try:
        from sklearn.metrics import pairwise_distances  # type: ignore

        return
    except Exception:
        for name in list(sys.modules):
            if name == "sklearn" or name.startswith("sklearn."):
                sys.modules.pop(name, None)

    import numpy as np

    sklearn_mod = types.ModuleType("sklearn")
    sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
    metrics_mod = types.ModuleType("sklearn.metrics")
    metrics_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)

    def _as_2d(value):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def pairwise_distances(x, y=None, metric=None, **_kwargs):
        x_arr = _as_2d(x)
        y_arr = _as_2d(x if y is None else y)
        if metric == "cosine":
            x_norm = np.linalg.norm(x_arr, axis=1, keepdims=True)
            y_norm = np.linalg.norm(y_arr, axis=1, keepdims=True)
            denom = np.maximum(x_norm @ y_norm.T, 1e-12)
            return 1.0 - ((x_arr @ y_arr.T) / denom)
        diff = x_arr[:, None, :] - y_arr[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    metrics_mod.pairwise_distances = pairwise_distances
    metrics_mod.roc_curve = lambda *_args, **_kwargs: (np.array([]), np.array([]), np.array([]))
    metrics_mod.f1_score = lambda *_args, **_kwargs: 0.0
    metrics_mod.matthews_corrcoef = lambda *_args, **_kwargs: 0.0
    metrics_mod.accuracy_score = lambda *_args, **_kwargs: 0.0
    sklearn_mod.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod


def _install_googletrans_stub_if_needed() -> None:
    try:
        import googletrans  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    googletrans_mod = types.ModuleType("googletrans")
    googletrans_mod.__spec__ = importlib.machinery.ModuleSpec("googletrans", loader=None)
    googletrans_mod.LANGUAGES = {
        "en": "english",
        "zh-cn": "chinese",
        "zh-tw": "chinese traditional",
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "ja": "japanese",
        "ko": "korean",
    }

    class _Translation:
        def __init__(self, text: str):
            self.text = text

    class Translator:
        def translate(self, text: str, *_args, **_kwargs):
            return _Translation(text)

    googletrans_mod.Translator = Translator
    sys.modules["googletrans"] = googletrans_mod


def _install_nltk_stub_if_needed() -> None:
    try:
        from nltk.tokenize import word_tokenize  # type: ignore  # noqa: F401

        return
    except Exception:
        for name in list(sys.modules):
            if name == "nltk" or name.startswith("nltk."):
                sys.modules.pop(name, None)

    nltk_mod = types.ModuleType("nltk")
    nltk_mod.__spec__ = importlib.machinery.ModuleSpec("nltk", loader=None)
    tokenize_mod = types.ModuleType("nltk.tokenize")
    tokenize_mod.__spec__ = importlib.machinery.ModuleSpec("nltk.tokenize", loader=None)

    tokenize_mod.word_tokenize = lambda text: str(text).split()
    nltk_mod.download = lambda *_args, **_kwargs: True
    nltk_mod.tokenize = tokenize_mod
    sys.modules["nltk"] = nltk_mod
    sys.modules["nltk.tokenize"] = tokenize_mod


def _install_rank_bm25_stub_if_needed() -> None:
    try:
        import rank_bm25  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    import numpy as np

    rank_bm25_mod = types.ModuleType("rank_bm25")
    rank_bm25_mod.__spec__ = importlib.machinery.ModuleSpec("rank_bm25", loader=None)

    class BM25Okapi:
        def __init__(self, documents):
            self.documents = documents

        def get_scores(self, query):
            query_terms = set(query or [])
            return np.array([len(query_terms & set(document or [])) for document in self.documents], dtype=float)

    rank_bm25_mod.BM25Okapi = BM25Okapi
    sys.modules["rank_bm25"] = rank_bm25_mod


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class LazyApiBankManager:
    """Lazy subset of API-Bank's ToolManager.

    API-Bank's official ToolManager imports every API file at startup. That
    pulls optional dependencies for APIs we are not evaluating. This manager
    loads only the requested API classes while preserving official API call and
    correctness behavior.
    """

    def __init__(self, root: Path | None = None, api_subdir: str = "apis"):
        self.root = root or apibank_root()
        self.api_subdir = api_subdir
        self.package_name = api_subdir
        self.apis_dir = self.root / api_subdir
        self.base_apis_dir = self.root / "apis"
        self.init_database_dir = self.root / "init_database"
        self._api_base = None
        self._classes: dict[str, type] = {}
        self._tools: dict[str, Any] = {}
        self._init_databases = self._load_init_databases()
        self._ensure_api_package()
        self._ensure_selected_api_package()
        if (self.apis_dir / "check_token.py").exists():
            self.token_checker = self.init_tool("CheckToken")
        else:
            self.token_checker = None

    def _load_init_databases(self) -> dict[str, Any]:
        databases: dict[str, Any] = {}
        for path in self.init_database_dir.glob("*.json"):
            with path.open(encoding="utf-8") as f:
                databases[path.stem] = json.load(f)
        return databases

    def _ensure_api_package(self) -> None:
        api_mod = _load_module_from_file("apis.api", self.base_apis_dir / "api.py")
        package = sys.modules.get("apis")
        if package is None or not getattr(package, "_apibank_package", False):
            package = types.ModuleType("apis")
            package.__path__ = [str(self.base_apis_dir)]
            package.__package__ = "apis"
            package._apibank_package = True
            sys.modules["apis"] = package
        package.API = api_mod.API
        self._api_base = api_mod.API

    def _ensure_selected_api_package(self) -> None:
        if self.package_name == "apis":
            return
        package = sys.modules.get(self.package_name)
        if package is not None and getattr(package, "_apibank_package", False):
            return
        package = types.ModuleType(self.package_name)
        package.__path__ = [str(self.apis_dir)]
        package.__package__ = self.package_name
        package._apibank_package = True
        sys.modules[self.package_name] = package

    def _load_class(self, api_name: str) -> type:
        if api_name in self._classes:
            return self._classes[api_name]
        file_stem = "tool_search" if api_name == "ToolSearcher" else _snake_case(api_name)
        file_path = self.apis_dir / f"{file_stem}.py"
        candidate_paths = [file_path] if file_path.exists() else []
        if not candidate_paths:
            candidate_paths = [
                path for path in sorted(self.apis_dir.glob("*.py")) if path.name not in {"__init__.py", "api.py"}
            ]
        api_class = None
        for candidate_path in candidate_paths:
            module_name = f"{self.package_name}.{candidate_path.stem}"
            module = _load_module_from_file(module_name, candidate_path)
            api_class = getattr(module, api_name, None)
            if not isinstance(api_class, type):
                for value in module.__dict__.values():
                    if (
                        isinstance(value, type)
                        and self._api_base is not None
                        and issubclass(value, self._api_base)
                        and value is not self._api_base
                        and value.__name__ == api_name
                    ):
                        api_class = value
                        break
            if isinstance(api_class, type):
                break
        if not isinstance(api_class, type):
            raise KeyError(f"API class {api_name!r} not found under {self.apis_dir}")
        self._classes[api_name] = api_class
        return api_class

    def get_api_description(self, api_name: str) -> str:
        if api_name == "ToolSearcher":
            payload = {
                "name": "ToolSearcher",
                "description": "Searches for relevant tools in the API library based on concise keywords.",
                "input_parameters": {
                    "keywords": {
                        "type": "str",
                        "description": "Concise keywords describing the user request, such as 'add schedule'.",
                    }
                },
                "output_parameters": {
                    "best_matchs": {
                        "type": "Union[List[dict], dict]",
                        "description": "The best matching API description or descriptions.",
                    }
                },
            }
            return json.dumps(payload, ensure_ascii=False)
        api_class = self._load_class(api_name)
        payload = {
            "name": api_name,
            "description": getattr(api_class, "description", ""),
            "input_parameters": getattr(api_class, "input_parameters", {}),
            "output_parameters": getattr(api_class, "output_parameters", {}),
        }
        return json.dumps(payload, ensure_ascii=False)

    def init_tool(self, api_name: str):
        if api_name in self._tools:
            return self._tools[api_name]
        api_class = self._load_class(api_name)
        args = []
        database_name = getattr(api_class, "database_name", None)
        if database_name in self._init_databases:
            args.append(copy.deepcopy(self._init_databases[database_name]))
        input_parameters = getattr(api_class, "input_parameters", {})
        if api_name != "CheckToken" and "token" in input_parameters and self.token_checker:
            args.append(self.token_checker)
        if api_name == "ToolSearcher":
            with _pushd(self.root):
                tool = api_class(*args)
        else:
            tool = api_class(*args)
        self._tools[api_name] = tool
        return tool

    def api_call(self, api_name: str, **kwargs):
        api_class = self._load_class(api_name)
        input_parameters = getattr(api_class, "input_parameters", {})
        processed = {}
        for key, value in kwargs.items():
            if key not in input_parameters:
                raise AssertionError(f"invalid parameter name. parameter: {key}")
            required_type = input_parameters[key].get("type")
            if required_type == "int":
                processed[key] = int(value)
            elif required_type == "float":
                processed[key] = float(value)
            elif required_type == "bool":
                processed[key] = value if isinstance(value, bool) else str(value) == "True"
            else:
                processed[key] = value
        if api_name == "ToolSearcher":
            with _pushd(self.root):
                return self.init_tool(api_name).call(**processed)
        return self.init_tool(api_name).call(**processed)


# API calls: extraction and parsing
def _normalize_value(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{(":
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _normalize_value(value) for key, value in (params or {}).items()}


def _iter_api_call_candidates(text: str | None) -> list[str]:
    """Return bracketed API-call candidates using quote/bracket balancing."""
    raw = text or ""
    candidates: list[str] = []
    for match in _ANSWER_CALL_START_RE.finditer(raw):
        start = match.start()
        square_depth = 0
        paren_depth = 0
        quote: str | None = None
        escape = False
        for idx in range(start, len(raw)):
            char = raw[idx]
            if quote is not None:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
            elif char == "[":
                square_depth += 1
            elif char == "]":
                square_depth -= 1
                if square_depth == 0 and paren_depth == 0:
                    candidates.append(raw[start : idx + 1].strip())
                    break
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
    return candidates


def _looks_parseable_api_call(call: str) -> bool:
    match = _ANSWER_CALL_RE.fullmatch((call or "").strip())
    if not match:
        return False
    args_text = match.group("args").strip()
    if not args_text:
        return True
    try:
        expr = ast.parse(f"_f({args_text})", mode="eval").body
    except SyntaxError:
        return False
    return isinstance(expr, ast.Call) and not expr.args


def extract_api_call(text: str | None) -> str:
    candidates = _iter_api_call_candidates(text)
    for candidate in reversed(candidates):
        if _looks_parseable_api_call(candidate):
            return candidate
    return candidates[-1] if candidates else (text or "").strip()


def parse_api_call(text: str | None) -> tuple[str, dict[str, Any]]:
    call = extract_api_call(text)
    match = _ANSWER_CALL_RE.fullmatch(call)
    if not match:
        raise ValueError("no [ApiName(...)] call found")
    name = match.group("name")
    args_text = match.group("args").strip()
    if not args_text:
        return name, {}
    try:
        expr = ast.parse(f"_f({args_text})", mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"invalid API call syntax: {exc.msg}") from exc
    if not isinstance(expr, ast.Call) or expr.args:
        raise ValueError("API call must use keyword arguments only")
    params: dict[str, Any] = {}
    for keyword in expr.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs are not supported in API calls")
        params[keyword.arg] = _literal_or_name(keyword.value)
    return name, normalize_params(params)


def _literal_or_name(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.literal_eval(node)
    except Exception as exc:
        raise ValueError(f"unsupported argument expression: {ast.dump(node)}") from exc


def _api_call_string(api_name: str, params: dict[str, Any]) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in (params or {}).items())
    return f"[{api_name}({args})]"


# Tasks: the dialogue before one API call
def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _level3_json_path(root: Path | None = None) -> Path:
    root = root or apibank_root()
    candidates = [
        root / "test_data" / "level-3.json",
        root / "test-data" / "level-3.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "API-Bank Level-3 JSON not found. Restore benchmarks/apibank/apibank_upstream "
        "or set APIBANK_LEVEL3_JSON to a local level-3.json file."
    )


def _load_level3_json(root: Path | None = None) -> list[dict]:
    override = os.environ.get("APIBANK_LEVEL3_JSON")
    path = Path(override).expanduser().resolve() if override else _level3_json_path(root)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list in API-Bank Level-3 JSON: {path}")
    return data


def _api_positions(chat_history: list[dict]) -> list[int]:
    return [idx for idx, row in enumerate(chat_history) if row.get("role") == "API"]


def _level3_api_row(api_step: dict) -> dict:
    return {
        "role": "API",
        "api_name": api_step.get("api_name"),
        "param_dict": normalize_params(api_step.get("input") or {}),
        "result": copy.deepcopy(api_step.get("output")),
    }


def _level3_task_from_json(sample: dict, sample_id: int, api_id: int) -> dict | None:
    apis = sample.get("apis") or []
    if api_id < 0 or api_id >= len(apis):
        return None
    ground = _level3_api_row(apis[api_id])
    api_names = sorted({item.get("api_name") for item in apis if item.get("api_name")})
    prior_api_turns = [_level3_api_row(item) for item in apis[:api_id]]
    chat_history = [{"role": "User", "text": str(sample.get("requirement") or "").strip()}]
    chat_history.extend(prior_api_turns)
    task_id = f"level-3.json::{sample_id}::{api_id}"
    return {
        "id": task_id,
        "idx": task_id,
        "level": "3",
        "level_name": level_dir("3"),
        "file": "level-3.json",
        "sample_id": sample_id,
        "api_id": api_id,
        "api_index": api_id,
        "api_names": api_names,
        "requirement": sample.get("requirement"),
        "final_response": sample.get("response"),
        "chat_history": chat_history,
        "prior_api_turns": prior_api_turns,
        "ground_truth": ground,
        "gold_api_call": _api_call_string(ground["api_name"], normalize_params(ground.get("param_dict") or {})),
    }


def _task_from_file(
    root: Path,
    filename: str,
    sample_id: int,
    level: str | int | None = None,
    api_id: int | None = None,
) -> dict | None:
    level_key = normalize_level(level)
    if level_key == "3":
        data = _load_level3_json(root)
        if sample_id < 0 or sample_id >= len(data):
            return None
        return _level3_task_from_json(data[sample_id], sample_id, int(api_id or 0))
    path = root / level_dir(level_key) / filename
    rows = _read_jsonl(path)
    positions = _api_positions(rows)
    if sample_id % 2 != 0:
        return None
    api_index = sample_id // 2
    if api_index >= len(positions):
        return None
    api_pos = positions[api_index]
    ground = rows[api_pos]
    api_names = sorted({row.get("api_name") for row in rows if row.get("role") == "API"})
    prior_api_turns = [row for row in rows[:api_pos] if row.get("role") == "API"]
    task_id = f"{filename}::{sample_id}"
    return {
        "id": task_id,
        "idx": task_id,
        "level": level_key,
        "level_name": level_dir(level_key),
        "file": filename,
        "sample_id": sample_id,
        "api_index": api_index,
        "api_names": api_names,
        "chat_history": rows[:api_pos],
        "prior_api_turns": prior_api_turns,
        "ground_truth": ground,
        "gold_api_call": _api_call_string(ground["api_name"], normalize_params(ground.get("param_dict") or {})),
    }


def _combined_task_id(level_key: str, task_id: str) -> str:
    return f"level-{level_key}::{task_id}"


def _parse_combined_task_id(task_id: str) -> tuple[str, str]:
    """Return ``(level, source_id)`` for a combined-manifest task id."""
    if "::" in task_id:
        prefix, rest = task_id.split("::", 1)
        try:
            level_key = normalize_level(prefix)
        except ValueError:
            level_key = None
        if level_key in _LEVEL_KEYS:
            return level_key, rest
    if task_id.startswith("level-3.json::"):
        return "3", task_id
    return "", task_id


def _task_from_manifest_id(root: Path, task_id: str, level_key: str) -> dict | None:
    if level_key == _COMBINED_LEVEL:
        sublevel, source_id = _parse_combined_task_id(task_id)
        candidate_levels = [sublevel] if sublevel else list(_LEVEL_KEYS)
        for candidate_level in candidate_levels:
            task = _task_from_manifest_id(root, source_id, candidate_level)
            if task is None:
                continue
            combined_id = task_id if sublevel else _combined_task_id(candidate_level, source_id)
            task = copy.deepcopy(task)
            task["source_id"] = source_id
            task["id"] = combined_id
            task["idx"] = combined_id
            return task
        return None

    parts = task_id.split("::")
    if level_key == "3":
        if len(parts) != 3:
            return None
        filename, sample_id, api_id = parts
        return _task_from_file(root, filename, int(sample_id), level=level_key, api_id=int(api_id))

    filename, sample_id = task_id.rsplit("::", 1)
    return _task_from_file(root, filename, int(sample_id), level=level_key)


def iter_level_tasks(level: str | int | None = None, root: Path | None = None) -> list[dict]:
    level_key = normalize_level(level)
    root = root or apibank_root()
    if level_key == _COMBINED_LEVEL:
        tasks = []
        for sublevel in _LEVEL_KEYS:
            for task in iter_level_tasks(level=sublevel, root=root):
                task = copy.deepcopy(task)
                source_id = task["id"]
                task["source_id"] = source_id
                task["id"] = _combined_task_id(sublevel, source_id)
                task["idx"] = task["id"]
                tasks.append(task)
        return tasks
    if level_key == "3":
        tasks = []
        for sample_id, sample in enumerate(_load_level3_json(root)):
            for api_id, _api in enumerate(sample.get("apis") or []):
                task = _level3_task_from_json(sample, sample_id, api_id)
                if task:
                    tasks.append(task)
        return tasks
    tasks = []
    for path in sorted((root / level_dir(level_key)).glob("*.jsonl")):
        rows = _read_jsonl(path)
        for api_index, _api_pos in enumerate(_api_positions(rows)):
            sample_id = api_index * 2
            task = _task_from_file(root, path.name, sample_id, level=level_key)
            if task:
                tasks.append(task)
    return tasks


# Scoring: replay the dialogue's calls, then the predicted one, through the upstream API classes
def _required_apis(task: dict) -> set[str]:
    names = set(task.get("api_names") or [])
    for row in task.get("prior_api_turns") or []:
        if row.get("api_name"):
            names.add(row["api_name"])
    ground = task.get("ground_truth") or {}
    if ground.get("api_name"):
        names.add(ground["api_name"])
    return names


def _manager_for_level(level: str | int | None = None) -> LazyApiBankManager:
    return LazyApiBankManager(api_subdir="lv3_apis" if normalize_level(level) == "3" else "apis")


def _replay_prior(manager: LazyApiBankManager, task: dict) -> None:
    for row in task.get("prior_api_turns") or []:
        if row.get("api_name") == "ToolSearcher":
            continue
        manager.api_call(row["api_name"], **normalize_params(row.get("param_dict") or {}))


def _normalized_toolsearch_keywords(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _score_toolsearcher_keyword(params: dict[str, Any], ground: dict) -> dict:
    predicted_keywords = _normalized_toolsearch_keywords(params.get("keywords"))
    gold_keywords = _normalized_toolsearch_keywords((ground.get("param_dict") or {}).get("keywords"))
    correct = predicted_keywords == gold_keywords
    return {
        "correct": correct,
        "stage": "score",
        "error": None if correct else f"ToolSearcher keyword mismatch: {predicted_keywords!r} != {gold_keywords!r}",
        "predicted_api_name": "ToolSearcher",
        "predicted_params": params,
        "execution_result": copy.deepcopy(ground.get("result")) if correct else None,
    }


def _score_toolsearcher_upstream(params: dict[str, Any], ground: dict) -> dict:
    if os.environ.get("APIBANK_TOOLSEARCHER_SKLEARN_SHIM", "1") != "0":
        _install_sklearn_metrics_stub_if_needed()
    if os.environ.get("APIBANK_TOOLSEARCHER_GOOGLETRANS_SHIM", "1") != "0":
        _install_googletrans_stub_if_needed()
    if os.environ.get("APIBANK_TOOLSEARCHER_NLTK_SHIM", "1") != "0":
        _install_nltk_stub_if_needed()
    if os.environ.get("APIBANK_TOOLSEARCHER_BM25_SHIM", "1") != "0":
        _install_rank_bm25_stub_if_needed()
    manager = LazyApiBankManager()
    try:
        result = manager.api_call("ToolSearcher", **params)
        api = manager.init_tool("ToolSearcher")
        correct = bool(
            api.check_api_call_correctness(
                copy.deepcopy(result),
                copy.deepcopy(ground["result"]),
            )
        )
    except Exception as exc:
        return {
            "correct": False,
            "stage": "score",
            "error": f"{type(exc).__name__}: {exc}",
            "predicted_api_name": "ToolSearcher",
            "predicted_params": params,
            "execution_result": None,
        }
    return {
        "correct": correct,
        "stage": "score",
        "error": None if correct else "ToolSearcher official API result mismatch",
        "predicted_api_name": "ToolSearcher",
        "predicted_params": params,
        "execution_result": result,
    }


def _split_api_name(name: str) -> str:
    return "".join([" " + char.lower() if char.isupper() else char for char in name]).strip()


def _toolsearch_item_desc(item: dict[str, Any]) -> str:
    return str(
        item.get("desc_for_search") or (_split_api_name(str(item.get("name", ""))) + str(item.get("description", "")))
    )


def _collect_frozen_toolsearcher_data(root: Path, level: str | int | None = None) -> dict[str, Any]:
    """The frozen ToolSearcher of one level, built once per (root, level) and cached.

    Returns the API ``catalog`` (name -> API description), the recorded ToolSearcher
    ``outputs_by_keyword``, the ``search_items`` with their ``desc_for_search`` texts,
    their ``desc_embeddings`` and the sentence-embedding ``model`` that produced them.
    Level 3 reads the catalog from the ``lv3_apis`` classes; levels 1 and 2 read the
    catalog and the outputs from the Level-2 dialogues.
    """
    level_key = normalize_level(level)
    cache_key = f"{root.resolve()}::{level_key}"
    with _IMPORT_LOCK:
        cached = _FROZEN_TOOLSEARCHER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if level_key == "3":
            catalog = _level3_api_catalog(root)
            outputs_by_keyword = _level3_toolsearcher_outputs(root)
        else:
            catalog, outputs_by_keyword = _level2_toolsearcher_records(root)
        search_items = []
        for item in catalog.values():
            search_item = copy.deepcopy(item)
            search_item["desc_for_search"] = _toolsearch_item_desc(search_item)
            search_items.append(search_item)

        if not search_items:
            raise RuntimeError(f"no frozen ToolSearcher API catalog found for Level-{level_key}")
        if level_key == "2" and "GetUserToken" not in catalog:
            raise RuntimeError("frozen ToolSearcher catalog does not contain GetUserToken")

        model = _toolsearcher_model()
        data = {
            "model": model,
            "search_items": search_items,
            "desc_embeddings": [model.encode(item["desc_for_search"]) for item in search_items],
            "catalog": catalog,
            "outputs_by_keyword": outputs_by_keyword,
        }
        _FROZEN_TOOLSEARCHER_CACHE[cache_key] = data
        return data


def _level3_api_catalog(root: Path) -> dict[str, dict[str, Any]]:
    """Name, description and parameters of every API class in ``lv3_apis`` (first definition wins)."""
    manager = LazyApiBankManager(root=root, api_subdir="lv3_apis")
    api_base = manager._api_base
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "lv3_apis").glob("*.py")):
        if path.name in {"__init__.py", "api.py", "tool_search.py"}:
            continue
        module = _load_module_from_file(f"lv3_apis.{path.stem}", path)
        for value in module.__dict__.values():
            if api_base is None or not isinstance(value, type) or not issubclass(value, api_base) or value is api_base:
                continue
            catalog.setdefault(
                value.__name__,
                {
                    "name": value.__name__,
                    "description": getattr(value, "description", ""),
                    "input_parameters": getattr(value, "input_parameters", {}),
                    "output_parameters": getattr(value, "output_parameters", {}),
                },
            )
    return catalog


def _level3_toolsearcher_outputs(root: Path) -> dict[str, Any]:
    """The recorded output of each Level-3 ToolSearcher call, keyed by its normalized keywords (first wins)."""
    outputs_by_keyword: dict[str, Any] = {}
    for sample in _load_level3_json(root):
        for row in sample.get("apis") or []:
            if row.get("api_name") != "ToolSearcher":
                continue
            keyword = _normalized_toolsearch_keywords((row.get("input") or {}).get("keywords"))
            result = row.get("output") or {}
            outputs_by_keyword.setdefault(keyword, copy.deepcopy(result.get("output")))
    return outputs_by_keyword


def _level2_toolsearcher_records(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The Level-2 dialogues' ToolSearcher calls: the APIs they returned by name, and their outputs by keywords."""
    catalog: dict[str, dict[str, Any]] = {}
    outputs_by_keyword: dict[str, Any] = {}
    for path in sorted((root / level_dir("2")).glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("role") != "API" or row.get("api_name") != "ToolSearcher":
                continue
            keyword = _normalized_toolsearch_keywords((row.get("param_dict") or {}).get("keywords"))
            result = row.get("result") or {}
            output = copy.deepcopy(result.get("output"))
            outputs_by_keyword.setdefault(keyword, output)
            for item in output if isinstance(output, list) else [output]:
                if isinstance(item, dict) and item.get("name"):
                    catalog.setdefault(str(item["name"]), copy.deepcopy(item))
    return catalog, outputs_by_keyword


def _toolsearcher_model() -> Any:
    """The sentence-embedding model of the frozen ToolSearcher (``APIBANK_TOOLSEARCHER_MODEL`` / ``_DEVICE``)."""
    if os.environ.get("APIBANK_TOOLSEARCHER_SKLEARN_SHIM", "1") != "0":
        _install_sklearn_metrics_stub_if_needed()
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("APIBANK_TOOLSEARCHER_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")
    return SentenceTransformer(model_name, device=os.environ.get("APIBANK_TOOLSEARCHER_DEVICE", "cpu"))


def _frozen_toolsearcher_result(params: dict[str, Any], level: str | int | None = None) -> dict[str, Any]:
    level_key = normalize_level(level)
    if os.environ.get("APIBANK_TOOLSEARCHER_SKLEARN_SHIM", "1") != "0":
        _install_sklearn_metrics_stub_if_needed()
    from sentence_transformers import util

    keywords = str(params.get("keywords", ""))
    input_parameters = {"keywords": keywords}
    data = _collect_frozen_toolsearcher_data(apibank_root(), level_key)
    normalized_keywords = _normalized_toolsearch_keywords(keywords)

    if normalized_keywords in data["outputs_by_keyword"]:
        output = copy.deepcopy(data["outputs_by_keyword"][normalized_keywords])
    else:
        kw_emb = data["model"].encode(keywords)
        best_item = None
        best_match_score = 0.0
        for item, desc_embedding in zip(data["search_items"], data["desc_embeddings"]):
            cos_sim = util.cos_sim(kw_emb, desc_embedding).item()
            if cos_sim > best_match_score:
                best_item = item
                best_match_score = cos_sim
        if best_item is None:
            raise RuntimeError("ToolSearcher did not find a best-match API")
        best_output = copy.deepcopy(data["catalog"][best_item["name"]])
        if level_key == "2" and "token" in (best_output.get("input_parameters") or {}):
            output = [copy.deepcopy(data["catalog"]["GetUserToken"]), best_output]
        else:
            output = best_output

    return {
        "api_name": "ToolSearcher",
        "input": input_parameters,
        "output": output,
        "exception": None,
    }


def _score_toolsearcher_official(
    params: dict[str, Any],
    ground: dict,
    level: str | int | None = None,
) -> dict:
    try:
        result = _frozen_toolsearcher_result(params, level)
        correct = result.get("output") == (ground.get("result") or {}).get("output") and result.get("exception") == (
            ground.get("result") or {}
        ).get("exception")
    except Exception as exc:
        return {
            "correct": False,
            "stage": "score",
            "error": f"{type(exc).__name__}: {exc}",
            "predicted_api_name": "ToolSearcher",
            "predicted_params": params,
            "execution_result": None,
        }
    return {
        "correct": correct,
        "stage": "score",
        "error": None if correct else "ToolSearcher official frozen API result mismatch",
        "predicted_api_name": "ToolSearcher",
        "predicted_params": params,
        "execution_result": result,
    }


def _score_toolsearcher(
    params: dict[str, Any],
    ground: dict,
    level: str | int | None = None,
) -> dict:
    scorer = toolsearcher_scorer(level)
    if scorer == "official":
        return _score_toolsearcher_official(params, ground, level)
    if scorer == "upstream":
        return _score_toolsearcher_upstream(params, ground)
    return _score_toolsearcher_keyword(params, ground)


def score_prediction(task: dict, model_output: str | None) -> dict:
    """Score one model API call against one curated API-Bank task."""
    level_key = normalize_level(task.get("level") or APIBANK_LEVEL)
    try:
        api_name, params = parse_api_call(model_output)
    except Exception as exc:
        return {
            "correct": False,
            "stage": "parse",
            "error": f"{type(exc).__name__}: {exc}",
            "predicted_api_name": None,
            "predicted_params": {},
            "execution_result": None,
        }

    ground = task["ground_truth"]
    if api_name != ground["api_name"]:
        return {
            "correct": False,
            "stage": "api_name",
            "error": f"API name mismatch: {api_name} != {ground['api_name']}",
            "predicted_api_name": api_name,
            "predicted_params": params,
            "execution_result": None,
        }

    if ground["api_name"] == "ToolSearcher":
        return _score_toolsearcher(params, ground, task.get("level"))

    manager = _manager_for_level(level_key)
    try:
        _replay_prior(manager, task)
        result = manager.api_call(api_name, **params)
        api = manager.init_tool(api_name)
        # Some official API-Bank checkers mutate their input dictionaries
        # while comparing fuzzy fields. Keep scoring side-effect free because
        # team-size variants score several replicas against the same task.
        correct = bool(
            api.check_api_call_correctness(
                copy.deepcopy(result),
                copy.deepcopy(ground["result"]),
            )
        )
    except Exception as exc:
        return {
            "correct": False,
            "stage": "score",
            "error": f"{type(exc).__name__}: {exc}",
            "predicted_api_name": api_name,
            "predicted_params": params,
            "execution_result": None,
        }
    return {
        "correct": correct,
        "stage": "score",
        "error": None if correct else "API result mismatch",
        "predicted_api_name": api_name,
        "predicted_params": params,
        "execution_result": result,
    }


def gold_replay(task: dict) -> dict:
    return score_prediction(task, task["gold_api_call"])


# Curated manifests and loading
def _skip_apis_for_level(level: str | int | None = None) -> set[str]:
    level_key = normalize_level(level)
    return _SPECIAL_SKIP_APIS if level_key == "1" else _NONDETERMINISTIC_SKIP_APIS


def _load_json_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_or_build_single_level_manifest(level_key: str) -> dict:
    path = _DEFAULT_CURATED_PATHS[level_key].resolve()
    if path.exists():
        return _load_json_manifest(path)
    return build_curated_manifest(limit=_DEFAULT_LEVEL_LIMITS[level_key], level=level_key)


def build_curated_manifest(
    limit: int | None = None,
    root: Path | None = None,
    level: str | int | None = None,
) -> dict:
    level_key = normalize_level(level)
    if level_key == _COMBINED_LEVEL:
        selected = []
        level_counts: Counter[str] = Counter()
        for sublevel in _LEVEL_KEYS:
            manifest = _load_or_build_single_level_manifest(sublevel)
            for task_id in manifest.get("ids", []):
                selected.append(_combined_task_id(sublevel, str(task_id)))
                level_counts[sublevel] += 1
        if limit is not None:
            selected = selected[:limit]
            level_counts = Counter(_parse_combined_task_id(task_id)[0] for task_id in selected)
        return {
            "benchmark": benchmark_name(level_key),
            "level": level_dir(level_key),
            "toolsearcher_scorer": toolsearcher_scorer(level_key),
            "source": "AlibabaResearch/DAMO-ConvAI/api-bank",
            "root_hint": "Set APIBANK_ROOT to an API-Bank checkout (ships under benchmarks/apibank/apibank_upstream)",
            "selection": "combined replay-valid API-Bank Level-1, Level-2, and Level-3 curated tasks",
            "limit": len(selected) if limit is None else limit,
            "level_counts": dict(level_counts),
            "ids": selected,
            "excluded_summary": {},
            "excluded_examples": {},
        }

    limit = _DEFAULT_LEVEL_LIMITS[level_key] if limit is None else limit
    root = root or apibank_root()
    selected = []
    excluded = Counter()
    examples: dict[str, list[str]] = {}
    skip_apis = _skip_apis_for_level(level_key)
    for task in iter_level_tasks(level=level_key, root=root):
        required = _required_apis(task)
        if required & skip_apis:
            reason = "special_dependency_api"
            excluded[reason] += 1
            examples.setdefault(reason, []).append(task["id"])
            continue
        replay = gold_replay(task)
        if not replay["correct"]:
            reason = replay.get("stage") or "gold_replay_failed"
            excluded[reason] += 1
            examples.setdefault(reason, []).append(task["id"])
            continue
        selected.append(task["id"])
        if len(selected) >= limit:
            break
    return {
        "benchmark": benchmark_name(level_key),
        "level": level_dir(level_key),
        "toolsearcher_scorer": toolsearcher_scorer(level_key),
        "source": "AlibabaResearch/DAMO-ConvAI/api-bank",
        "root_hint": "Set APIBANK_ROOT to an API-Bank checkout (ships under benchmarks/apibank/apibank_upstream)",
        "selection": f"first replay-valid Level-{level_key} API-call tasks in sorted file/sample order",
        "limit": limit,
        "ids": selected,
        "excluded_summary": dict(excluded),
        "excluded_examples": {k: v[:10] for k, v in examples.items()},
    }


def _load_manifest(level: str | int | None = None) -> dict:
    level_key = normalize_level(level)
    path = curated_path(level_key)
    if path.exists():
        return _load_json_manifest(path)
    return build_curated_manifest(
        limit=(int(os.environ["APIBANK_CURATED_LIMIT"]) if os.environ.get("APIBANK_CURATED_LIMIT") else None),
        level=level_key,
    )


def _matches_requested_id(manifest_id: str, wanted: set[str]) -> bool:
    if manifest_id in wanted:
        return True
    sublevel, source_id = _parse_combined_task_id(manifest_id)
    return bool(sublevel and source_id in wanted)


def load_instances(
    limit: int | None = None,
    offset: int = 0,
    only: list[str | int] | None = None,
    level: str | int | None = None,
) -> list[dict]:
    level_key = normalize_level(level)
    root = apibank_root()
    manifest = _load_manifest(level_key)
    ids = [str(item) for item in manifest.get("ids", [])]
    if only:
        wanted = {str(item) for item in only}
        ids = [item for item in ids if _matches_requested_id(item, wanted)]
    ids = ids[offset:]
    if limit is not None:
        ids = ids[:limit]
    rows = []
    for task_id in ids:
        task = _task_from_manifest_id(root, task_id, level_key)
        if task is not None:
            rows.append(task)
    return rows


def dataset_summary(limit: int | None = None, level: str | int | None = None) -> dict:
    level_key = normalize_level(level)
    rows = load_instances(limit=limit, level=level_key)
    api_counts = Counter(row["ground_truth"]["api_name"] for row in rows)
    replay = [gold_replay(row) for row in rows]
    return {
        "benchmark": benchmark_name(level_key),
        "level": level_dir(level_key),
        "toolsearcher_scorer": toolsearcher_scorer(level_key),
        "root": str(apibank_root()),
        "curated_path": str(curated_path(level_key)),
        "num_samples": len(rows),
        "unique_ids": len({row["id"] for row in rows}),
        "level_counts": dict(Counter(row.get("level") for row in rows)),
        "api_counts": dict(api_counts),
        "gold_replay_correct": sum(1 for item in replay if item["correct"]),
        "gold_replay_failures": [
            {"id": row["id"], "stage": item.get("stage"), "error": item.get("error")}
            for row, item in zip(rows, replay)
            if not item["correct"]
        ][:10],
    }


# Prompts
_GENERIC_PROMPT = (
    "You are solving API-Bank API-call tasks. Given a dialogue "
    "history and API descriptions, emit the exact next API call with "
    "the correct API name and keyword arguments."
)
HARD_RULE = (
    "\n\nAPI-BANK HARD RULE: Every evaluated row requires one complete, "
    "parseable bracketed API call. Never answer with a refusal, blocker, "
    "analysis-only response, or missing final call. If a required value "
    "appears unavailable, still choose the best next API and fill every "
    "required keyword with the best-supported value; use 'UNKNOWN' only "
    "when no value can be inferred. Wrong arguments are model errors, but "
    "unparseable or absent calls are invalid."
)


def system_prompt(topology: str, role: str, style: str, prompt_suffix: str = "") -> str:
    """System prompt of ``role``: its prompt file (else a generic prompt naming ``style``),
    the hard rule and ``prompt_suffix``, under the role's protected output contract."""
    path = PROMPTS_DIR / topology / DATASET / f"{role}.txt"
    if path.exists():
        prompt = path.read_text().strip()
    else:
        prompt = _GENERIC_PROMPT + f"\n\nImplementation style: {style}."
    prompt = prompt.rstrip() + HARD_RULE
    if prompt_suffix:
        prompt = prompt.rstrip() + "\n" + prompt_suffix
    return append_output_contract(prompt, DATASET, topology, role)


def format_chat_history(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        role = row.get("role")
        if role == "User":
            lines.append(f"USER: {row.get('text', '')}")
        elif role == "AI":
            lines.append(f"ASSISTANT: {row.get('text', '')}")
        elif role == "API":
            output = row.get("result", {}).get("output")
            if row.get("api_name") == "ToolSearcher":
                output = _compact_toolsearch_output(output)
            compact = {
                "api_name": row.get("api_name"),
                "input": row.get("result", {}).get("input"),
                "output": output,
                "exception": row.get("result", {}).get("exception"),
            }
            lines.append("API_RESULT: " + json.dumps(compact, ensure_ascii=False, default=str))
    return "\n".join(lines)


def _compact_toolsearch_output(output: Any) -> Any:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: value for key, value in item.items() if key != "desc_for_search"}
        return item

    if isinstance(output, list):
        return [clean(item) for item in output]
    return clean(output)


def _description_names_for_task(task: dict) -> list[str]:
    level_key = normalize_level(task.get("level") or APIBANK_LEVEL)
    if level_key in {"2", "3"}:
        return ["ToolSearcher"]
    return [
        api_name for api_name in sorted(task.get("api_names") or []) if api_name not in _skip_apis_for_level(level_key)
    ]


def format_prompt(task: dict) -> str:
    """User message for one task: the API descriptions, the dialogue so far and the answer format."""
    manager = LazyApiBankManager()
    descriptions = [manager.get_api_description(api_name) for api_name in _description_names_for_task(task)]
    level_key = normalize_level(task.get("level") or APIBANK_LEVEL)
    level_guidance = ""
    if level_key == "2":
        level_guidance = (
            "\nFor Level 2, use ToolSearcher first when the dialogue has not yet "
            "searched for a tool. If ToolSearcher output already appears in the "
            "history, use the returned API descriptions from that API_RESULT."
        )
    elif level_key == "3":
        level_guidance = (
            "\nFor Level 3, the user requirement may require multiple API calls. "
            "Predict only the next API call for the current step. Use ToolSearcher "
            "when the needed tool has not yet been searched. If ToolSearcher output "
            "already appears in the history, use those returned API descriptions."
        )
    return (
        "Predict the next API call in the dialogue.\n\n"
        "AVAILABLE API DESCRIPTIONS:\n"
        + "\n".join(descriptions)
        + level_guidance
        + "\n\nDIALOGUE HISTORY BEFORE THE NEXT API CALL:\n"
        + format_chat_history(task.get("chat_history") or [])
        + "\n\nReturn exactly one API call in this format:\n"
        "[ApiName(arg1='value', arg2=123)]\n"
        "Do not include prose before or after the API call. If any required "
        "argument appears missing, still output a complete call and use the "
        "best-supported value or 'UNKNOWN'."
    )


def user_message(task: dict, extra_context: str = "") -> str:
    """:func:`format_prompt` plus the reports of other agents or earlier stages, if any."""
    message = format_prompt(task)
    if extra_context:
        message += "\n\nCONTEXT FROM OTHER AGENTS OR PRIOR STAGES:\n" + str(extra_context).strip()
    return message


# Tools: the row's APIs as callable function tools (OpenAI Agents SDK runner)
@dataclass(frozen=True)
class Tool:
    """One API as a function tool: JSON-schema ``parameters`` and ``call(arguments)``."""

    name: str
    description: str
    parameters: dict
    call: Callable[[dict], Any]


def api_tools(task: dict) -> list[Tool]:
    """One tool per API the row describes (Level 2/3: ToolSearcher).

    The tools share one manager built from the level's API library with the
    dialogue's prior API calls replayed; gold fields never participate.
    """
    public_task = {key: task[key] for key in ("level", "api_names", "prior_api_turns") if key in task}
    level = normalize_level(public_task.get("level") or APIBANK_LEVEL)
    manager = _manager_for_level(level)
    _replay_prior(manager, public_task)
    tools = []
    for name in _description_names_for_task(public_task):
        description = json.loads(manager.get_api_description(name))
        properties = {}
        required = []
        for parameter_name, parameter in (description.get("input_parameters") or {}).items():
            parameter = dict(parameter or {})
            properties[str(parameter_name)] = {
                "type": parameter.get("type", "string"),
                "description": parameter.get("description", ""),
            }
            required.append(str(parameter_name))
        tools.append(
            Tool(
                name=str(name),
                description=str(description.get("description") or ""),
                parameters={"type": "object", "properties": properties, "required": required},
                call=lambda arguments, api_name=str(name): manager.api_call(api_name, **dict(arguments)),
            )
        )
    return tools


# Model calls
def client(base_url: str) -> Any:
    """Raw OpenAI client: ``APIBANK_REQUEST_TIMEOUT`` (60 s) per request, ``APIBANK_OPENAI_MAX_RETRIES`` (0) retries."""
    return openai_client(
        base_url=base_url,
        timeout=float(os.environ.get("APIBANK_REQUEST_TIMEOUT", "60")),
        max_retries=int(os.environ.get("APIBANK_OPENAI_MAX_RETRIES", "0")),
    )


def complete(llm: Any, system: str, user: str, *, model: str, seed: int | None = None) -> dict:
    """One chat completion: ``{"messages", "raw", "solve_s", "telemetry"}``."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    telemetry = normalize(None)
    start = time.time()
    response = llm.chat.completions.create(messages=messages, **completion_kwargs(seed, model=model))
    openai_sdk_accumulate(telemetry, response)
    content = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": content})
    return {"messages": messages, "raw": content, "solve_s": time.time() - start, "telemetry": normalize(telemetry)}


# Agents: one scored model call each, and how a topology combines them
def answer_key(answer: str | None) -> str:
    """Voting key of an API call: its name and parsed parameters (the raw text if it does not parse)."""
    try:
        api_name, params = parse_api_call(answer)
    except Exception:
        return (answer or "").strip().lower()
    return json.dumps({"name": api_name, "params": params}, sort_keys=True, default=str)


def scored_call(task: dict, raw: str) -> dict:
    """The API call extracted from ``raw`` and its score."""
    predicted = extract_api_call(raw)
    scored = score_prediction(task, predicted)
    return {
        "raw": raw,
        "predicted_answer": predicted,
        "answer_key": answer_key(predicted),
        "answer_correct": int(bool(scored.get("correct"))),
        "correct": bool(scored.get("correct")),
        "stage": scored.get("stage"),
        "error": scored.get("error"),
        "predicted_api_name": scored.get("predicted_api_name"),
        "predicted_params": scored.get("predicted_params"),
    }


def solve_agent(solve: Callable[..., dict], task: dict, *, role: str, seed: int, **kwargs: Any) -> dict:
    """Run ``solve(task, role=, seed=, **kwargs)`` for one agent and score it; a failure becomes ``error``."""
    out, latency_s, error = attempt(lambda: solve(task, role=role, seed=seed, **kwargs))
    if error is not None:
        return {
            "role": role,
            "seed": seed,
            "solve_s": round(latency_s, 1),
            "error": error,
            "raw": "",
            "messages": [],
            "telemetry": {},
        }
    return {
        "role": role,
        "seed": seed,
        "solve_s": round(float(out.get("solve_s") or 0.0), 1),
        **scored_call(task, out.get("raw") or ""),
        "turns": 1,
        "tool_calls": 0,
        "messages": out.get("messages") or [],
        "telemetry": out.get("telemetry") or {},
    }


def answer_fields(agent: dict) -> dict:
    """A topology's prediction fields, taken from the agent whose call it submits."""
    return {
        "predicted_answer": agent.get("predicted_answer", ""),
        "predicted_api_name": agent.get("predicted_api_name"),
        "predicted_params": agent.get("predicted_params"),
        "answer_correct": int(bool(agent.get("correct"))),
        "correct": bool(agent.get("correct")),
        "stage": agent.get("stage"),
        "error": agent.get("error"),
    }


def _voters(agents: list[dict]) -> list[tuple[int, str]]:
    return [
        (idx, agent.get("answer_key") or "")
        for idx, agent in enumerate(agents)
        if str(agent.get("predicted_answer") or "").strip()
    ]


def vote_buckets(agents: list[dict]) -> dict[str, int]:
    """Votes per answer key among the agents that produced a call."""
    return dict(Counter(key for _, key in _voters(agents)))


def choose_winner(agents: list[dict]) -> int | None:
    """Index of the first agent with the most common call (None if no agent produced one)."""
    candidates = _voters(agents)
    if not candidates:
        return None
    counts = Counter(key for _, key in candidates)
    first_seen: dict[str, int] = {}
    for idx, key in candidates:
        first_seen.setdefault(key, idx)
    winner_key = max(counts, key=lambda key: (counts[key], -first_seen[key]))
    return first_seen[winner_key]


def compact_agent(agent: dict) -> dict:
    """An agent's entry in a topology output: without messages, telemetry and raw text."""
    return {key: value for key, value in agent.items() if key not in {"messages", "telemetry", "raw"}}


# Batch records
def _gold_fields(instance: dict) -> dict:
    return {
        "file": instance.get("file"),
        "sample_id": instance.get("sample_id"),
        "question": format_chat_history(instance.get("chat_history") or []),
        "gold_api_call": instance.get("gold_api_call"),
        "gold_api_name": instance.get("ground_truth", {}).get("api_name"),
    }


def _trace_path(out_dir: Path, iid: str) -> Path:
    return Path(out_dir) / "traces" / f"{iid.replace('/', '_').replace(':', '_')}.json"


def run_instance(instance: dict, out_dir: Path, *, style: str, solve: Callable[[], dict]) -> dict:
    """Record of one instance solved by ``solve()`` (a topology's output); writes ``traces/<id>.json``."""
    iid = instance["id"]
    summary: dict[str, Any] = {
        "id": iid,
        "idx": iid,
        "level": instance.get("level") or APIBANK_LEVEL,
        **_gold_fields(instance),
        "style": style,
    }
    try:
        out = solve()
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["stage"] = "solve"
        return summary

    summary["solve_s"] = round(float(out.get("solve_s") or 0.0), 1)
    summary["predicted_answer"] = out.get("predicted_answer", "")
    summary["predicted_api_name"] = out.get("predicted_api_name")
    summary["predicted_params"] = out.get("predicted_params")
    summary["answer_correct"] = int(bool(out.get("answer_correct")))
    summary["correct"] = bool(out.get("correct"))
    summary["stage"] = out.get("stage")
    summary["error"] = out.get("error")
    summary["turns"] = int(out.get("turns") or 0)
    summary["tool_calls"] = int(out.get("tool_calls") or 0)
    for key in agent_runs.OUTPUT_KEYS:
        if key in out:
            summary[key] = out[key]
    summary.update(out.get("telemetry") or {})
    trace = {"summary": summary, "messages": out.get("messages") or [], "ground_truth": instance.get("ground_truth")}
    agent_runs.write_json(_trace_path(out_dir, iid), trace)
    return summary


def _prediction(instance: dict, record: dict) -> dict:
    return {
        "idx": instance["id"],
        "id": instance["id"],
        "question": record.get("question"),
        "predicted_answer": record.get("predicted_answer"),
    }


def run_rows(
    instances: list[dict],
    run_one: Callable[[dict, Path], dict],
    *,
    style: str,
    model_id: str,
    out_dir: Path | None = None,
    predictions: Path | None = None,
    level: str | int | None = None,
    team_size: int | None = None,
    verbose: bool = True,
) -> dict:
    """:func:`core.agent_runs.run_rows` for API-Bank (``out_dir`` default ``results/apibank/<style>``)."""
    return agent_runs.run_rows(
        instances,
        run_one,
        dataset=DATASET,
        source=benchmark_name(level),
        style=style,
        model_id=model_id,
        prediction=_prediction,
        out_dir=out_dir,
        predictions=predictions,
        team_size=team_size,
        verbose=verbose,
    )


# Team sizes: majority vote over seeded replicas of one role
_NOT_IN_REPLICA = {"role", "predicted_api_name", "predicted_params", "messages"}


def solve_replicas(
    solve: Callable[..., dict], instance: dict, *, style: str, topology: str, role: str, n: int
) -> list[dict]:
    """``n`` scored calls of ``solve`` under ``role`` (seed = replica index)."""
    agents = [solve_agent(solve, instance, style=style, topology=topology, role=role, seed=seed) for seed in range(n)]
    return [{key: value for key, value in agent.items() if key not in _NOT_IN_REPLICA} for agent in agents]


def run_replicas(instance: dict, out_dir: Path, *, style: str, team_size: int, solve: Callable[[], list[dict]]) -> dict:
    """Record of one instance solved by ``solve()`` (the replicas); the winning call is re-scored."""
    iid = instance["id"]
    summary: dict[str, Any] = {
        "id": iid,
        "idx": iid,
        **_gold_fields(instance),
        "style": style,
        "team_size": team_size,
        "n_agents": team_size,
    }
    start = time.time()
    replicas = solve()
    summary["solve_s"] = round(time.time() - start, 1)
    summary["per_agent"] = [{key: value for key, value in agent.items() if key != "raw"} for agent in replicas]
    summary["buckets"] = vote_buckets(replicas)
    winner = choose_winner(replicas)
    summary["winner"] = winner
    if winner is None:
        summary["predicted_answer"] = ""
        summary["answer_correct"] = 0
        summary["correct"] = False
        summary["stage"] = "solve"
        summary["error"] = "all API-Bank replicas failed"
    else:
        predicted = replicas[winner].get("predicted_answer") or ""
        scored = score_prediction(instance, predicted)
        summary["predicted_answer"] = predicted
        summary["answer_correct"] = int(bool(scored.get("correct")))
        summary["correct"] = bool(scored.get("correct"))
        summary["stage"] = scored.get("stage")
        summary["error"] = scored.get("error")
    summary["tool_calls"] = 0
    summary["turns"] = len(replicas)
    summary.update(sum_telemetry(replicas))
    agent_runs.write_json(_trace_path(out_dir, iid), {"summary": summary, "per_agent": replicas})
    return summary


# Command line
DEFAULT_LIMIT = 2
SUMMARY = cli.InfoMode("--summary", dataset_summary)


def add_arguments(parser) -> None:
    """API-Bank options of the runners' command line."""
    parser.add_argument("--level", default=None, help="API-Bank slice: all, 1, 2, or 3")
    parser.add_argument("--curated-path", default=None)
    parser.add_argument(
        "--toolsearcher-scorer",
        choices=["official", "upstream", "keyword"],
        default=None,
        help="ToolSearcher scorer.",
    )


def configure(curated_path: str | None = None, toolsearcher_scorer: str | None = None) -> None:
    """Export ``--curated-path`` and ``--toolsearcher-scorer`` as ``APIBANK_CURATED_PATH`` / ``APIBANK_TOOLSEARCHER_SCORER``."""
    if curated_path:
        os.environ["APIBANK_CURATED_PATH"] = str(Path(curated_path).expanduser().resolve())
    if toolsearcher_scorer:
        os.environ["APIBANK_TOOLSEARCHER_SCORER"] = toolsearcher_scorer


def main(
    argv: list[str] | None,
    *,
    description: str,
    run_one: Callable[[dict, Path], dict],
    style: str,
    model_id: str,
    team_size: int | None = None,
    preflight: Callable[[], None] | None = None,
) -> int:
    """Command line of an API-Bank runner (:mod:`core.cli`); ``--summary`` prints :func:`dataset_summary`."""

    def run_batch(instances: list[dict], out_dir: Path | None = None, out_path: Path | None = None, level=None) -> dict:
        return run_rows(
            instances,
            run_one,
            style=style,
            model_id=model_id,
            out_dir=out_dir,
            predictions=out_path,
            level=level,
            team_size=team_size,
        )

    return cli.main(
        argv,
        description=description,
        load_instances=load_instances,
        run_batch=run_batch,
        add_arguments=add_arguments,
        default_limit=DEFAULT_LIMIT,
        info=SUMMARY,
        configure=configure,
        preflight=preflight,
    )
