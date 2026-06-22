"""Generic GEPA pilot for registered real-runner datasets."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
from dspy.teleprompt import GEPA

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WORKSPACE_ROOT.parent.parent
for path in (str(WORKSPACE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from real_runner_gepa.early_stop import build_early_stopper
from real_runner_gepa.lm import (
    REFL_MODEL,
    TASK_MODEL,
    build_reflection_lm,
    build_task_pool,
    reflection_endpoint,
    task_endpoints,
)
from real_runner_gepa.output_contracts import OUTPUT_CONTRACT_VERSION
from real_runner_gepa.programs import RealRunnerProgram
from real_runner_gepa.registry import datasets, topologies
from real_runner_gepa.datasets.split_utils import (
    exclude_real_eval_ids_enabled,
    real_eval_ids,
    swe_protected_sample_name,
)


LCB_COMPILED_PROMPT_CHAR_LIMIT = int(os.environ.get("LCB_COMPILED_PROMPT_CHAR_LIMIT", "24000"))
LCB_COMPILED_PROMPT_HEAD_CHARS = int(os.environ.get("LCB_COMPILED_PROMPT_HEAD_CHARS", "6000"))
HARD_INFRA_MARKERS = (
    "api connection error",
    "apiconnectionerror",
    "apierror",
    "api status error",
    "apistatuserror",
    "api timeout error",
    "apitimeouterror",
    "badrequesterror",
    "connecterror",
    "connection refused",
    "cuda out of memory",
    "error code: 400",
    "error code: 500",
    "exception in asgi",
    "expecting value: line 1 column 1",
    "importerror:",
    "input_tokens",
    "internalservererror",
    "jinja2>=3.1.0",
    "maximum context length",
    "modulenotfounderror:",
    "outofmemoryerror",
    "readtimeout",
    "remoteprotocolerror",
    "runtimeerror: cuda",
    "timeout while connecting",
    "tool failure",
    "traceback (most recent call last)",
    "valueerror: apply_chat_template",
    "wikipedia_search tool returned an error",
)


def strip_reflection_preamble(text: str) -> str:
    original = text
    saw_think = "</think>" in text.lower()
    idx = text.lower().rfind("</think>")
    if idx >= 0:
        text = text[idx + len("</think>"):]
    stripped = text.lstrip()
    if saw_think or stripped.startswith("blocks."):
        fences = list(re.finditer(r"(?m)^```[^\n]*\n", text))
        if fences:
            start = fences[-1].end()
            closing = re.search(r"(?m)^```\s*$", text[start:])
            end = start + closing.start() if closing else len(text)
            inner = text[start:end].strip()
            if inner:
                return inner + "\n"
    return (text or original).strip() + "\n"


def write_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    tmp.replace(path)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def update_status(out: Path, phase: str, **fields) -> None:
    payload = {
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        **fields,
    }
    write_json(out / "status.json", payload)
    print(json.dumps({"status": payload}, default=str), flush=True)


def load_dataset_module(dataset: str):
    return importlib.import_module(f"real_runner_gepa.datasets.{dataset}")


def parse_teamsizes_topology(topology: str) -> dict | None:
    match = re.fullmatch(r"(independent|decentralized|sequential|centralized)_r(2|4|8|10)", topology)
    if not match:
        return None
    return {
        "base_topology": match.group(1),
        "team_size": int(match.group(2)),
    }


def parse_communications_topology(topology: str) -> dict | None:
    match = re.fullmatch(
        r"(independent|decentralized|sequential|centralized)_communications_(freeform|semi_structured|structured_soft)",
        topology,
    )
    if not match:
        return None
    return {
        "base_topology": match.group(1),
        "communications_format": match.group(2),
    }


def build_program(dataset: str, topology: str, n_agents: int, n_rounds: int):
    kwargs = {}
    teamsizes = parse_teamsizes_topology(topology)
    if teamsizes is not None:
        kwargs["n_agents"] = teamsizes["team_size"]
        if teamsizes["base_topology"] == "decentralized":
            kwargs["n_rounds"] = n_rounds
        return RealRunnerProgram(dataset, topology, **kwargs)
    msg = parse_communications_topology(topology)
    if msg is not None:
        if msg["base_topology"] in {"independent", "decentralized"}:
            kwargs["n_agents"] = n_agents
        if msg["base_topology"] == "decentralized":
            kwargs["n_rounds"] = n_rounds
        return RealRunnerProgram(dataset, topology, **kwargs)
    if topology in {"independent", "decentralized", "decentralized_openai"}:
        kwargs["n_agents"] = n_agents
    if topology in {"decentralized", "decentralized_openai"}:
        kwargs["n_rounds"] = n_rounds
    return RealRunnerProgram(dataset, topology, **kwargs)


def row_level_counts(rows) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        level = getattr(row, "level", None)
        if level is not None:
            counts[str(level)] += 1
    return dict(counts)


def row_repo_counts(rows) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        repo = getattr(row, "repo", None)
        if repo is None:
            task_instance = getattr(row, "task_instance", None)
            if isinstance(task_instance, dict):
                repo = task_instance.get("repo")
        if repo is not None:
            counts[str(repo)] += 1
    return dict(counts)


def row_profile_counts(dataset_mod, rows) -> dict:
    profile_fn = getattr(dataset_mod, "split_profile_counts", None)
    if not callable(profile_fn):
        return {}
    return profile_fn(rows)


def split_manifest_metadata(dataset_mod, train_size: int, val_size: int, seed: int, offset: int) -> dict:
    metadata_fn = getattr(dataset_mod, "split_manifest_metadata", None)
    if not callable(metadata_fn):
        return {}
    return metadata_fn(train_size, val_size, seed, offset) or {}


def failure_summary(records: list[dict]) -> dict:
    infra_ids = []
    zero_score_ids = []
    for record in records:
        rid = str(record.get("id"))
        score = float(record.get("score") or 0.0)
        if score <= 0.0:
            zero_score_ids.append(rid)
        text = "\n".join(
            str(record.get(key) or "")
            for key in ("agent_trace", "answer", "predicted_answer")
        ).lower()
        if any(marker in text for marker in HARD_INFRA_MARKERS):
            infra_ids.append(rid)
    return {
        "n": len(records),
        "zero_score_count": len(zero_score_ids),
        "zero_score_ids": zero_score_ids,
        "infra_error_count": len(infra_ids),
        "infra_error_ids": infra_ids,
        "model_failure_count": max(len(zero_score_ids) - len(infra_ids), 0),
    }


TOOLHOP_MONTH_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) "
    r"\d{1,2}, \d{4}\b"
)
TOOLHOP_ISO_DATE_RE = re.compile(r"\b(?:18|19|20)\d{2}-\d{2}-\d{2}\b")
TOOLHOP_ANSWER_TAG_RE = re.compile(r"<answer>\s*([^<>]+?)\s*</answer>", re.IGNORECASE)
TOOLHOP_QUOTED_PROPER_NOUN_RE = re.compile(
    r"['\"][A-ZÀ-ÖØ-Þ][^\W\d_]*(?:-[^\W\d_]+)?"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:-[^\W\d_]+)?)+['\"]"
)
TOOLHOP_EXAMPLE_LINE_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:e\.g\.|example)\b.*$")
TOOLHOP_UNQUOTED_PROPER_NOUN_RE = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:-[^\W\d_]+)?"
    r"(?:\s+[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:-[^\W\d_]+)?)+\b"
)
TOOLHOP_PLACEHOLDER_RE = re.compile(
    r"\b(?:ENTITY|TARGET|ATTRIBUTE|RELATIONSHIP|VALUE|FINAL_VALUE|EXACT_VALUE|CORRECT_VALUE|"
    r"ENTITY_VALUE|TOOL_OUTPUT|DATE|YEAR|QUESTION|ANSWER|RESULT|INPUT|OUTPUT|FINAL_ANSWER)\b"
    r"|\b[A-Z][A-Z0-9_]{2,}\b|\[[^\]]+\]|\{[^}]+\}"
)
TOOLHOP_PROHIBITED_PHRASES = (
    "domain-specific knowledge",
    "the longest reigning british monarch is",
    "common film query patterns",
)


def _toolhop_answer_tag_is_placeholder(value: str) -> bool:
    normalized = value.strip()
    if not normalized or normalized in {"...", "…"}:
        return True
    return bool(TOOLHOP_PLACEHOLDER_RE.search(normalized))


def _toolhop_answer_replacement(match: re.Match) -> str:
    value = match.group(1).strip()
    if _toolhop_answer_tag_is_placeholder(value):
        return match.group(0)
    return "<answer>VALUE</answer>"


def sanitize_toolhop_compiled_prompt(text: str) -> str:
    """Remove train/val-specific examples while keeping reusable behavior guidance."""
    text = TOOLHOP_MONTH_DATE_RE.sub("DATE", text)
    text = TOOLHOP_ISO_DATE_RE.sub("DATE", text)
    text = TOOLHOP_QUOTED_PROPER_NOUN_RE.sub("'ENTITY_VALUE'", text)
    text = TOOLHOP_ANSWER_TAG_RE.sub(_toolhop_answer_replacement, text)

    sanitized_lines: list[str] = []
    drop_domain_block = False
    for line in text.splitlines():
        lowered = line.strip().lower()
        if "domain-specific knowledge" in lowered:
            drop_domain_block = True
            continue
        if drop_domain_block and lowered.startswith("## ") and "toolhop" in lowered:
            drop_domain_block = False
        if drop_domain_block:
            continue
        if any(phrase in lowered for phrase in TOOLHOP_PROHIBITED_PHRASES):
            continue
        exampleish = "e.g." in lowered or "example" in lowered
        has_concrete_value = (
            TOOLHOP_UNQUOTED_PROPER_NOUN_RE.search(line)
            or TOOLHOP_MONTH_DATE_RE.search(line)
            or TOOLHOP_ISO_DATE_RE.search(line)
            or any(
                not _toolhop_answer_tag_is_placeholder(match.group(1))
                for match in TOOLHOP_ANSWER_TAG_RE.finditer(line)
            )
        )
        if exampleish and has_concrete_value and not TOOLHOP_PLACEHOLDER_RE.search(line):
            continue
        sanitized_lines.append(line)
    return "\n".join(sanitized_lines).strip() + "\n"


def sanitize_lcb_compiled_prompt(text: str) -> str:
    """Cap compiled LCB prompts so eval requests stay inside model context."""
    text = strip_reflection_preamble(text).strip()
    if len(text) <= LCB_COMPILED_PROMPT_CHAR_LIMIT:
        return text + "\n"

    marker = (
        "\n\n[...compiled prompt truncated to stay under the model context "
        "limit; preserve general reusable guidance, not validation traces...]\n\n"
    )
    head_chars = min(LCB_COMPILED_PROMPT_HEAD_CHARS, max(0, LCB_COMPILED_PROMPT_CHAR_LIMIT // 2))
    tail_chars = max(0, LCB_COMPILED_PROMPT_CHAR_LIMIT - head_chars - len(marker))
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip() + "\n"


def sanitize_compiled_program_prompts(dataset: str, compiled) -> dict:
    if dataset not in {"toolhop", "lcb"}:
        return {"policy": "none", "changed_prompts": []}

    changed = []
    for name, predictor in compiled.named_predictors():
        before = predictor.signature.instructions or ""
        if dataset == "toolhop":
            after = sanitize_toolhop_compiled_prompt(before)
        else:
            after = sanitize_lcb_compiled_prompt(before)
        if after != before:
            predictor.signature = predictor.signature.with_instructions(after)
            if hasattr(predictor, "sync_to_adapter"):
                predictor.sync_to_adapter()
            changed.append(
                {
                    "prompt": name.replace("/", "_").replace(".", "_"),
                    "before_chars": len(before),
                    "after_chars": len(after),
                }
            )
    policy = (
        "toolhop_placeholder_sanitize_before_eval"
        if dataset == "toolhop"
        else "lcb_context_safety_cap_before_eval"
    )
    return {
        "policy": policy,
        "changed_prompts": changed,
        "changed_count": len(changed),
    }


def assess_compiled_prompt_quality(dataset: str, prompt_texts: dict[str, str]) -> dict:
    if dataset != "toolhop":
        return {"policy": "none", "reject_compiled_prompt": False, "issues": []}

    issues = []
    for name, text in prompt_texts.items():
        lowered = text.lower()
        if "example successful pattern" in lowered or "learn from successful examples" in lowered:
            issues.append(
                {
                    "prompt": name,
                    "type": "sample_success_pattern_section",
                    "detail": "ToolHop prompts must use placeholders instead of train/val trace examples",
                }
            )
        for phrase in TOOLHOP_PROHIBITED_PHRASES:
            if phrase in lowered:
                issues.append({"prompt": name, "type": "prohibited_phrase", "match": phrase})
        for issue_type, pattern in (
            ("concrete_month_date", TOOLHOP_MONTH_DATE_RE),
            ("concrete_iso_date", TOOLHOP_ISO_DATE_RE),
            ("quoted_proper_noun_example", TOOLHOP_QUOTED_PROPER_NOUN_RE),
        ):
            match = pattern.search(text)
            if match:
                issues.append({"prompt": name, "type": issue_type, "match": match.group(0)})
        for match in TOOLHOP_ANSWER_TAG_RE.finditer(text):
            value = match.group(1).strip()
            if not _toolhop_answer_tag_is_placeholder(value):
                issues.append(
                    {
                        "prompt": name,
                        "type": "concrete_answer_tag_example",
                        "match": match.group(0),
                    }
                )
        for raw_line in text.splitlines():
            line = raw_line.strip()
            lowered_line = line.lower()
            if not line or ("e.g." not in lowered_line and not TOOLHOP_EXAMPLE_LINE_RE.match(line)):
                continue
            if TOOLHOP_PLACEHOLDER_RE.search(line):
                continue
            if (
                re.search(r"['\"]|\b(?:18|19|20)\d{2}\b|<answer>", line)
                or TOOLHOP_UNQUOTED_PROPER_NOUN_RE.search(line)
            ):
                issues.append(
                    {
                        "prompt": name,
                        "type": "concrete_example_line",
                        "match": line[:200],
                    }
                )

    return {
        "policy": "toolhop_no_memorized_success_examples",
        "reject_compiled_prompt": bool(issues),
        "issues": issues,
    }


def prompt_acceptance_decision(
    dataset: str,
    baseline_score: float,
    compiled_score: float,
    baseline_records: list[dict],
    compiled_records: list[dict],
    prompt_quality: dict | None = None,
) -> tuple[bool, str, dict]:
    eps = 1e-9
    baseline_failures = failure_summary(baseline_records)
    compiled_failures = failure_summary(compiled_records)
    if compiled_failures.get("infra_error_count"):
        diagnostics = {
            "policy": "reject_compiled_on_infra_validation_failure",
            "baseline_failure_summary": baseline_failures,
            "compiled_failure_summary": compiled_failures,
            "prompt_quality": prompt_quality or {},
        }
        return (
            False,
            "compiled validation has infra errors; keep baseline prompt for final eval",
            diagnostics,
        )

    if dataset != "toolhop":
        accept = compiled_score + eps >= baseline_score
        return (
            accept,
            "compiled_score >= baseline_score"
            if accept
            else "compiled_score < baseline_score; keep baseline prompt for final eval",
            {
                "policy": "score_ge_baseline",
                "baseline_failure_summary": baseline_failures,
                "compiled_failure_summary": compiled_failures,
                "prompt_quality": prompt_quality or {},
            },
        )

    baseline_by_id = {str(record.get("id")): float(record.get("score") or 0.0) for record in baseline_records}
    compiled_by_id = {str(record.get("id")): float(record.get("score") or 0.0) for record in compiled_records}
    shared_ids = sorted(set(baseline_by_id) & set(compiled_by_id))
    improved_ids = [rid for rid in shared_ids if compiled_by_id[rid] > baseline_by_id[rid] + eps]
    regressed_ids = [rid for rid in shared_ids if compiled_by_id[rid] + eps < baseline_by_id[rid]]
    diagnostics = {
        "policy": "toolhop_score_gt_or_tie_without_row_regressions",
        "row_improved_count": len(improved_ids),
        "row_regressed_count": len(regressed_ids),
        "row_improved_ids": improved_ids,
        "row_regressed_ids": regressed_ids,
        "baseline_failure_summary": baseline_failures,
        "compiled_failure_summary": compiled_failures,
        "prompt_quality": prompt_quality or {},
    }
    if prompt_quality and prompt_quality.get("reject_compiled_prompt"):
        return (
            False,
            "toolhop compiled prompt failed anti-memorization quality checks; keep baseline prompt",
            diagnostics,
        )
    if compiled_score > baseline_score + eps:
        return True, "toolhop compiled_score > baseline_score", diagnostics
    if compiled_score + eps < baseline_score:
        return False, "toolhop compiled_score < baseline_score; keep baseline prompt for final eval", diagnostics
    if regressed_ids:
        return (
            False,
            "toolhop compiled_score tied baseline but regressed validation rows; keep baseline prompt",
            diagnostics,
        )
    if improved_ids:
        return (
            True,
            "toolhop compiled_score tied baseline with row-level improvements and no regressions",
            diagnostics,
        )
    return (
        False,
        "toolhop compiled_score tied baseline with no row-level gain; keep baseline prompt",
        diagnostics,
    )


def _evaluate_one(program, row, plain_metric):
    t0 = time.time()
    pred = program(task_instance=row.task_instance)
    score = float(plain_metric(row, pred))
    record = {
        "id": row.id,
        "score": score,
        "latency_s": round(time.time() - t0, 2),
        "answer": getattr(pred, "answer", None),
        "winner": getattr(pred, "winner", None),
        "vote_summary": getattr(pred, "vote_summary", None),
        "agent_trace": getattr(pred, "agent_trace", None),
    }
    for key in (
        "predicted_answer",
        "runner_correct",
        "runner_answer_correct",
        "scoring_prev_tool_content",
        "previous_tool_content",
        "communication_format",
        "communication_parse_ok",
        "communication_all_parse_ok",
        "communication_parse_rate",
        "communication_required_report_count",
        "communication_missing_roles",
        "communication_infra_error",
        "communication_parse_errors",
        "communication_parse_warnings",
        "communication_report_ok_count",
        "communication_report_total",
        "communication_reports",
        "communication_rendered_reports",
        "communication_inflight_handoffs",
        "communication_inflight_handoff_count",
        "communication_inflight_all_parse_ok",
    ):
        if hasattr(pred, key):
            record[key] = getattr(pred, key)
    return record


def evaluate(program, rows, plain_metric, num_threads: int = 1):
    rows = list(rows)
    if not rows:
        return 0.0, []

    max_workers = max(1, min(int(num_threads or 1), len(rows)))
    if max_workers == 1:
        records = [_evaluate_one(program, row, plain_metric) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            records = list(
                executor.map(
                    lambda row: _evaluate_one(program, row, plain_metric),
                    rows,
                )
            )
    total = sum(float(record.get("score") or 0.0) for record in records)
    return total / len(rows), records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=datasets(), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--train-size", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--n-rounds", type=int, default=1)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
    parser.add_argument("--max-full-evals", type=int, default=1)
    parser.add_argument("--reflection-minibatch-size", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--component-selector", default="all", choices=["all", "round_robin"])
    parser.add_argument("--skip-perfect-score", action="store_true", default=False)
    parser.add_argument("--baseline-only", action="store_true", default=False)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.topology not in topologies(args.dataset):
        raise ValueError(f"unknown topology {args.topology!r} for dataset {args.dataset!r}")

    out = args.out or WORKSPACE_ROOT / "results" / f"{args.topology}_{args.dataset}_real_gepa"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compiled").mkdir(exist_ok=True)
    (out / "compiled_raw").mkdir(exist_ok=True)
    gepa_log_dir = out / "gepa_state"
    gepa_log_dir.mkdir(exist_ok=True)
    update_status(
        out,
        "started",
        dataset=args.dataset,
        topology=args.topology,
        train_size=args.train_size,
        val_size=args.val_size,
        max_full_evals=args.max_full_evals,
        num_threads=args.num_threads,
    )

    try:
        cache_dir = WORKSPACE_ROOT / "cache" / f"{args.topology}_{args.dataset}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["DSPY_CACHEDIR"] = str(cache_dir)
        os.environ["DSP_CACHEDIR"] = str(cache_dir)

        update_status(out, "loading_dataset")
        dataset_mod = load_dataset_module(args.dataset)
        rows = dataset_mod.load_all()
        train, val = dataset_mod.train_val_split(rows, args.train_size, args.val_size, args.split_seed, args.offset)
        split_manifest = split_manifest_metadata(
            dataset_mod, args.train_size, args.val_size, args.split_seed, args.offset
        )
        exclude_eval_ids_active = exclude_real_eval_ids_enabled() or args.dataset in {"apibank", "toolhop"}
        excluded_real_eval_ids = sorted(real_eval_ids(args.dataset)) if exclude_eval_ids_active else []
        protected_eval_sample = swe_protected_sample_name() if args.dataset == "swe" else None
        metric = dataset_mod.metric
        plain_metric = lambda ex, pred: float(metric(ex, pred).score)

        dspy.configure(lm=build_task_pool(), track_usage=True)
        seed_program = build_program(args.dataset, args.topology, args.n_agents, args.n_rounds)
        seed_adapter = getattr(seed_program, "_adapter", None)
        adapter_roles = seed_adapter.roles() if seed_adapter is not None and hasattr(seed_adapter, "roles") else []
        update_status(
            out,
            "baseline_eval_started",
            train_ids=[row.id for row in train],
            val_ids=[row.id for row in val],
            train_level_counts=row_level_counts(train),
            val_level_counts=row_level_counts(val),
            train_repo_counts=row_repo_counts(train),
            val_repo_counts=row_repo_counts(val),
            train_profile_counts=row_profile_counts(dataset_mod, train),
            val_profile_counts=row_profile_counts(dataset_mod, val),
            split_manifest=split_manifest,
            protected_eval_sample=protected_eval_sample,
            adapter_roles=adapter_roles,
        )
        baseline_score, baseline_records = evaluate(seed_program, val, plain_metric, args.num_threads)
        write_jsonl(out / "baseline_val.jsonl", baseline_records)
        update_status(out, "baseline_eval_done", baseline_score=baseline_score)

        if args.baseline_only:
            meta = {
                "cell": f"{args.topology}/{args.dataset}",
                "mode": "real-runner-gepa-baseline-only",
                "dataset": args.dataset,
                "topology": args.topology,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "offset": args.offset,
                "split_seed": args.split_seed,
                "actual_train_ids": [row.id for row in train],
                "actual_val_ids": [row.id for row in val],
                "actual_train_level_counts": row_level_counts(train),
                "actual_val_level_counts": row_level_counts(val),
                "actual_train_repo_counts": row_repo_counts(train),
                "actual_val_repo_counts": row_repo_counts(val),
                "actual_train_profile_counts": row_profile_counts(dataset_mod, train),
                "actual_val_profile_counts": row_profile_counts(dataset_mod, val),
                "split_manifest": split_manifest,
                "protected_eval_sample": protected_eval_sample,
                "exclude_real_eval_ids": exclude_eval_ids_active,
                "excluded_real_eval_id_count": len(excluded_real_eval_ids),
                "excluded_real_eval_ids": excluded_real_eval_ids,
                "train_real_eval_overlap": sorted(set(row.id for row in train) & set(excluded_real_eval_ids)),
                "val_real_eval_overlap": sorted(set(row.id for row in val) & set(excluded_real_eval_ids)),
                "adapter_roles": adapter_roles,
                "compiled_prompt_files": [],
                "accept_compiled_prompt": False,
                "selected_prompt_source": "baseline",
                "selection_reason": "baseline-only run; no compiled prompt produced",
                "n_agents": args.n_agents,
                "n_rounds": args.n_rounds,
                "max_full_evals": args.max_full_evals,
                "reflection_minibatch_size": args.reflection_minibatch_size,
                "num_threads": args.num_threads,
                "component_selector": args.component_selector,
                "skip_perfect_score": args.skip_perfect_score,
                "task_model": TASK_MODEL,
                "reflection_model": REFL_MODEL,
                "task_endpoints": list(task_endpoints()),
                "reflection_endpoint": reflection_endpoint(),
                "baseline_score": baseline_score,
                "compiled_score": None,
                "delta": None,
                "baseline_failure_summary": failure_summary(baseline_records),
                "compiled_failure_summary": {},
                "baseline_records": baseline_records,
                "compiled_records": [],
            }
            write_json(out / "meta.json", meta)
            update_status(out, "complete", baseline_score=baseline_score, baseline_only=True)
            print(json.dumps(meta, indent=2, default=str), flush=True)
            return 0

        stopper = build_early_stopper(args.early_stop_patience)
        gepa_kwargs = {"stop_callbacks": [stopper]} if stopper is not None else {}
        budget_kwargs = {"auto": args.auto} if args.auto else {"max_full_evals": args.max_full_evals}
        optimizer = GEPA(
            metric=metric,
            reflection_lm=build_reflection_lm(),
            reflection_minibatch_size=args.reflection_minibatch_size,
            num_threads=args.num_threads,
            track_stats=True,
            component_selector=args.component_selector,
            skip_perfect_score=args.skip_perfect_score,
            log_dir=str(gepa_log_dir),
            gepa_kwargs=gepa_kwargs,
            **budget_kwargs,
        )
        update_status(out, "gepa_compile_started")
        compiled = optimizer.compile(seed_program, trainset=train, valset=val)
        compiled_prompt_sanitizer = sanitize_compiled_program_prompts(args.dataset, compiled)
        update_status(out, "gepa_compile_done", prompt_sanitizer=compiled_prompt_sanitizer)
        update_status(out, "compiled_eval_started", prompt_sanitizer=compiled_prompt_sanitizer)
        compiled_score, compiled_records = evaluate(compiled, val, plain_metric, args.num_threads)

        compiled_prompt_files = []
        compiled_prompt_texts = {}
        for name, predictor in compiled.named_predictors():
            safe = name.replace("/", "_").replace(".", "_")
            raw = (predictor.signature.instructions or "").strip() + "\n"
            clean = strip_reflection_preamble(raw)
            raw_path = out / "compiled_raw" / f"{safe}.txt"
            clean_path = out / "compiled" / f"{safe}.txt"
            raw_path.write_text(raw)
            clean_path.write_text(clean)
            compiled_prompt_files.append(str(clean_path.relative_to(out)))
            compiled_prompt_texts[safe] = clean

        compiled_prompt_quality = assess_compiled_prompt_quality(args.dataset, compiled_prompt_texts)
        write_jsonl(out / "compiled_val.jsonl", compiled_records)
        accept_compiled_prompt, selection_reason, acceptance_diagnostics = prompt_acceptance_decision(
            args.dataset,
            baseline_score,
            compiled_score,
            baseline_records,
            compiled_records,
            compiled_prompt_quality,
        )
        selected_records = compiled_records if accept_compiled_prompt else baseline_records
        write_jsonl(out / "optimized_val.jsonl", selected_records)
        teamsizes_meta = parse_teamsizes_topology(args.topology)
        communications_meta = parse_communications_topology(args.topology)
        meta = {
            "cell": f"{args.topology}/{args.dataset}",
            "mode": "real-runner-gepa",
            "dataset": args.dataset,
            "topology": args.topology,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "offset": args.offset,
            "split_seed": args.split_seed,
            "actual_train_ids": [row.id for row in train],
            "actual_val_ids": [row.id for row in val],
            "actual_train_level_counts": row_level_counts(train),
            "actual_val_level_counts": row_level_counts(val),
            "actual_train_repo_counts": row_repo_counts(train),
            "actual_val_repo_counts": row_repo_counts(val),
            "actual_train_profile_counts": row_profile_counts(dataset_mod, train),
            "actual_val_profile_counts": row_profile_counts(dataset_mod, val),
            "split_manifest": split_manifest,
            "protected_eval_sample": protected_eval_sample,
            "exclude_real_eval_ids": exclude_eval_ids_active,
            "excluded_real_eval_id_count": len(excluded_real_eval_ids),
            "excluded_real_eval_ids": excluded_real_eval_ids,
            "train_real_eval_overlap": sorted(set(row.id for row in train) & set(excluded_real_eval_ids)),
            "val_real_eval_overlap": sorted(set(row.id for row in val) & set(excluded_real_eval_ids)),
            "adapter_roles": adapter_roles,
            "compiled_prompt_files": compiled_prompt_files,
            "compiled_prompt_quality": compiled_prompt_quality,
            "compiled_prompt_sanitizer": compiled_prompt_sanitizer,
            "n_agents": args.n_agents,
            "n_rounds": args.n_rounds,
            "auto": args.auto,
            "max_full_evals": args.max_full_evals,
            "reflection_minibatch_size": args.reflection_minibatch_size,
            "num_threads": args.num_threads,
            "component_selector": args.component_selector,
            "skip_perfect_score": args.skip_perfect_score,
            "output_contracts": "enabled",
            "output_contract_version": OUTPUT_CONTRACT_VERSION,
            "gepa_log_dir": str(gepa_log_dir),
            "dspy_cache_dir": str(cache_dir),
            "dspy_version": dspy.__version__,
            "task_model": TASK_MODEL,
            "reflection_model": REFL_MODEL,
            "task_endpoints": list(task_endpoints()),
            "reflection_endpoint": reflection_endpoint(),
            "baseline_score": baseline_score,
            "compiled_score": compiled_score,
            "delta": compiled_score - baseline_score,
            "accept_compiled_prompt": accept_compiled_prompt,
            "selected_prompt_source": "compiled" if accept_compiled_prompt else "baseline",
            "selection_reason": selection_reason,
            "acceptance_diagnostics": acceptance_diagnostics,
            "baseline_failure_summary": failure_summary(baseline_records),
            "compiled_failure_summary": failure_summary(compiled_records),
            "baseline_records": baseline_records,
            "compiled_records": compiled_records,
        }
        if teamsizes_meta is not None:
            base_topology = teamsizes_meta["base_topology"]
            team_size = teamsizes_meta["team_size"]
            meta.update(
                {
                    "teamsizes_enabled": True,
                    "base_topology": base_topology,
                    "team_size": team_size,
                    "teamsizes_module": f"teamsizes.{base_topology}.{args.dataset}.{args.dataset}_r{team_size}",
                }
            )
        if communications_meta is not None:
            base_topology = communications_meta["base_topology"]
            communications_format = communications_meta["communications_format"]
            adapter = getattr(seed_program, "_adapter", None)
            meta.update(
                {
                    "communications_enabled": True,
                    "base_topology": base_topology,
                    "communications_format": communications_format,
                    "communications_module": getattr(
                        adapter,
                        "communications_module",
                        f"communications.{base_topology}.{args.dataset}.{args.dataset}_{communications_format}",
                    ),
                    "communications_base_module": getattr(adapter, "module_name", None),
                }
            )
        write_json(out / "meta.json", meta)
        update_status(out, "complete", baseline_score=baseline_score, compiled_score=compiled_score)
        print(json.dumps(meta, indent=2, default=str), flush=True)
        return 0
    except Exception as exc:
        update_status(
            out,
            "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
