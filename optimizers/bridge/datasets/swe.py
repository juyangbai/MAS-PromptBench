"""SWE-bench loader and structural patch scorer for real-runner optimization."""

from __future__ import annotations

import json
import re

import dspy

HF_DATASET = "princeton-nlp/SWE-bench_Verified"
HF_SPLIT = "test"
PATCH_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DIFF_HEADER_RE = re.compile(r"^(?:diff --git|---\s+a/|\+\+\+\s+b/|@@ )", re.MULTILINE)


def strip_thinking(text: str | None) -> str:
    text = text or ""
    idx = text.lower().rfind("</think>")
    if idx >= 0:
        text = text[idx + len("</think>") :]
    return text.strip()


def extract_patch(text: str | None) -> str | None:
    text = strip_thinking(text)
    for block in reversed(PATCH_BLOCK_RE.findall(text)):
        if DIFF_HEADER_RE.search(block):
            return block.strip()
    match = DIFF_HEADER_RE.search(text)
    return text[match.start() :].strip() if match else None


def _coerce_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def load_all() -> list[dspy.Example]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    rows: list[dspy.Example] = []
    for raw in ds:
        iid = raw.get("instance_id")
        problem = (raw.get("problem_statement") or "").strip()
        if not iid or not problem:
            continue
        repo = str(raw.get("repo") or "")
        instance = {
            "id": str(iid),
            "instance_id": str(iid),
            "repo": repo,
            "base_commit": str(raw.get("base_commit") or ""),
            "problem_statement": problem,
            "hints_text": raw.get("hints_text") or "",
            "test_patch": raw.get("test_patch") or "",
            "FAIL_TO_PASS": _coerce_list(raw.get("FAIL_TO_PASS")),
            "PASS_TO_PASS": _coerce_list(raw.get("PASS_TO_PASS")),
            "raw": dict(raw),
        }
        rows.append(
            dspy.Example(
                id=str(iid),
                repo=repo,
                task_instance=instance,
                issue_description=problem,
                answer=str(raw.get("patch") or ""),
            ).with_inputs("task_instance")
        )
    return rows


def score_patch(patch: str | None) -> dict:
    if not patch:
        return {"ok": False, "detail": "no unified diff extracted"}
    has_change = any(
        (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---"))
        for line in patch.splitlines()
    )
    ok = bool(DIFF_HEADER_RE.search(patch) and has_change)
    return {"ok": ok, "detail": "structural patch accepted" if ok else "patch is trivial or malformed"}


def exact_match_score(ok: bool) -> float:
    return 1.0 if ok else 0.0


def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    raw = getattr(prediction, "patch", None) or getattr(prediction, "answer", None) or str(prediction)
    result = score_patch(extract_patch(raw))
    score = exact_match_score(result["ok"])
    feedback = (
        f"Patch accepted: {result['detail']}."
        if score
        else "Format failure: emit one non-trivial fenced ```diff``` unified diff patch."
    )
    return dspy.Prediction(score=score, feedback=feedback)
