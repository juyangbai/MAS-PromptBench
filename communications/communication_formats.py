"""Shared communication-format helpers for msg baselines."""
from __future__ import annotations

import inspect
import json
import re
import time
from contextvars import ContextVar
from importlib import import_module
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]

FORMATS = {"freeform", "semi_structured", "structured_soft"}
REQUIRED_TAGS = ("STATUS", "SUMMARY", "EVIDENCE_OR_TESTS", "CONFIDENCE", "NEXT")
STATUSES = {"not_started", "in_progress", "completed", "blocked"}
CONFIDENCES = {"low", "medium", "high"}
STRICT_COMMUNICATION_FIELDS = (
    "communication_all_parse_ok",
    "communication_parse_rate",
    "communication_required_report_count",
    "communication_missing_roles",
    "communication_infra_error",
    "communication_inflight_handoff_count",
    "communication_inflight_all_parse_ok",
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_REPORT_RE = re.compile(r"(?im)^\s*JSON_REPORT\s*:\s*")
_INFLIGHT_HANDOFFS: ContextVar[list[dict] | None] = ContextVar(
    "communications_inflight_handoffs",
    default=None,
)


BASE_MODULES = {
    ("independent", "hotpotqa"): "topologies.independent.hotpotqa.langgraph_hotpotqa",
    ("independent", "lcb"): "topologies.independent.lcb.langgraph_lcb",
    ("independent", "toolhop"): "topologies.independent.toolhop.langgraph_toolhop",
    ("independent", "apibank"): "topologies.independent.apibank.langgraph_apibank",
    ("independent", "swe"): "topologies.independent.swe.langgraph_swe",
    ("sequential", "hotpotqa"): "topologies.sequential.langgraph.hotpotqa.langgraph_hotpotqa",
    ("sequential", "lcb"): "topologies.sequential.langgraph.lcb.langgraph_lcb",
    ("sequential", "toolhop"): "topologies.sequential.langgraph.toolhop.langgraph_toolhop",
    ("sequential", "apibank"): "topologies.sequential.langgraph.apibank.langgraph_apibank",
    ("sequential", "swe"): "topologies.sequential.langgraph.swe.langgraph_swe",
    ("centralized", "hotpotqa"): "topologies.centralized.langgraph.hotpotqa.langgraph_hotpotqa",
    ("centralized", "lcb"): "topologies.centralized.langgraph.lcb.langgraph_lcb",
    ("centralized", "toolhop"): "topologies.centralized.langgraph.toolhop.langgraph_toolhop",
    ("centralized", "apibank"): "topologies.centralized.langgraph.apibank.langgraph_apibank",
    ("centralized", "swe"): "topologies.centralized.langgraph.swe.langgraph_swe",
    ("decentralized", "hotpotqa"): "topologies.decentralized.langgraph.hotpotqa.langgraph_hotpotqa",
    ("decentralized", "lcb"): "topologies.decentralized.langgraph.lcb.langgraph_lcb",
    ("decentralized", "toolhop"): "topologies.decentralized.langgraph.toolhop.langgraph_toolhop",
    ("decentralized", "apibank"): "topologies.decentralized.langgraph.apibank.langgraph_apibank",
    ("decentralized", "swe"): "topologies.decentralized.langgraph.swe.langgraph_swe",
}


def communication_contract(fmt: str, dataset: str) -> str:
    """Return prompt text for the requested communication format."""
    if fmt == "freeform":
        return ""
    if fmt == "semi_structured":
        dataset_part = _semi_dataset_guidance(dataset)
        return (
            "\n\nINTER-AGENT COMMUNICATION FORMAT:\n"
            "When you report your intermediate work to another agent, peer, "
            "manager, next pipeline stage, or aggregator, use the following "
            "tagged report before any scorer-facing final artifact:\n\n"
            "[STATUS]\n"
            "One of: not_started, in_progress, completed, blocked.\n\n"
            "[SUMMARY]\n"
            "A concise statement of your current belief, result, or decision.\n\n"
            "[EVIDENCE_OR_TESTS]\n"
            "The evidence, retrieved facts, checks, or tests supporting the summary.\n\n"
            "[CONFIDENCE]\n"
            "One of: low, medium, high, followed by a short reason.\n\n"
            "[NEXT]\n"
            "What the next stage, manager, peer, or aggregator should use from this message.\n"
            f"{dataset_part}\n"
            "If you must provide the final benchmark answer/code, put it AFTER "
            "this communication report and still satisfy the protected final output contract."
        )
    if fmt == "structured_soft":
        payload = _json_payload_guidance(dataset)
        return (
            "\n\nINTER-AGENT COMMUNICATION FORMAT:\n"
            "When you report your intermediate work to another agent, peer, "
            "manager, next pipeline stage, or aggregator, first emit one "
            "JSON report using this soft schema. Prefix the report with "
            "JSON_REPORT: and do not wrap the report in triple-backtick fences:\n\n"
            "JSON_REPORT:\n"
            "{\n"
            '  "status": "completed",\n'
            '  "summary": "...",\n'
            '  "confidence": "medium",\n'
            '  "next": "...",\n'
            '  "payload": {}\n'
            "}\n"
            "END_JSON_REPORT\n\n"
            "status must be one of not_started, in_progress, completed, blocked. "
            "confidence must be one of low, medium, high.\n"
            f"{payload}\n"
            "If you must provide the final benchmark answer/code, put it AFTER "
            "END_JSON_REPORT and still satisfy the protected final output contract. "
            "Do not put the final answer/code only inside the JSON report."
        )
    raise ValueError(f"unknown communication format {fmt!r}")


def _semi_dataset_guidance(dataset: str) -> str:
    if dataset == "hotpotqa":
        return (
            "\nOptional HotpotQA tags may be included when useful:\n"
            "[ENTITIES], [HOPS], [ANSWER_CANDIDATE]."
        )
    if dataset == "lcb":
        return (
            "\nOptional LCB tags may be included when useful:\n"
            "[APPROACH], [COMPLEXITY], [EDGE_CASES], [CODE_STATUS]."
        )
    if dataset == "toolhop":
        return (
            "\nOptional ToolHop tags may be included when useful:\n"
            "[TOOL_CHAIN], [OBSERVATIONS], [ANSWER_CANDIDATE]."
        )
    if dataset == "apibank":
        return (
            "\nFor API-Bank, do not add optional tags. Keep each required "
            "section to one short sentence. Put the exact final API call "
            "after [NEXT]."
        )
    if dataset == "swe":
        return (
            "\nOptional SWE-bench tags may be included when useful:\n"
            "[BUG_LOCATION], [PATCH_PLAN], [RISK_OR_REGRESSION]."
        )
    return ""


def _json_payload_guidance(dataset: str) -> str:
    if dataset == "hotpotqa":
        return (
            "For HotpotQA, payload should use fields when known: entities, "
            "hops, evidence, answer_candidate. Evidence entries should include "
            "a source/page title and fact."
        )
    if dataset == "lcb":
        return (
            "For LCB, payload should use fields when known: approach, "
            "complexity, edge_cases, tests, code_status. Test entries should "
            "include input, expected, observed, and passed when available."
        )
    if dataset == "toolhop":
        return (
            "For ToolHop, payload should use fields when known: tool_chain, "
            "observations, answer_candidate. Each tool_chain entry should "
            "name the tool and summarize the observation used for the next hop."
        )
    if dataset == "apibank":
        return (
            "For API-Bank, keep payload compact with fields when known: "
            "api_choice and call_candidate. The final API call must still "
            "appear after END_JSON_REPORT."
        )
    if dataset == "swe":
        return (
            "For SWE-bench, payload should use fields when known: bug_location, "
            "root_cause, patch_plan, regression_risk, tests_or_checks. The "
            "model patch is still computed from the repository diff."
        )
    return ""


def append_contract(prompt: str, fmt: str, dataset: str) -> str:
    contract = communication_contract(fmt, dataset)
    if not contract or "INTER-AGENT COMMUNICATION FORMAT:" in (prompt or ""):
        return prompt
    return (prompt or "").rstrip() + contract


def _restore_base_module(base: Any) -> None:
    original = getattr(base, "_communications_original_load_prompt", None)
    if original is not None and hasattr(base, "_load_prompt"):
        base._load_prompt = original
    original_prompt = getattr(base, "_communications_original_system_prompt", None)
    if original_prompt is not None and hasattr(base, "SYSTEM_PROMPT"):
        base.SYSTEM_PROMPT = original_prompt


def configure_base_module(base: Any, *, fmt: str, dataset: str) -> None:
    """Permanently patch prompt-loading hooks in an imported topology module."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown communication format {fmt!r}")
    if fmt == "freeform":
        _restore_base_module(base)
        base.COMMUNICATION_FORMAT = fmt
        return

    if hasattr(base, "_load_prompt"):
        original = getattr(base, "_communications_original_load_prompt", None)
        if original is None:
            original = base._load_prompt
            base._communications_original_load_prompt = original

        def _load_prompt_with_msg(role: str, _original=original):
            return append_contract(_original(role), fmt, dataset)

        base._load_prompt = _load_prompt_with_msg

    if hasattr(base, "SYSTEM_PROMPT"):
        original_prompt = getattr(base, "_communications_original_system_prompt", None)
        if original_prompt is None:
            original_prompt = base.SYSTEM_PROMPT
            base._communications_original_system_prompt = original_prompt
        base.SYSTEM_PROMPT = append_contract(str(original_prompt), fmt, dataset)

    base.COMMUNICATION_FORMAT = fmt


def parse_message(text: str, fmt: str) -> dict:
    """Parse one model message under a communication format."""
    raw = text or ""
    if fmt == "freeform":
        return {"ok": True, "parsed": {}, "errors": []}
    if fmt == "semi_structured":
        return _parse_semi(raw)
    if fmt == "structured_soft":
        return _parse_structured(raw)
    raise ValueError(f"unknown communication format {fmt!r}")


def _parse_semi(text: str) -> dict:
    tags: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^\[([A-Z_]+)\]\s*$", text or ""))
    for idx, match in enumerate(matches):
        tag = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        tags[tag.lower()] = text[start:end].strip()

    errors = []
    for tag in REQUIRED_TAGS:
        if not tags.get(tag.lower()):
            errors.append(f"missing [{tag}]")
    status = _first_token(tags.get("status") or "")
    if status and status not in STATUSES:
        errors.append(f"invalid status {status!r}")
    confidence = _first_token(tags.get("confidence") or "")
    if confidence and confidence not in CONFIDENCES:
        errors.append(f"invalid confidence {confidence!r}")
    return {"ok": not errors, "parsed": tags, "errors": errors}


def _first_token(text: str) -> str:
    parts = (text or "").strip().split(None, 1)
    return parts[0].lower().strip(".,;:") if parts else ""


def _escape_embedded_semi_tags(text: str) -> str:
    """Prevent nested semi-structured tags from splitting outer sections."""
    return re.sub(r"(?m)^(\s*)\[([A-Z_]+)\]\s*$", r"\1> [\2]", text or "")


def _parse_structured(text: str) -> dict:
    candidate = _extract_json_candidate(text or "")
    if not candidate:
        return {"ok": False, "parsed": {}, "errors": ["missing JSON report"]}
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"ok": False, "parsed": {}, "errors": [f"malformed JSON: {exc.msg}"]}
    if not isinstance(parsed, dict):
        return {"ok": False, "parsed": parsed, "errors": ["JSON report is not an object"]}

    errors = []
    for key in ("status", "summary", "confidence", "next", "payload"):
        if key not in parsed:
            errors.append(f"missing {key!r}")
    status = str(parsed.get("status", "")).strip().lower().strip(".,;:")
    if status and status not in STATUSES:
        errors.append(f"invalid status {status!r}")
    confidence = str(parsed.get("confidence", "")).strip().lower().strip(".,;:")
    if confidence and confidence not in CONFIDENCES:
        errors.append(f"invalid confidence {confidence!r}")
    if "payload" in parsed and not isinstance(parsed.get("payload"), dict):
        errors.append("payload is not an object")
    return {"ok": not errors, "parsed": parsed, "errors": errors}


def _extract_json_candidate(text: str) -> str | None:
    marker = _JSON_REPORT_RE.search(text or "")
    if marker:
        start = text.find("{", marker.end())
        if start >= 0:
            return _decode_first_json_object(text[start:])

    match = _FENCED_JSON_RE.search(text or "")
    if match:
        candidate = match.group(1).strip()
        if candidate.lstrip().startswith("{"):
            return candidate or None

    stripped = (text or "").strip()
    if stripped.startswith("{"):
        return _decode_first_json_object(stripped)
    return None


def _decode_first_json_object(text: str) -> str | None:
    try:
        _, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return text.strip() or None
    return text[:end].strip() or None


def normalize_report(
    role: str,
    text: Any,
    *,
    status: str = "completed",
    confidence: str = "medium",
    next_action: str | None = None,
    payload: dict | None = None,
    dataset: str = "",
    topology: str = "",
) -> dict:
    """Normalize arbitrary agent text into the communications report shape."""
    role_name = str(role or "report")
    raw_text = str(text or "").strip()
    payload_obj = _json_safe_payload(payload)
    payload_obj.setdefault("role", role_name)
    if dataset:
        payload_obj.setdefault("dataset", dataset)
    if topology:
        payload_obj.setdefault("topology", topology)
    if raw_text:
        payload_obj.setdefault("raw_excerpt", _clip_text(raw_text, 800))
    return {
        "role": role_name,
        "status": _coerce_status(status),
        "summary": _derive_summary(raw_text),
        "evidence_or_tests": _derive_evidence(raw_text),
        "confidence": _coerce_confidence(confidence),
        "next": next_action or "Use this report as context for the next agent handoff.",
        "payload": payload_obj,
        "raw_text": raw_text,
    }


def render_report(report: dict, fmt: str) -> str:
    """Render one normalized report deterministically in the requested format."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown communication format {fmt!r}")
    role = str(report.get("role") or "report")
    raw_text = str(report.get("raw_text") or "").strip()
    summary = str(report.get("summary") or "No substantive report was produced.").strip()
    evidence = str(report.get("evidence_or_tests") or "No evidence reported.").strip()
    confidence = _coerce_confidence(str(report.get("confidence") or "medium"))
    status = _coerce_status(str(report.get("status") or "completed"))
    next_action = str(report.get("next") or "Use this report as context for the next agent handoff.").strip()
    payload = _json_safe_payload(report.get("payload") if isinstance(report.get("payload"), dict) else {})
    payload.setdefault("role", role)

    if fmt == "freeform":
        body = raw_text or summary
        return f"{role}:\n{body}" if body else f"{role}:"
    if fmt == "semi_structured":
        summary = _escape_embedded_semi_tags(summary)
        evidence = _escape_embedded_semi_tags(evidence)
        next_action = _escape_embedded_semi_tags(next_action)
        return (
            f"[STATUS]\n{status}\n\n"
            f"[SUMMARY]\n{summary}\n\n"
            f"[EVIDENCE_OR_TESTS]\n{evidence}\n\n"
            f"[CONFIDENCE]\n{confidence}\n\n"
            f"[NEXT]\n{next_action}"
        )
    if fmt == "structured_soft":
        rendered = {
            "status": status,
            "summary": summary,
            "confidence": confidence,
            "next": next_action,
            "payload": payload,
        }
        return "JSON_REPORT:\n" + json.dumps(rendered, ensure_ascii=False, sort_keys=True) + "\nEND_JSON_REPORT"
    raise ValueError(f"unknown communication format {fmt!r}")


def begin_handoff_recording():
    """Start collecting deterministic in-flight handoff evidence."""
    return _INFLIGHT_HANDOFFS.set([])


def end_handoff_recording(token) -> list[dict]:
    """Stop collecting handoff evidence and return records captured so far."""
    records = list(_INFLIGHT_HANDOFFS.get() or [])
    _INFLIGHT_HANDOFFS.reset(token)
    return records


def format_handoff(
    role: str,
    text: Any,
    *,
    fmt: str | None,
    dataset: str,
    topology: str,
    status: str = "completed",
    confidence: str = "medium",
    next_action: str | None = None,
    payload: dict | None = None,
) -> str:
    """Convert one inter-agent handoff into the requested communications format.

    This is the in-flight counterpart to ``collect_reports()``. It is called
    before another agent receives a prior agent's output, so semi-structured
    and structured-soft experiments actually constrain the receiver context
    instead of only normalizing artifacts after the run.
    """
    raw_text = "" if text is None else str(text)
    if not fmt or fmt == "freeform":
        rendered = raw_text
        parsed = {"ok": True, "errors": []}
    else:
        normalized = normalize_report(
            role,
            raw_text,
            status=status,
            confidence=confidence,
            next_action=next_action,
            payload=payload,
            dataset=dataset,
            topology=topology,
        )
        rendered = render_report(normalized, fmt)
        parsed = parse_message(rendered, fmt)

    records = _INFLIGHT_HANDOFFS.get()
    if records is not None:
        records.append(
            {
                "role": str(role),
                "dataset": dataset,
                "topology": topology,
                "format": fmt or "freeform",
                "ok": bool(parsed.get("ok")),
                "errors": list(parsed.get("errors") or []),
                "raw_excerpt": _clip_text(raw_text, 1000),
                "rendered_excerpt": _clip_text(rendered, 2000),
            }
        )
    return rendered


def collect_reports(out: dict, *, topology: str, fmt: str, dataset: str = "") -> dict:
    """Render raw runner reports through communications infra and compute strict metrics.

    ``communication_parse_ok`` intentionally remains the legacy loose boolean:
    true if any rendered report parsed. New callers should use
    ``communication_all_parse_ok`` or ``communication_parse_rate``.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown communication format {fmt!r}")

    reports = []
    infra_errors = []
    for role, text in _report_texts(out or {}, topology):
        try:
            normalized = normalize_report(role, text, dataset=dataset, topology=topology)
            rendered = render_report(normalized, fmt)
            parsed = parse_message(rendered, fmt)
        except Exception as exc:  # pragma: no cover - defensive infra guard
            normalized = normalize_report(role, text, dataset=dataset, topology=topology)
            rendered = ""
            parsed = {"ok": False, "parsed": {}, "errors": [f"infra render/parse error: {type(exc).__name__}: {exc}"]}
            infra_errors.append(f"{role}: {type(exc).__name__}: {exc}")
        reports.append(
            {
                "role": str(role),
                "ok": bool(parsed["ok"]),
                "errors": list(parsed["errors"]),
                "parsed": _clip_value(parsed["parsed"]),
                "raw_excerpt": _clip_text(text, 2000),
                "rendered_excerpt": _clip_text(rendered, 2000),
            }
        )

    if not reports and fmt != "freeform":
        infra_errors.append("no communication reports found")

    missing_roles = _missing_report_roles(topology, reports) if fmt != "freeform" else []
    ok_count = sum(1 for report in reports if report["ok"])
    total = len(reports)
    parse_rate = 1.0 if fmt == "freeform" and total == 0 else (ok_count / total if total else 0.0)
    parse_errors = [f"{r['role']}: {err}" for r in reports for err in r["errors"]]
    parse_errors.extend(infra_errors)
    role_warnings = [f"missing role report: {role}" for role in missing_roles]
    warnings = parse_errors + role_warnings
    all_parse_ok = not parse_errors and (fmt == "freeform" or (total > 0 and ok_count == total))
    legacy_parse_ok = True if fmt == "freeform" else ok_count > 0
    infra_error = "; ".join(infra_errors) if infra_errors else None

    return {
        "communication_format": fmt,
        "communication_parse_ok": legacy_parse_ok,
        "communication_all_parse_ok": all_parse_ok,
        "communication_parse_rate": parse_rate,
        "communication_required_report_count": total,
        "communication_missing_roles": missing_roles,
        "communication_infra_error": infra_error,
        "communication_parse_errors": [] if legacy_parse_ok and not infra_error else parse_errors,
        "communication_parse_warnings": warnings,
        "communication_report_ok_count": ok_count,
        "communication_report_total": total,
        "communication_reports": reports,
        "communication_rendered_reports": [
            {"role": report["role"], "rendered": report["rendered_excerpt"]}
            for report in reports
        ],
    }


def _clip_text(value: Any, limit: int = 1000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _derive_summary(text: str) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return "No substantive report was produced."
    return _clip_text(clean, 500)


def _derive_evidence(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return "No evidence reported."
    return _clip_text(clean, 1000)


def _coerce_status(status: str) -> str:
    value = _first_token(status or "completed")
    return value if value in STATUSES else "completed"


def _coerce_confidence(confidence: str) -> str:
    value = _first_token(confidence or "medium")
    return value if value in CONFIDENCES else "medium"


def _json_safe_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {}
    try:
        safe = json.loads(json.dumps(payload, default=str))
    except Exception:
        return {str(key): str(value) for key, value in payload.items()}
    return safe if isinstance(safe, dict) else {}


def _missing_report_roles(topology: str, reports: list[dict]) -> list[str]:
    roles = [str(report.get("role") or "").lower() for report in reports]
    if not roles:
        return ["<any_report>"]
    missing = []
    topology_key = (topology or "").replace("_openai", "")
    if topology_key == "centralized":
        if not any("manager" in role for role in roles):
            missing.append("manager")
        if not any("worker" in role or (role and "manager" not in role) for role in roles):
            missing.append("worker")
    return missing


def _report_texts(out: dict, topology: str) -> list[tuple[str, str]]:
    if isinstance(out.get("by_stage"), dict):
        return [(str(role), str(text or "")) for role, text in out["by_stage"].items() if text]
    if isinstance(out.get("messages"), list):
        pairs = []
        for idx, msg in enumerate(out["messages"]):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            source = str(msg.get("source") or (role if role else f"message_{idx}"))
            if role and role != "assistant":
                continue
            if source in {"system", "user", "tool"}:
                continue
            content = str(msg.get("content") or "")
            if content.strip():
                pairs.append((source, content))
        if pairs:
            return pairs
    if isinstance(out.get("per_agent"), list):
        pairs = []
        for idx, item in enumerate(out["per_agent"]):
            text = item.get("raw") or item.get("final_content") or item.get("predicted_answer") or item.get("raw_tail")
            if text:
                pairs.append((f"agent_{item.get('agent_id', idx)}", str(text)))
        if pairs:
            return pairs
    if isinstance(out.get("per_peer"), list):
        pairs = []
        for idx, item in enumerate(out["per_peer"]):
            text = item.get("raw") or item.get("final_content") or item.get("predicted_answer") or item.get("raw_tail")
            if text:
                pairs.append((f"peer_{item.get('peer', idx)}", str(text)))
        if pairs:
            return pairs
    if isinstance(out.get("workers"), list):
        pairs = []
        for idx, item in enumerate(out["workers"]):
            text = item.get("raw") or item.get("final_content") or item.get("predicted_answer") or item.get("raw_tail")
            if text:
                pairs.append((str(item.get("role") or f"worker_{idx}"), str(text)))
        manager = out.get("manager")
        if isinstance(manager, dict):
            text = manager.get("raw") or manager.get("final_content") or manager.get("predicted_answer") or manager.get("raw_tail")
            if text:
                pairs.append((str(manager.get("role") or "manager"), str(text)))
        if pairs:
            return pairs
    contexts = out.get("all_contexts") or []
    pairs = []
    for idx, ctx in enumerate(contexts):
        text = _last_ai_text(ctx)
        if text:
            label = "agent" if topology == "independent" else "peer"
            pairs.append((f"{label}_{idx}", text))
    if pairs:
        return pairs
    raw = out.get("raw")
    if raw:
        return [("final", str(raw))]
    return []


def _clip_value(value: Any, limit: int = 2000) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "\n...<truncated>..."
    if isinstance(value, dict):
        return {str(k): _clip_value(v, limit) for k, v in list(value.items())[:50]}
    if isinstance(value, list):
        return [_clip_value(item, limit) for item in value[:50]]
    return value


def _last_ai_text(messages: list) -> str:
    for msg in reversed(messages or []):
        if getattr(msg, "type", None) == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip() and not getattr(msg, "tool_calls", None):
                return content
    for msg in reversed(messages or []):
        if getattr(msg, "type", None) == "ai":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def install_proxy(module_globals: dict, *, topology: str, dataset: str, fmt: str) -> None:
    """Populate one tiny msg module with wrapped topology functions."""
    base = import_module(BASE_MODULES[(topology, dataset)])
    configure_base_module(base, fmt=fmt, dataset=dataset)
    original_solve = getattr(base, "solve", None)

    def _ensure_configured() -> None:
        # The base topology module is shared in sys.modules. Re-assert this
        # proxy's format before execution so importing another msg variant in
        # the same process cannot silently leak its prompt contract here.
        configure_base_module(base, fmt=fmt, dataset=dataset)

    def solve(*args, **kwargs):
        _ensure_configured()
        token = begin_handoff_recording()
        handoffs: list[dict] = []
        try:
            if dataset == "toolhop":
                out = _solve_toolhop(base, *args, topology=topology, fmt=fmt, **kwargs)
            elif dataset == "apibank":
                out = _solve_apibank(base, *args, topology=topology, fmt=fmt, **kwargs)
            else:
                out = original_solve(*args, **kwargs)
        finally:
            handoffs = end_handoff_recording(token)
        if isinstance(out, dict):
            out = dict(out)
            out["communication_inflight_handoffs"] = handoffs
            out["communication_inflight_handoff_count"] = len(handoffs)
            out["communication_inflight_all_parse_ok"] = all(
                bool(item.get("ok")) for item in handoffs
            )
            out.update(collect_reports(out, topology=topology, fmt=fmt, dataset=dataset))
        return out

    def run_one(instance: dict, *args, out_dir: Path | None = None, **kwargs) -> dict:
        _ensure_configured()
        if dataset == "swe":
            rec = base.run_one(instance, *args, **kwargs)
            if isinstance(rec, dict):
                rec = dict(rec)
                rec["communication_format"] = fmt
                rec["base_module"] = base.__name__
            return rec
        if dataset == "hotpotqa":
            return _run_one_hotpotqa(base, solve, instance, topology=topology, fmt=fmt)
        if dataset == "lcb":
            return _run_one_lcb(base, solve, instance, topology=topology, fmt=fmt)
        if dataset == "toolhop":
            return _run_one_toolhop(base, solve, instance, topology=topology, fmt=fmt)
        if dataset == "apibank":
            return _run_one_apibank(base, solve, instance, topology=topology, fmt=fmt)
        raise ValueError(f"unsupported dataset {dataset!r}")

    def run_batch(instances: list[dict], out_path: Path | None = None, verbose: bool = True, **kwargs) -> dict:
        _ensure_configured()
        return _run_batch(base, run_one, instances, out_path=out_path, verbose=verbose, dataset=dataset)

    public_names = {
        "COMMUNICATION_FORMAT": fmt,
        "TOPOLOGY": topology,
        "DATASET": dataset,
        "BASE_MODULE": base.__name__,
        "VLLM_BASE_URL": getattr(base, "VLLM_BASE_URL", "?"),
        "MODEL_ID": getattr(base, "MODEL_ID", "?"),
        "load_instances": base.load_instances,
        "solve": solve,
        "run_one": run_one,
        "run_batch": run_batch,
    }
    for name in (
        "extract_answer",
        "exact_match_score",
        "f1_score",
        "extract_code",
        "run_tests",
        "format_prompt",
        "N_AGENTS",
        "N_ROUNDS",
        "dataset_summary",
    ):
        if hasattr(base, name):
            public_names[name] = getattr(base, name)
    module_globals.update(public_names)


def cli_main(g: dict) -> int:
    """Run one communications pair as a standalone batch eval (used by each pair's ``__main__``).

    The pair's namespace already holds ``load_instances`` and ``run_batch`` (wired
    by :func:`install_proxy`), so a pair runs without any external launcher:
        python communications/<topology>/<dataset>/<dataset>_<format>.py --batch --limit 100
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=f"communications pair {g.get('TOPOLOGY')}/{g.get('DATASET')} [{g.get('COMMUNICATION_FORMAT')}]"
    )
    parser.add_argument("--batch", action="store_true", help="run the real benchmark slice (default)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--only", action="append", default=None)
    parser.add_argument("--out", default=None, help="output JSONL path")
    args = parser.parse_args()

    instances = g["load_instances"](limit=args.limit, offset=args.offset, only=args.only)
    if args.out:
        out_path = Path(args.out)
    else:
        tag = f"{g['TOPOLOGY']}_{g['DATASET']}_{g['COMMUNICATION_FORMAT']}"
        out_path = _REPO_ROOT / "results" / "communications_baseline" / tag / "results.jsonl"
    summary = g["run_batch"](instances, out_path=out_path)
    print(json.dumps({k: summary.get(k) for k in ("n", "em", "total_s")}, default=str))
    return 0


def _solve_toolhop(base: Any, sample: dict, *, topology: str, fmt: str, **kwargs) -> dict:
    return base.solve_topology(
        sample,
        style=f"{getattr(base, 'STYLE', topology)}_communications_{fmt}",
        topology=getattr(base, "TOPOLOGY", topology),
        role=getattr(base, "ROLE", "solver"),
        prompt_suffix="",
        **kwargs,
    )


def _solve_apibank(base: Any, sample: dict, *, topology: str, fmt: str, **kwargs) -> dict:
    return base.solve_topology(
        sample,
        style=f"{getattr(base, 'STYLE', topology)}_communications_{fmt}",
        topology=getattr(base, "TOPOLOGY", topology),
        role=getattr(base, "ROLE", "solver"),
        prompt_suffix="",
        **kwargs,
    )


def _run_one_hotpotqa(base: Any, solve_fn: Any, inst: dict, *, topology: str, fmt: str) -> dict:
    t0 = time.time()
    out = solve_fn(inst["question"])
    latency_s = time.time() - t0
    pred = out.get("answer")
    gold = inst["answer"]
    if pred is not None:
        em = base.exact_match_score(pred, gold)
        f1, prec, rec = base.f1_score(pred, gold)
    else:
        em = f1 = prec = rec = 0.0
    rec = {
        "id": inst["id"],
        "question": inst["question"],
        "gold_answer": gold,
        "predicted_answer": pred,
        "em": em,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "type": inst.get("type"),
        "level": inst.get("level"),
        "latency_s": round(latency_s, 2),
        **_compact_output_fields(out),
        **_telemetry(base, out),
        "error": None,
    }
    return rec


def _run_one_lcb(base: Any, solve_fn: Any, inst: dict, *, topology: str, fmt: str) -> dict:
    t0 = time.time()
    kwargs = {"starter_code": inst.get("starter_code") or None}
    sig = inspect.signature(base.solve)
    if "tests" in sig.parameters:
        kwargs["tests"] = inst.get("tests")
    out = solve_fn(inst["problem"], **kwargs)
    latency_s = time.time() - t0
    code = out.get("code")
    if code:
        scored = base.run_tests(code, inst["tests"], timeout_s=6)
    else:
        scored = {"pass": 0, "total": len(inst["tests"]), "pass_rate": 0.0, "details": []}
    em = base.exact_match_score(scored["pass_rate"]) if code else 0.0
    return {
        "id": inst["id"],
        "problem": inst["problem"][:400],
        "starter_code": inst.get("starter_code") or "",
        "predicted_code": code,
        "winner": out.get("winner"),
        "pass": scored["pass"],
        "total": scored["total"],
        "pass_rate": scored["pass_rate"],
        "em": em,
        "difficulty": inst.get("difficulty"),
        "platform": inst.get("platform"),
        "latency_s": round(latency_s, 2),
        **_compact_output_fields(out),
        **_telemetry(base, out),
        "error": None,
    }


def _run_one_toolhop(base: Any, solve_fn: Any, inst: dict, *, topology: str, fmt: str) -> dict:
    t0 = time.time()
    out = solve_fn(inst)
    latency_s = time.time() - t0
    messages = out.get("messages") or []
    final_content = base._last_assistant_content(messages)
    pred = out.get("predicted_answer")
    if pred is None:
        pred = base.extract_answer(final_content)
    if "correct" in out:
        correct = bool(out.get("correct"))
    else:
        correct = base.score_answer(
            str(inst.get("answer", "")),
            final_content,
            base._previous_tool_content(messages),
        )
    return {
        "id": inst["id"],
        "idx": inst["id"],
        "question": inst.get("question"),
        "gold_answer": inst.get("answer"),
        "predicted_answer": pred,
        "answer_correct": int(correct),
        "correct": bool(correct),
        "em": float(bool(correct)),
        "latency_s": round(latency_s, 2),
        "n_agents": out.get("n_agents"),
        "n_rounds": out.get("n_rounds"),
        "winner": out.get("winner"),
        "buckets": out.get("buckets"),
        "tool_calls": int(out.get("tool_calls") if out.get("tool_calls") is not None else sum(len(message.get("tool_calls") or []) for message in messages)),
        "turns": int(out.get("turns") if out.get("turns") is not None else sum(1 for message in messages if message.get("role") == "assistant")),
        **_compact_output_fields(out),
        **_telemetry(base, out),
        "error": None,
    }


def _run_one_apibank(base: Any, solve_fn: Any, inst: dict, *, topology: str, fmt: str) -> dict:
    t0 = time.time()
    out = solve_fn(inst)
    latency_s = time.time() - t0
    raw = out.get("raw") or ""
    pred = out.get("predicted_answer") or base.extract_api_call(raw)
    scored = (
        {
            "correct": bool(out.get("correct")),
            "predicted_api_name": out.get("predicted_api_name"),
            "predicted_params": out.get("predicted_params"),
            "stage": out.get("stage"),
            "error": out.get("error"),
        }
        if "correct" in out
        else base.score_prediction(inst, pred)
    )
    return {
        "id": inst["id"],
        "idx": inst["id"],
        "file": inst.get("file"),
        "sample_id": inst.get("sample_id"),
        "question": base._format_chat_history(inst.get("chat_history") or []),
        "gold_api_call": inst.get("gold_api_call"),
        "gold_api_name": inst.get("ground_truth", {}).get("api_name"),
        "predicted_answer": pred,
        "predicted_api_name": scored.get("predicted_api_name"),
        "predicted_params": scored.get("predicted_params"),
        "answer_correct": int(bool(scored.get("correct"))),
        "correct": bool(scored.get("correct")),
        "em": float(bool(scored.get("correct"))),
        "stage": scored.get("stage"),
        "latency_s": round(latency_s, 2),
        "n_agents": out.get("n_agents"),
        "n_rounds": out.get("n_rounds"),
        "winner": out.get("winner"),
        "buckets": out.get("buckets"),
        "tool_calls": int(out.get("tool_calls") or 0),
        "turns": int(out.get("turns") or 1),
        **_compact_output_fields(out),
        **_telemetry(base, out),
        "error": scored.get("error"),
    }


def _run_batch(base: Any, run_one_fn: Any, instances: list[dict], *, out_path: Path | None, verbose: bool, dataset: str) -> dict:
    per_instance = []
    total = 0.0
    start = time.time()
    out_f = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w")
    try:
        for idx, inst in enumerate(instances):
            try:
                rec = run_one_fn(inst, None)
            except Exception as exc:
                rec = _error_record(inst, dataset, exc)
            per_instance.append(rec)
            total += float(rec.get("em") or 0.0)
            if out_f is not None:
                out_f.write(json.dumps(rec, default=str) + "\n")
                out_f.flush()
            if verbose:
                print(f"[{idx + 1:>3}/{len(instances)}] {rec.get('id')} em={rec.get('em', 0):.0f} lat={rec.get('latency_s', 0)}s", flush=True)
    finally:
        if out_f is not None:
            out_f.close()
    n = len(instances)
    return {
        "n": n,
        "em": total / n if n else 0.0,
        "em_sum": total,
        "total_s": round(time.time() - start, 1),
        "per_instance": per_instance,
    }


def _compact_output_fields(out: dict) -> dict:
    data = {
        "communication_format": out.get("communication_format"),
        "communication_parse_ok": out.get("communication_parse_ok"),
        "communication_all_parse_ok": out.get("communication_all_parse_ok"),
        "communication_parse_rate": out.get("communication_parse_rate"),
        "communication_required_report_count": out.get("communication_required_report_count", 0),
        "communication_missing_roles": out.get("communication_missing_roles") or [],
        "communication_infra_error": out.get("communication_infra_error"),
        "communication_parse_errors": out.get("communication_parse_errors") or [],
        "communication_parse_warnings": out.get("communication_parse_warnings") or [],
        "communication_report_ok_count": out.get("communication_report_ok_count", 0),
        "communication_report_total": out.get("communication_report_total", 0),
        "communication_reports": out.get("communication_reports") or [],
        "communication_rendered_reports": out.get("communication_rendered_reports") or [],
        "communication_inflight_handoffs": out.get("communication_inflight_handoffs") or [],
        "communication_inflight_handoff_count": out.get("communication_inflight_handoff_count", 0),
        "communication_inflight_all_parse_ok": out.get("communication_inflight_all_parse_ok"),
    }
    if isinstance(out.get("by_stage"), dict):
        data["by_stage"] = {k: (v or "")[:800] for k, v in out["by_stage"].items()}
    if isinstance(out.get("messages"), list):
        data["n_messages"] = len(out["messages"])
    if isinstance(out.get("per_agent"), list):
        data["per_agent"] = _compact_members(out["per_agent"], "agent_id")
    if isinstance(out.get("per_peer"), list):
        data["per_peer"] = _compact_members(out["per_peer"], "peer")
    if isinstance(out.get("stage_outputs"), list):
        data["stage_outputs"] = _compact_members(out["stage_outputs"], "stage")
    if isinstance(out.get("workers"), list):
        data["workers"] = _compact_members(out["workers"], "worker")
    if isinstance(out.get("manager"), dict):
        data["manager"] = _compact_members([out["manager"]], "manager")[0]
    if out.get("raw"):
        data["raw"] = str(out.get("raw") or "")[:2000]
    return data


def _compact_members(items: list[dict], id_key: str) -> list[dict]:
    compact = []
    for item in items:
        compact.append(
            {
                id_key: item.get(id_key),
                "role": item.get("role"),
                "seed": item.get("seed"),
                "answer": item.get("answer") or item.get("predicted_answer"),
                "has_code": bool(item.get("code")),
                "pass_rate": item.get("pass_rate"),
                "resolved": item.get("resolved"),
                "raw_tail": str(item.get("raw") or item.get("raw_tail") or "")[-300:],
            }
        )
    return compact


def _telemetry(base: Any, out: dict) -> dict:
    if out.get("telemetry"):
        return dict(out["telemetry"])
    if out.get("per_agent") and hasattr(base, "langchain_ensemble_telemetry"):
        return base.normalize(base.langchain_ensemble_telemetry(out.get("per_agent") or []))
    return {}


def _error_record(inst: dict, dataset: str, exc: Exception) -> dict:
    rec = {
        "id": inst.get("id"),
        "latency_s": 0,
        "em": 0.0,
        "communication_format": None,
        "communication_parse_ok": False,
        "communication_all_parse_ok": False,
        "communication_parse_rate": 0.0,
        "communication_required_report_count": 0,
        "communication_missing_roles": [],
        "communication_infra_error": f"{type(exc).__name__}: {exc}",
        "communication_parse_errors": [],
        "communication_reports": [],
        "error": f"{type(exc).__name__}: {exc}",
    }
    if dataset == "hotpotqa":
        rec.update({"question": inst.get("question"), "gold_answer": inst.get("answer"), "predicted_answer": None})
    elif dataset == "lcb":
        rec.update({"problem": str(inst.get("problem") or "")[:400], "predicted_code": None, "pass_rate": 0.0})
    elif dataset == "toolhop":
        rec.update({"question": inst.get("question"), "gold_answer": inst.get("answer"), "predicted_answer": None})
    elif dataset == "apibank":
        rec.update(
            {
                "idx": inst.get("idx") or inst.get("id"),
                "question": str(inst.get("chat_history") or "")[:400],
                "gold_api_call": inst.get("gold_api_call"),
                "gold_api_name": (inst.get("ground_truth") or {}).get("api_name"),
                "predicted_answer": "",
                "answer_correct": 0,
                "correct": False,
                "stage": "solve",
            }
        )
    return rec
