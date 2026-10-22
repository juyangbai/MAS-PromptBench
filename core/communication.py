"""Communication formats: how agents report to each other, and how those reports are scored.

Three formats are studied (:data:`FORMATS`): ``freeform`` (plain text),
``semi_structured`` (tagged sections) and ``structured_soft`` (a JSON report).
A :class:`CommPolicy` carries one run's format and applies it at the three
points where it matters:

* :meth:`CommPolicy.system_prompt` appends the format's reporting contract to an
  agent's system prompt;
* :meth:`CommPolicy.handoff` renders one agent's output in the format before
  another agent receives it, and records the handoff;
* :meth:`CommPolicy.solve` runs a solve function while recording its handoffs and
  adds the parse metrics of the run's reports (:func:`collect_reports`) to its
  output; :func:`compact_output_fields` gives the fields a batch record keeps.

A topology runner builds its policy once, as ``COMMUNICATION``, from the
``COMMUNICATION_FORMAT`` it is loaded with (:mod:`core.variant`), and applies it
itself; nothing rebinds it afterwards.

Parse metrics judge each agent's raw text against the format; the deterministic
rendering always parses and is reported separately.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

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
        return "\nOptional HotpotQA tags may be included when useful:\n[ENTITIES], [HOPS], [ANSWER_CANDIDATE]."
    if dataset == "lcb":
        return (
            "\nOptional LCB tags may be included when useful:\n[APPROACH], [COMPLEXITY], [EDGE_CASES], [CODE_STATUS]."
        )
    if dataset == "toolhop":
        return "\nOptional ToolHop tags may be included when useful:\n[TOOL_CHAIN], [OBSERVATIONS], [ANSWER_CANDIDATE]."
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
    if dataset == "bfcl":
        return "\nOptional BFCL tags may be included when useful:\n[FUNCTION_CHOICE], [ARG_PLAN], [CALL_CANDIDATE]."
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
    if dataset == "bfcl":
        return (
            "For BFCL, payload should use fields when known: function_choice, "
            "arg_plan, call_candidate. The final canonical call list must "
            "still appear as the protected fenced json artifact."
        )
    return ""


def append_contract(prompt: str, fmt: str, dataset: str) -> str:
    contract = communication_contract(fmt, dataset)
    if not contract or "INTER-AGENT COMMUNICATION FORMAT:" in (prompt or ""):
        return prompt
    return (prompt or "").rstrip() + contract


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
        parsed = rendered_parsed = {"ok": True, "errors": []}
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
        # Compliance is measured on the model's own text; the deterministic
        # rendering always parses, so it is only kept as a sanity check.
        parsed = parse_message(raw_text, fmt)
        rendered_parsed = parse_message(rendered, fmt)

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
                "rendered_ok": bool(rendered_parsed.get("ok")),
                "raw_excerpt": _clip_text(raw_text, 1000),
                "rendered_excerpt": _clip_text(rendered, 2000),
            }
        )
    return rendered


def collect_reports(out: dict, *, topology: str, fmt: str, dataset: str = "") -> dict:
    """Render raw runner reports through communications infra and compute strict metrics.

    Parse metrics judge each role's raw model text against the format
    (``communication_parse_rate`` = share of reports that parse as written).
    The deterministic rendering always parses; its rate is reported separately
    as ``communication_render_parse_rate``. ``communication_parse_ok`` remains
    the loose boolean: true if any report parsed.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown communication format {fmt!r}")

    reports = []
    infra_errors = []
    for role, text in _report_texts(out or {}, topology):
        try:
            normalized = normalize_report(role, text, dataset=dataset, topology=topology)
            rendered = render_report(normalized, fmt)
            parsed = parse_message(str(text or ""), fmt)
            rendered_parsed = parse_message(rendered, fmt)
        except Exception as exc:  # pragma: no cover - defensive infra guard
            normalized = normalize_report(role, text, dataset=dataset, topology=topology)
            rendered = ""
            parsed = rendered_parsed = {
                "ok": False,
                "parsed": {},
                "errors": [f"infra render/parse error: {type(exc).__name__}: {exc}"],
            }
            infra_errors.append(f"{role}: {type(exc).__name__}: {exc}")
        reports.append(
            {
                "role": str(role),
                "ok": bool(parsed["ok"]),
                "errors": list(parsed["errors"]),
                "rendered_ok": bool(rendered_parsed["ok"]),
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
    rendered_ok_count = sum(1 for report in reports if report["rendered_ok"])
    render_parse_rate = 1.0 if fmt == "freeform" and total == 0 else (rendered_ok_count / total if total else 0.0)
    parse_errors = [f"{r['role']}: {err}" for r in reports for err in r["errors"]]
    parse_errors.extend(infra_errors)
    role_warnings = [f"missing role report: {role}" for role in missing_roles]
    warnings = parse_errors + role_warnings
    all_parse_ok = not parse_errors and (fmt == "freeform" or (total > 0 and ok_count == total))
    loose_parse_ok = True if fmt == "freeform" else ok_count > 0
    infra_error = "; ".join(infra_errors) if infra_errors else None

    return {
        "communication_format": fmt,
        "communication_parse_ok": loose_parse_ok,
        "communication_all_parse_ok": all_parse_ok,
        "communication_parse_rate": parse_rate,
        "communication_render_parse_rate": render_parse_rate,
        "communication_required_report_count": total,
        "communication_missing_roles": missing_roles,
        "communication_infra_error": infra_error,
        "communication_parse_errors": [] if loose_parse_ok and not infra_error else parse_errors,
        "communication_parse_warnings": warnings,
        "communication_report_ok_count": ok_count,
        "communication_report_total": total,
        "communication_reports": reports,
        "communication_rendered_reports": [
            {"role": report["role"], "rendered": report["rendered_excerpt"]} for report in reports
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
    """The ``(role, text)`` reports of one solve output, read from the first layout that has any.

    A ``by_stage`` mapping (sequential runs) is authoritative even when it holds no report.
    Otherwise the layouts are tried in this order and the first with a report wins: chat
    ``messages``, ``per_agent`` replicas, ``per_peer`` debaters, ``workers`` plus ``manager``,
    the last AI message of each of ``all_contexts``, and ``raw`` as a single ``final`` report.
    """
    if isinstance(out.get("by_stage"), dict):
        return [(str(role), str(text or "")) for role, text in out["by_stage"].items() if text]
    for reports in _layout_reports(out, topology):
        if reports:
            return reports
    return []


def _layout_reports(out: dict, topology: str) -> Iterator[list[tuple[str, str]]]:
    """The reports of each layout present in ``out``, in priority order (computed lazily)."""
    if isinstance(out.get("messages"), list):
        yield _chat_reports(out["messages"])
    if isinstance(out.get("per_agent"), list):
        yield _replica_reports(out["per_agent"])
    if isinstance(out.get("per_peer"), list):
        yield _member_reports(out["per_peer"], lambda item, idx: f"peer_{item.get('peer', idx)}")
    if isinstance(out.get("workers"), list):
        yield _hub_reports(out["workers"], out.get("manager"))
    yield _context_reports(out.get("all_contexts") or [], topology)
    raw = out.get("raw")
    yield [("final", str(raw))] if raw else []


def _chat_reports(messages: list) -> list[tuple[str, str]]:
    """Non-empty assistant messages, labelled by their ``source`` (system, user and tool excluded)."""
    reports = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role and role != "assistant":
            continue
        source = str(msg.get("source") or role or f"message_{idx}")
        if source in {"system", "user", "tool"}:
            continue
        content = str(msg.get("content") or "")
        if content.strip():
            reports.append((source, content))
    return reports


def _member_text(item: dict) -> Any:
    """A team member's report: the first non-empty of its raw, final, predicted and tail texts."""
    return item.get("raw") or item.get("final_content") or item.get("predicted_answer") or item.get("raw_tail")


def _member_reports(items: list, label: Callable[[dict, int], str]) -> list[tuple[str, str]]:
    """The non-empty member texts of ``items``, each labelled by ``label(item, index)``."""
    reports = []
    for idx, item in enumerate(items):
        text = _member_text(item)
        if text:
            reports.append((label(item, idx), str(text)))
    return reports


def _replica_reports(items: list) -> list[tuple[str, str]]:
    """``per_agent`` replicas as ``agent_<id>``; a BFCL replica may have answered only with tool calls."""
    reports = []
    for idx, item in enumerate(items):
        text = _member_text(item) or _model_output_text(item)
        if text:
            reports.append((f"agent_{item.get('agent_id', idx)}", str(text)))
    return reports


def _hub_reports(workers: list, manager: Any) -> list[tuple[str, str]]:
    """The workers (labelled by role, else ``worker_<i>``), then the manager when there is one."""
    reports = _member_reports(workers, lambda item, idx: str(item.get("role") or f"worker_{idx}"))
    if isinstance(manager, dict):
        reports += _member_reports([manager], lambda item, _idx: str(item.get("role") or "manager"))
    return reports


def _context_reports(contexts: list, topology: str) -> list[tuple[str, str]]:
    """The last AI text of each agent context, as ``agent_<i>`` (independent) or ``peer_<i>``."""
    label = "agent" if topology == "independent" else "peer"
    reports = []
    for idx, ctx in enumerate(contexts):
        text = _last_ai_text(ctx)
        if text:
            reports.append((f"{label}_{idx}", text))
    return reports


def _model_output_text(item: dict) -> str:
    """Canonical call list of a BFCL replica that answered via schema tools."""
    calls = item.get("model_output")
    if isinstance(calls, list) and calls:
        return json.dumps(calls, ensure_ascii=False, sort_keys=True, default=str)
    return ""


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


def compact_output_fields(out: dict) -> dict:
    """The communication fields of a solve output that a batch record keeps (long texts clipped)."""
    data = {
        "communication_format": out.get("communication_format"),
        "communication_parse_ok": out.get("communication_parse_ok"),
        "communication_all_parse_ok": out.get("communication_all_parse_ok"),
        "communication_parse_rate": out.get("communication_parse_rate"),
        "communication_render_parse_rate": out.get("communication_render_parse_rate"),
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


@dataclass(frozen=True)
class CommPolicy:
    """The communication format of one run on one dataset and topology."""

    fmt: str = "freeform"
    dataset: str = ""
    topology: str = ""

    def __post_init__(self) -> None:
        if self.fmt not in FORMATS:
            raise ValueError(f"unknown communication format {self.fmt!r}")

    def system_prompt(self, prompt: str) -> str:
        """``prompt`` with the format's reporting contract appended (unchanged for freeform)."""
        return append_contract(prompt, self.fmt, self.dataset)

    def handoff(self, role: str, text: Any, **report: Any) -> str:
        """``role``'s output rendered for the next agent; ``report`` takes :func:`format_handoff`'s report fields."""
        return format_handoff(role, text, fmt=self.fmt, dataset=self.dataset, topology=self.topology, **report)

    def solve(self, solve: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``solve`` recording its handoffs; a dict result gains the handoff records and report metrics."""
        token = begin_handoff_recording()
        try:
            out = solve(*args, **kwargs)
        finally:
            handoffs = end_handoff_recording(token)
        if not isinstance(out, dict):
            return out
        out = dict(out)
        out["communication_inflight_handoffs"] = handoffs
        out["communication_inflight_handoff_count"] = len(handoffs)
        out["communication_inflight_all_parse_ok"] = all(bool(item.get("ok")) for item in handoffs)
        out.update(collect_reports(out, topology=self.topology, fmt=self.fmt, dataset=self.dataset))
        return out
