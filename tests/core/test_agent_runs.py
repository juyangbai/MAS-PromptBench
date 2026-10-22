"""``core.agent_runs``: the topology check, handoff contexts and record batches of the API-Bank and ToolHop runners."""

import json
import logging

import pytest

from core import agent_runs


def test_check_topology():
    agent_runs.check_topology("single_openai", "single")
    with pytest.raises(ValueError, match="this runner handles 'single'; received topology='independent'"):
        agent_runs.check_topology("independent", "single")
    agent_runs.check_topology("decentralized_openai_agents", "decentralized", suffixes=("_openai_agents", "_openai"))


def test_reports_context_plain_and_formatted():
    reports = [{"role": "a", "raw": "x" * 2000}, {"role": "b", "raw": ""}, {"seed": 1, "predicted_answer": "[C()]"}]
    plain = agent_runs.reports_context(reports, dataset="apibank")
    assert plain.startswith("a:\n" + "x" * 1200) and plain.endswith("agent_1:\n[C()]")
    assert agent_runs.reports_context(reports, dataset="apibank", char_budget=10) == plain[-10:]
    assert "[STATUS]" in agent_runs.reports_context(reports, "semi_structured", dataset="toolhop")


def test_run_rows_writes_records_and_predictions_afresh(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="core")

    def run_one(instance, out_dir):
        return {"id": instance["id"], "correct": instance["id"] == "a", "per_agent": ["ü"], "out": out_dir.name}

    def prediction(instance, record):
        return {"id": instance["id"], "answer": record["correct"]}

    options = {"dataset": "d", "source": "src", "style": "s", "model_id": "m", "prediction": prediction}
    for _ in range(2):
        result = agent_runs.run_rows(
            [{"id": "a"}, {"id": "b"}], run_one, out_dir=tmp_path, team_size=2, hidden=("per_agent",), **options
        )
    assert result == {"n": 2, "correct": 1, "accuracy": 0.5, "style": "s", "team_size": 2}
    records = (tmp_path / "results.jsonl").read_text().splitlines()
    assert len(records) == 2 and '"per_agent": ["ü"]' in records[0]
    predictions = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert predictions[1] == {"id": "b", "answer": False, "model_name_or_path": "m"}
    assert caplog.messages[:2] == ["loaded 2 instance(s) from src (s, N=2)", "\n[1/2] a"]
    assert caplog.messages[2].startswith("  -> {")
    assert "per_agent" not in caplog.text


def test_write_json_keeps_unicode(tmp_path):
    path = tmp_path / "traces" / "x.json"
    agent_runs.write_json(path, {"a": "ü", "p": tmp_path})
    assert json.loads(path.read_text()) == {"a": "ü", "p": str(tmp_path)} and "ü" in path.read_text()
