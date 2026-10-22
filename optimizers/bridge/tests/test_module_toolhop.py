"""ToolHop adapter output when the runner returns only its message transcript."""

from __future__ import annotations

from optimizers.bridge.adapters.module_common import import_real_module
from optimizers.bridge.adapters.module_toolhop import SingleToolHopAdapter


def test_answer_falls_back_to_the_last_assistant_message():
    adapter = SingleToolHopAdapter()
    runner = import_real_module(adapter.runtime_module_name())
    messages = [
        {"role": "user", "content": "Which city?"},
        {"role": "assistant", "content": "Let me look it up."},
        {"role": "tool", "content": "Paris"},
        {"role": "assistant", "content": "<answer>Paris</answer>"},
    ]

    output = adapter.adapter_output(runner, {"id": "0"}, {"messages": messages})

    assert output["raw"] == output["answer"] == "<answer>Paris</answer>"
    assert output["predicted_answer"] == "Paris"
