"""``core.runtime``: replay-safe LangChain histories and the per-row wall-clock guard."""

import signal
import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from core.batch import attempt
from core.runtime import openai_safe_messages, row_timeout


def test_openai_safe_messages_rebuilds_ai_messages():
    raw = AIMessage(
        content="x",
        tool_calls=[{"name": "f", "args": {"p": Path("a")}, "id": None}],
        additional_kwargs={"source": "coder", "tool_calls": [{"bad": "payload"}]},
        response_metadata={"token_usage": {"total_tokens": 3}},
    )
    human = HumanMessage("q")
    safe = openai_safe_messages([human, raw, AIMessage(content=["part"])])
    assert safe[0] is human
    assert safe[1].content == "x" and safe[1].additional_kwargs == {"source": "coder"}
    assert safe[1].tool_calls == [{"name": "f", "args": {"p": "a"}, "id": "call_0", "type": "tool_call"}]
    assert safe[1].response_metadata == {}
    assert safe[2].content == "['part']" and safe[2].additional_kwargs == {}


def sleep_under(guard, seconds: float) -> dict:
    with guard:
        time.sleep(seconds)
    return {"answer": "ok"}


def test_row_timeout_interrupts_a_slow_row():
    out, _, error = attempt(lambda: sleep_under(row_timeout(1), 3))
    assert out == {"answer": None} and error == "_RowTimeout: row exceeded 1s"
    assert sleep_under(row_timeout(5), 0) == {"answer": "ok"}
    assert signal.alarm(0) == 0


def test_row_timeout_is_a_no_op_off_the_main_thread():
    done = []
    worker = threading.Thread(target=lambda: done.append(sleep_under(row_timeout(1), 0)))
    worker.start()
    worker.join()
    assert done == [{"answer": "ok"}]
