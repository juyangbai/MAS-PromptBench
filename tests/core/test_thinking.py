"""``core.thinking``: reasoning removal from model text and LangChain messages."""

from langchain_core.messages import AIMessage, HumanMessage

from core.thinking import strip_ai_thinking, strip_thinking


def test_strip_thinking_keeps_the_text_after_the_last_tag():
    assert strip_thinking("<think>a</think> b </THINK>  answer ") == "answer"
    assert strip_thinking("  no reasoning  ") == "no reasoning"


def test_strip_ai_thinking_edits_ai_text_only():
    messages = [HumanMessage("<think>q</think>keep"), AIMessage("<think>r</think>done"), AIMessage(["part"])]
    strip_ai_thinking(messages)
    assert [m.content for m in messages] == ["<think>q</think>keep", "done", ["part"]]
