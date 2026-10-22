"""In-process fake OpenAI-compatible chat-completions server.

Every request body is recorded verbatim (messages, tools, decoding fields,
extra_body keys ...). Responses are a pure function of the request body and
the per-cell script installed by the harness (see ``responder.py``), so a
run's transcript does not depend on thread scheduling even when a topology
fans requests out concurrently.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Responder = Callable[[dict], dict]


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _approx_tokens(value: Any) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return max(1, len(value) // 4)


class _ThreadingServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with a listen backlog sized for bursts.

    The stdlib default backlog is 5. A method that fans out dozens of
    concurrent requests (MAPRO's reflection, judge and probe calls) overflows
    it when the machine is loaded and the accept loop lags; the kernel then
    drops the half-open connection and the client, which sends with no SDK
    retry, sees ``Connection reset by peer`` instead of a reply.
    """

    daemon_threads = True
    request_queue_size = 1024


class FakeChatServer:
    """Threaded localhost server; ``base_url`` ends with ``/v1``."""

    def __init__(self, responder: Responder, normalize: Callable[[dict], Any] | None = None):
        """``normalize`` maps a request body to the machine-independent form
        used to derive response ids and token counts (e.g. temp paths masked)."""
        self.responder = responder
        self.normalize = normalize or (lambda body: body)
        self.requests: list[dict] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._inflight = 0
        self.max_inflight = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args, **kwargs):  # silence stderr access log
                return None

            def _send_json(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # http.server handler name
                if self.path.rstrip("/").endswith("/models"):
                    self._send_json(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": "golden-model", "object": "model", "created": 0, "owned_by": "golden"}],
                        },
                    )
                    return
                self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})

            def do_POST(self):  # http.server handler name
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                except Exception as exc:  # pragma: no cover - defensive
                    server.errors.append(f"bad json: {exc}")
                    self._send_json(400, {"error": {"message": "bad json"}})
                    return
                path = self.path.split("?", 1)[0]
                if not path.rstrip("/").endswith("/chat/completions"):
                    with server._lock:
                        server.requests.append({"path": path, "body": body, "seq": len(server.requests)})
                    server.errors.append(f"unsupported path {path}")
                    self._send_json(404, {"error": {"message": f"unsupported path {path}"}})
                    return
                with server._lock:
                    server._inflight += 1
                    server.max_inflight = max(server.max_inflight, server._inflight)
                    seq = len(server.requests)
                    server.requests.append({"path": path, "body": body, "seq": seq})
                try:
                    try:
                        reply = server.responder(body)
                    except Exception as exc:  # responder bug: surface, do not hang the client
                        server.errors.append(f"responder error: {type(exc).__name__}: {exc}")
                        reply = {"content": f"responder error: {exc}"}
                    if body.get("stream"):
                        self._stream(body, reply)
                    else:
                        self._send_json(200, server.completion(server.normalize(body), reply))
                finally:
                    with server._lock:
                        server._inflight -= 1

            def _stream(self, body: dict, reply: dict) -> None:
                completion = server.completion(server.normalize(body), reply)
                choice = completion["choices"][0]
                message = choice["message"]
                chunks = []
                base = {k: completion[k] for k in ("id", "object", "created", "model")}
                base["object"] = "chat.completion.chunk"
                delta: dict = {"role": "assistant", "content": message.get("content") or ""}
                if message.get("tool_calls"):
                    delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]
                chunks.append({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                chunks.append(
                    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}
                )
                if (body.get("stream_options") or {}).get("include_usage"):
                    chunks.append({**base, "choices": [], "usage": completion["usage"]})
                payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                data = payload.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = _ThreadingServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="golden-fake-llm", daemon=True)

    # lifecycle
    def start(self) -> FakeChatServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def reset(self) -> None:
        with self._lock:
            self.requests = []
            self.errors = []
            self.max_inflight = 0

    def record(self, path: str, body: dict) -> None:
        """Log a pseudo request that did not travel over HTTP (Agents SDK turns)."""
        with self._lock:
            self.requests.append({"path": path, "body": body, "seq": len(self.requests)})

    def drain(self, timeout: float = 5.0) -> None:
        """Wait until no request is being served (late concurrent stragglers)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._inflight == 0:
                    return
            time.sleep(0.01)

    # response construction (deterministic)
    @staticmethod
    def completion(body: dict, reply: dict) -> dict:
        """Deterministic completion for ``body`` (already normalized)."""
        key = _digest(body)
        content = reply.get("content")
        tool_calls = []
        for index, call in enumerate(reply.get("tool_calls") or []):
            arguments = call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": "call_" + hashlib.sha1(f"{key}|{index}".encode()).hexdigest()[:16],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": arguments},
                }
            )
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        prompt_tokens = _approx_tokens(body.get("messages")) + _approx_tokens(body.get("tools"))
        completion_tokens = _approx_tokens(content) + _approx_tokens(
            [c["function"] for c in tool_calls] if tool_calls else None
        )
        return {
            "id": "chatcmpl-" + key[:24],
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model") or "golden-model",
            "system_fingerprint": "golden",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
