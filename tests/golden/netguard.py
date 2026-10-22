"""Block outbound network access except to the in-process fake LLM server.

Installed in every golden worker process before any runner is imported.
Unix-domain sockets (multiprocessing, asyncio self-pipes) and socketpairs are
allowed; TCP/UDP connects are allowed only to 127.0.0.1/::1 on the ports the
harness registered. Blocked attempts raise ``ConnectionRefusedError`` and are
counted (not stored in goldens: background telemetry threads make them
timing dependent). A redirected port (``redirect``) is a stable virtual port
whose connections reach the fake server's real, randomly assigned port, so
endpoint URLs (and anything derived from them) do not change between runs.
"""

from __future__ import annotations

import socket
import threading

_ALLOWED_PORTS: set[int] = set()
_REDIRECTS: dict[int, int] = {}
_BLOCKED: list[str] = []
_LOCK = threading.Lock()
_INSTALLED = False
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def allow_port(port: int) -> None:
    _ALLOWED_PORTS.add(int(port))


def redirect(virtual_port: int, real_port: int) -> None:
    """Send local connections to ``virtual_port`` to ``real_port`` instead."""
    _REDIRECTS[int(virtual_port)] = int(real_port)
    allow_port(real_port)


def _target(address):
    if isinstance(address, tuple) and len(address) >= 2 and address[0] in _LOCAL_HOSTS:
        port = _REDIRECTS.get(int(address[1]))
        if port is not None:
            return (address[0], port, *address[2:])
    return address


def blocked_attempts() -> list[str]:
    with _LOCK:
        return list(_BLOCKED)


def _check(address) -> None:
    if not isinstance(address, tuple) or len(address) < 2:
        return  # AF_UNIX path or abstract socket
    host, port = address[0], address[1]
    if host in _LOCAL_HOSTS and int(port) in _ALLOWED_PORTS:
        return
    with _LOCK:
        _BLOCKED.append(f"{host}:{port}")
    raise ConnectionRefusedError(f"golden netguard: outbound connection to {host}:{port} blocked")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            address = _target(address)
            _check(address)
        return original_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            address = _target(address)
            try:
                _check(address)
            except ConnectionRefusedError:
                return 111  # ECONNREFUSED
        return original_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in (None, "", *_LOCAL_HOSTS) and not str(host).startswith("127."):
            with _LOCK:
                _BLOCKED.append(f"dns:{host}")
            raise socket.gaierror(socket.EAI_NONAME, f"golden netguard: DNS lookup of {host!r} blocked")
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
