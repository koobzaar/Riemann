from __future__ import annotations

import contextlib
import contextvars
import socket
import threading
from typing import Any, Iterator

_inference_guard_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "riemann_inference_guard_active", default=False
)
_attempt_counter_var: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "riemann_outbound_attempts", default=None
)

_lock = threading.Lock()
_installed = False
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_global_attempts: list[str] = []


class OutboundNetworkBlockedError(RuntimeError):
    """Raised when outbound network connection is attempted during inference."""


def _is_loopback_address(address: Any) -> bool:
    if isinstance(address, tuple) and len(address) >= 1:
        host = str(address[0])
        return host in {"127.0.0.1", "::1", "localhost"}
    return False


def _record_and_block(target: str) -> None:
    with _lock:
        _global_attempts.append(target)
    lst = _attempt_counter_var.get()
    if lst is not None:
        lst.append(target)
    raise OutboundNetworkBlockedError(
        f"Outbound network connection blocked during inference: {target}"
    )


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    if _inference_guard_active.get() and not _is_loopback_address(address):
        _record_and_block(f"connect({address!r})")
    return _orig_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    if _inference_guard_active.get() and not _is_loopback_address(address):
        _record_and_block(f"connect_ex({address!r})")
    return _orig_connect_ex(self, address)


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    if _inference_guard_active.get() and not _is_loopback_address(address):
        _record_and_block(f"create_connection({address!r})")
    return _orig_create_connection(address, *args, **kwargs)


def install_socket_guard() -> None:
    global _installed
    with _lock:
        if _installed:
            return
        socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
        socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
        _installed = True


def get_global_outbound_attempts() -> list[str]:
    with _lock:
        return list(_global_attempts)


def reset_global_outbound_attempts() -> None:
    with _lock:
        _global_attempts.clear()


@contextlib.contextmanager
def inference_no_outbound_context(enabled: bool = True) -> Iterator[list[str]]:
    """Block and record any outbound socket attempts inside an inference call."""
    attempts: list[str] = []
    if not enabled:
        yield attempts
        return
    install_socket_guard()
    tok_active = _inference_guard_active.set(True)
    tok_list = _attempt_counter_var.set(attempts)
    try:
        yield attempts
    finally:
        _inference_guard_active.reset(tok_active)
        _attempt_counter_var.reset(tok_list)
    if attempts:
        raise OutboundNetworkBlockedError(
            f"Swallowed or uncaught outbound connection attempts detected: {attempts}"
        )
