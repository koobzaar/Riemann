from __future__ import annotations

import socket
from pathlib import Path

import pytest

from riemann.network_guard import OutboundNetworkBlockedError, inference_no_outbound_context
from riemann.storage import SnapshotStore


def test_empty_sqlite_store_has_no_active_snapshot(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "test.sqlite")
    assert store.get_active_snapshot_id() is None
    assert store.load_active_snapshot() is None


def test_socket_guard_detects_swallowed_outbound_attempt() -> None:
    with pytest.raises(OutboundNetworkBlockedError):
        with inference_no_outbound_context(enabled=True):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.connect(("192.0.2.1", 80))
                finally:
                    s.close()
            except Exception:
                # Even if application code catches and swallows the error,
                # the context manager MUST fail because an attempt occurred.
                pass
