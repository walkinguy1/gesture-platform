"""
Tests for the WebSocket bridge -- cross-thread broadcasting and client
message dispatch. Uses real localhost WebSocket connections (via
websockets.sync.client) rather than mocks, since the bug this bridge had
(asyncio.create_task() called from the wrong thread) only reproduces with a
genuine separate event-loop thread.
"""

import itertools
import json
import time

import pytest
from websockets.sync.client import connect

from gesture_platform.ws_bridge import WSBridgeThread

_port_counter = itertools.count(8790)


@pytest.fixture
def bridge_thread():
    received = []
    port = next(_port_counter)
    bridge = WSBridgeThread(
        host="127.0.0.1",
        port=port,
        on_message=lambda data, client: received.append(data),
    )
    assert bridge.start(timeout=5.0)
    yield bridge, received, port
    bridge.stop()


def test_client_receives_broadcast_from_main_thread(bridge_thread):
    """The exact scenario that broke pre-fix: broadcast() called from a
    thread other than the one running the bridge's asyncio event loop."""
    bridge, _received, port = bridge_thread

    with connect(f"ws://127.0.0.1:{port}") as client:
        time.sleep(0.2)  # let the server register the connection

        bridge.broadcast_prediction("A", 0.95, fps=30.0, prediction_kind="static")

        message = json.loads(client.recv(timeout=3.0))
        assert message["type"] == "prediction"
        assert message["prediction"] == "A"
        assert message["confidence"] == pytest.approx(0.95)
        assert message["prediction_kind"] == "static"


def test_broadcast_languages_and_language_changed(bridge_thread):
    bridge, _received, port = bridge_thread

    with connect(f"ws://127.0.0.1:{port}") as client:
        time.sleep(0.2)

        bridge.broadcast_languages(["ASL", "BSL"], active="ASL")
        msg1 = json.loads(client.recv(timeout=3.0))
        assert msg1 == {"type": "languages", "languages": ["ASL", "BSL"], "active": "ASL"}

        bridge.broadcast_language_changed("BSL", {"dynamic_ready": False})
        msg2 = json.loads(client.recv(timeout=3.0))
        assert msg2["type"] == "language_changed"
        assert msg2["code"] == "BSL"


def test_server_receives_client_message(bridge_thread):
    bridge, received, port = bridge_thread

    with connect(f"ws://127.0.0.1:{port}") as client:
        client.send(json.dumps({"type": "set_language", "code": "BSL"}))
        time.sleep(0.3)

    assert any(
        msg.get("type") == "set_language" and msg.get("code") == "BSL"
        for msg in received
    )


def test_malformed_message_does_not_crash_bridge(bridge_thread):
    bridge, received, port = bridge_thread

    with connect(f"ws://127.0.0.1:{port}") as client:
        client.send("not json")
        time.sleep(0.2)

    assert received == []
    assert bridge.is_running()


def test_broadcast_with_no_clients_is_a_noop(bridge_thread):
    bridge, _received, _port = bridge_thread
    # Should not raise even though nobody is connected.
    bridge.broadcast_prediction(None, 0.0, fps=0.0)
