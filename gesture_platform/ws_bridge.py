"""
WebSocket Bridge Module
Real-time, bidirectional bridge between the Python recognition backend and
the React frontend.

Runs an asyncio WebSocket server on ws://127.0.0.1:8765. Broadcasts
prediction results (and the active sign-language list) to every connected
client, and forwards client -> backend commands (e.g. "switch to BSL") to
an ``on_message`` callback supplied by the owner (see ``realtime_demo.py``).

Usage:
    def handle_message(data, client):
        if data.get("type") == "set_language":
            ...

    bridge = WSBridge(on_message=handle_message)
    await bridge.start()
    bridge.broadcast_prediction("A", 0.95, fps=28.0, prediction_kind="static")
    await bridge.stop()
"""

import asyncio
import json
import logging
import threading
from typing import Any, Callable, Optional, Set

import websockets

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict, Any], None]


class WSBridge:
    """
    WebSocket bridge for real-time prediction broadcasting and client commands.

    Manages WebSocket connections, broadcasts prediction/language data to all
    connected frontend clients, and dispatches incoming client JSON messages
    to an optional callback.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_message: Optional[MessageHandler] = None,
    ):
        """
        Args:
            host: Host address to bind to
            port: Port to listen on
            on_message: Optional callback ``(data: dict, client) -> None``
                invoked for every JSON message received from a client.
        """
        self.host = host
        self.port = port
        self.clients: Set[Any] = set()
        self.server = None
        self._running = False
        self.on_message = on_message

        # The event loop actually running the server. `broadcast*()` may be
        # called from a different thread (e.g. the pipeline's main capture
        # loop), so sends must be scheduled onto this loop rather than
        # assumed to already be running on the calling thread.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _handle_client(self, websocket) -> None:
        """Handle a connected WebSocket client for its whole lifetime."""
        self.clients.add(websocket)
        logger.info("Client connected. Total clients: %d", len(self.clients))

        try:
            async for message in websocket:
                self._dispatch_message(message, websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected normally")
        except Exception as e:
            logger.error("Error handling client: %s", e)
        finally:
            self.clients.discard(websocket)
            logger.info("Client removed. Total clients: %d", len(self.clients))

    def _dispatch_message(self, message: str, client: Any) -> None:
        """Parse an incoming client message and forward it to on_message."""
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Ignoring non-JSON client message: %r", message)
            return

        if not isinstance(data, dict):
            logger.debug("Ignoring non-object client message: %r", data)
            return

        if self.on_message is not None:
            try:
                self.on_message(data, client)
            except Exception:
                logger.exception("Error in on_message handler for: %r", data)

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            logger.warning("WSBridge already running")
            return

        self._loop = asyncio.get_running_loop()
        self.server = await websockets.serve(self._handle_client, self.host, self.port)
        self._running = True
        logger.info("WebSocket bridge started on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self._running = False
        self._loop = None
        logger.info("WebSocket bridge stopped")

    # ------------------------------------------------------------------
    # Outbound messages (safe to call from any thread)
    # ------------------------------------------------------------------

    def broadcast(self, payload: dict) -> None:
        """
        Broadcast a JSON-serializable payload to all connected clients.

        Safe to call from any thread: if the caller isn't running on the
        bridge's own event loop, the send is scheduled onto it via
        ``run_coroutine_threadsafe`` instead of ``create_task`` (which
        requires an already-running loop on the *current* thread).
        """
        if not self.clients or self._loop is None:
            return

        message_str = json.dumps(payload)
        clients_to_send = list(self.clients)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        for client in clients_to_send:
            coro = self._safe_send(client, message_str)
            if running_loop is self._loop:
                self._loop.create_task(coro)
            else:
                asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _safe_send(self, client: Any, message_str: str) -> None:
        try:
            await client.send(message_str)
        except Exception as e:
            logger.debug("Error sending to client: %s", e)
            # Client will be removed by _handle_client on its next error.

    def broadcast_prediction(
        self,
        prediction: Optional[str],
        confidence: float,
        fps: float,
        prediction_kind: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        """Broadcast a single recognition result to all connected clients."""
        self.broadcast({
            "type": "prediction",
            "prediction": prediction,
            "confidence": confidence,
            "fps": fps,
            "prediction_kind": prediction_kind,
            "language": language,
        })

    def broadcast_languages(self, languages: list, active: Optional[str]) -> None:
        """Broadcast the available sign-language list and the active one."""
        self.broadcast({
            "type": "languages",
            "languages": languages,
            "active": active,
        })

    def broadcast_language_changed(self, code: str, status: dict) -> None:
        """Broadcast confirmation that the active language changed."""
        self.broadcast({
            "type": "language_changed",
            "code": code,
            "status": status,
        })

    def broadcast_error(self, message: str) -> None:
        """Broadcast a backend error string for the UI to surface."""
        self.broadcast({"type": "error", "message": message})

    def is_running(self) -> bool:
        """Check if the bridge is running."""
        return self._running

    def client_count(self) -> int:
        """Get the number of connected clients."""
        return len(self.clients)


class WSBridgeThread:
    """
    Convenience wrapper that runs a WSBridge on a dedicated background
    thread with its own asyncio event loop, so synchronous callers (e.g. a
    camera capture loop) can start/stop/broadcast without managing asyncio
    themselves.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, on_message: Optional[MessageHandler] = None):
        self.bridge = WSBridge(host=host, port=port, on_message=on_message)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self, timeout: float = 5.0) -> bool:
        """Start the bridge on a background thread; blocks until it's listening."""
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _run():
                await self.bridge.start()
                self._ready.set()
                while self.bridge.is_running():
                    await asyncio.sleep(0.5)

            try:
                self._loop.run_until_complete(_run())
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=run, name="WSBridge", daemon=True)
        self._thread.start()
        return self._ready.wait(timeout=timeout)

    def stop(self) -> None:
        """Stop the bridge and join its thread."""
        if self._loop is None or not self.bridge.is_running():
            return
        asyncio.run_coroutine_threadsafe(self.bridge.stop(), self._loop)
        if self._thread:
            self._thread.join(timeout=2.0)

    def broadcast_prediction(self, *args, **kwargs) -> None:
        self.bridge.broadcast_prediction(*args, **kwargs)

    def broadcast_languages(self, *args, **kwargs) -> None:
        self.bridge.broadcast_languages(*args, **kwargs)

    def broadcast_language_changed(self, *args, **kwargs) -> None:
        self.bridge.broadcast_language_changed(*args, **kwargs)

    def broadcast_error(self, *args, **kwargs) -> None:
        self.bridge.broadcast_error(*args, **kwargs)

    def is_running(self) -> bool:
        return self.bridge.is_running()

    def client_count(self) -> int:
        return self.bridge.client_count()


# Convenience function for standalone usage
async def run_bridge(host: str = "127.0.0.1", port: int = 8765):
    """
    Run the bridge standalone (useful for testing).

    Args:
        host: Host address
        port: Port number
    """
    bridge = WSBridge(host, port)
    await bridge.start()

    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down bridge...")
    finally:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(run_bridge())
