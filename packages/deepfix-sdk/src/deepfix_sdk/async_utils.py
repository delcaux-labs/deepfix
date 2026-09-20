"""Asynchronous execution utilities for synchronous environments.

Provides a persistent background event loop thread for executing async workflows,
coroutines, and callables synchronously without event loop conflicts.
"""

import asyncio
import concurrent.futures
import inspect
import threading
from typing import Any, Optional


class AsyncLoopThread:
    """Persistent background event loop thread for executing async workflows synchronously."""

    _instance: Optional["AsyncLoopThread"] = None
    _lock = threading.Lock()

    def __init__(self, thread_name: str = "deepfix-async-runner"):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever,
            daemon=True,
            name=thread_name,
        )
        self.thread.start()

    @classmethod
    def get_instance(cls) -> "AsyncLoopThread":
        with cls._lock:
            if cls._instance is None or not cls._instance.thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def run(self, coro_or_fn: Any) -> Any:
        """Run a coroutine, awaitable, or callable returning an awaitable on the background loop."""
        future: concurrent.futures.Future = concurrent.futures.Future()

        def _schedule():
            async def _coro():
                try:
                    res = coro_or_fn() if callable(coro_or_fn) else coro_or_fn
                    if inspect.isawaitable(res):
                        res = await res
                    future.set_result(res)
                except BaseException as exc:
                    future.set_exception(exc)

            asyncio.create_task(_coro())

        self.loop.call_soon_threadsafe(_schedule)
        return future.result()

    def stop(self) -> None:
        """Stop the event loop and wait for thread termination."""
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive() and threading.current_thread() != self.thread:
            self.thread.join(timeout=2.0)


def run_async(coro_or_fn: Any) -> Any:
    """Run an async coroutine or callable synchronously using the persistent background event loop."""
    return AsyncLoopThread.get_instance().run(coro_or_fn)
