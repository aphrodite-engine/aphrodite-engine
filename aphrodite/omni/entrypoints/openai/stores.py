# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from aphrodite.omni.entrypoints.openai.protocol.videos import VideoResponse

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U")


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> asyncio.Task[None] | None:
        async with self._lock:
            return self._tasks.get(key)

    async def pop(self, key: str) -> asyncio.Task[None] | None:
        async with self._lock:
            return self._tasks.pop(key, None)

    async def upsert(self, key: str, task: asyncio.Task[None]) -> None:
        async with self._lock:
            self._track(key, task)

    def _track(self, key: str, task: asyncio.Task[None]) -> None:
        def _cleanup(completed: asyncio.Task[None]) -> None:
            # Callbacks run on the event loop; do not remove a replacement job.
            if self._tasks.get(key) is completed:
                self._tasks.pop(key)

        task.add_done_callback(_cleanup)

        self._tasks[key] = task

    async def try_start(self, key: str, create: Callable[[], Coroutine[Any, Any, None]], limit: int) -> bool:
        async with self._lock:
            if sum(not task.done() for task in self._tasks.values()) >= limit:
                return False
            self._track(key, asyncio.create_task(create()))
            return True

    async def cancel_all(self) -> None:
        async with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class AsyncDictStore(Generic[T]):
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: dict[str, T] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: T) -> None:
        async with self._lock:
            self._items[key] = value

    async def update_fields(self, key: str, updates: dict[str, Any]) -> T | None:
        async with self._lock:
            item: T | None = self._items.get(key)
            if item is None:
                return None
            new_item = item.model_copy(update=updates)
            self._items[key] = new_item
            return new_item

    async def get(self, key: str) -> T | None:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> T | None:
        async with self._lock:
            return self._items.pop(key, None)

    async def list_values(self) -> list[T]:
        async with self._lock:
            return list(self._items.values())


VIDEO_STORE: AsyncDictStore[VideoResponse] = AsyncDictStore()
VIDEO_TASKS: TaskRegistry = TaskRegistry()
