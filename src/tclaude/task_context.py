# tclaude -- Claude in the terminal
#
# Copyright (C) 2025 Thomas Müller <contact@tom94.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, AsyncContextManager, Generic, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__package__)


class TaskAsyncContextManager(Generic[T]):
    """Wrapper for a single long-lived async context manager"""

    def __init__(self, context_manager: AsyncContextManager[T]):
        self.context_manager: AsyncContextManager[T] = context_manager
        self.task: asyncio.Task[None] | None = None
        self.ready_future: asyncio.Future[T] | None = None

    async def __aenter__(self) -> asyncio.Future[T]:
        return self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        await self.stop()

    def start(self) -> asyncio.Future[T]:
        """Start the context manager and return the resource"""
        if self.task is not None:
            raise RuntimeError("Already started")

        self.ready_future = asyncio.Future()
        self.task = asyncio.create_task(self._run())

        # Wait for context manager to be ready
        return self.ready_future

    async def _run(self):
        """Run the context manager in a long-lived task"""
        try:
            async with self.context_manager as resource:
                assert self.ready_future is not None, "Ready future must be set before running"
                self.ready_future.set_result(resource)

                # Keep the context alive indefinitely until externally cancelled
                try:
                    _ = await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise
        except Exception as e:
            if self.ready_future and not self.ready_future.done():
                self.ready_future.set_exception(e)
            raise

    async def stop(self):
        """Stop the context manager"""
        if self.task and not self.task.done():
            _ = self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass


class AsyncContextPool:
    """Manager for multiple parallel async context managers"""

    def __init__(self):
        self.managers: list[TaskAsyncContextManager[Any]] = []  # pyright: ignore[reportExplicitAny]

    async def add[T](self, context_manager: AsyncContextManager[T]) -> T:
        """Add and start a new context manager"""
        manager = TaskAsyncContextManager[T](context_manager)
        self.managers.append(manager)
        return await manager.start()

    async def add_many(self, *args: AsyncContextManager[T]) -> list[T | BaseException]:
        """Add multiple context managers in parallel"""
        tasks: list[Awaitable[T]] = []
        for cm in args:
            manager = TaskAsyncContextManager[T](cm)
            self.managers.append(manager)
            tasks.append(manager.start())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def clear(self):
        """Stop and remove all context managers"""
        if not self.managers:
            return

        # Stop all in parallel
        _ = await asyncio.gather(*[m.stop() for m in self.managers], return_exceptions=True)
        self.managers.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        await self.clear()
