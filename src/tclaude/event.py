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
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import Callable, Generic, ParamSpec, TypeAlias, cast

P = ParamSpec('P')

Handler: TypeAlias = Callable[P, None | Awaitable[None]]

class AsyncEventEmitter(Generic[P]):
    def __init__(self):
        self._handlers: list[Handler[P]] = []

    def subscribe(self, handler: Handler[P]):
        self._handlers.append(handler)

    def unsubscribe(self, handler: Handler[P]):
        self._handlers.remove(handler)

    async def emit(self, *args: P.args, **kwargs: P.kwargs):
        for handler in self._handlers:
            if asyncio.iscoroutinefunction(handler):
                await handler(*args, **kwargs)
            else:
                cb = cast(Callable[..., None], handler)
                cb(*args, **kwargs)

    @asynccontextmanager
    async def temp_subscribe(self, handler: Handler[P]):
        self.subscribe(handler)
        try:
            yield
        finally:
            self.unsubscribe(handler)
