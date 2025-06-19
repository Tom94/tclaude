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
import inspect
import os
import sys
from collections.abc import Awaitable
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Callable, TextIO, cast

from .common import ANSI_HIDE_CURSOR, ANSI_SHOW_CURSOR
from .spinner import SPINNER_FPS


def nth_rfind(string: str, char: str, n: int) -> int:
    pos = len(string)
    for _ in range(n):
        pos = string.rfind(char, 0, pos)
        if pos == -1:
            return -1
    return pos


@asynccontextmanager
async def live_print(get_live_text: Callable[[bool], str] | Callable[[bool], Awaitable[str]], transient: bool = True):
    original_stdout: TextIO = sys.stdout
    with StringIO() as stdout, redirect_stdout(stdout), redirect_stderr(stdout):
        num_newlines_printed = 0

        async def clear_and_print(final: bool):
            nonlocal num_newlines_printed

            to_print = StringIO()
            _ = to_print.write(ANSI_SHOW_CURSOR if final else ANSI_HIDE_CURSOR)

            # Move the cursor up by the number of newlines printed so far, then clear the screen from the cursor down
            if num_newlines_printed > 0:
                _ = to_print.write(f"\033[{num_newlines_printed}F")
            _ = to_print.write("\r\033[J")

            if final and transient:
                _ = to_print.write(stdout.getvalue())
                print(to_print.getvalue(), end="", flush=True, file=original_stdout)
                return

            term_height = os.get_terminal_size().lines

            if inspect.iscoroutinefunction(get_live_text):
                text = cast(str, await get_live_text(final))
            else:
                text = cast(str, get_live_text(final))

            if not stdout.tell() == 0:
                text = f"{text}\n\n{stdout.getvalue().rstrip()}"

            # Print the last term_height - 1 lines of the history to avoid terminal problems upon clearing again.
            # However, if we're the final print, we no longer need to clear, so we should print all lines.
            if not final:
                split_idx = nth_rfind(text, "\n", term_height)
                if split_idx != -1:
                    text = text[split_idx + 1 :]

            _ = to_print.write(text)

            print(to_print.getvalue(), end="", flush=True, file=original_stdout)

            num_newlines_printed = text.count("\n")

        async def live_print_task():
            # Initial wait for 1ms in case the task is already cancelled and nothing has to be printed.
            await asyncio.sleep(1.0 / 1000.0)
            while True:
                await clear_and_print(final=False)
                await asyncio.sleep(1.0 / SPINNER_FPS)

        task = asyncio.create_task(live_print_task())
        try:
            yield task
        finally:
            _ = task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                await clear_and_print(final=True)
