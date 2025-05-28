#!/usr/bin/env python3

import asyncio
import contextlib
import os
import sys

from io import StringIO
from typing import Callable, TextIO

from . import common
from .print import rstrip


def nth_rfind(string, char, n):
    pos = len(string)
    for _ in range(n):
        pos = string.rfind(char, 0, pos)
        if pos == -1:
            return -1
    return pos


@contextlib.asynccontextmanager
async def live_print(print_fun, get_live_text: Callable[[], str], transient: bool = True):
    with StdoutProxy() as stdout_proxy:
        num_newlines_printed = 0

        def clear_and_print(final: bool):
            nonlocal num_newlines_printed

            to_print = StringIO()

            # Move the cursor up by the number of newlines printed so far, then clear the screen from the cursor down
            if num_newlines_printed > 0:
                to_print.write(f"\033[{num_newlines_printed}F")
            to_print.write("\r\033[J")

            if final and transient:
                to_print.write(stdout_proxy.getvalue())
                print_fun(to_print.getvalue(), end="", flush=True)
                return

            term_height = os.get_terminal_size().lines

            text = get_live_text()
            if not stdout_proxy.empty:
                text = f"{text}\n\n{stdout_proxy.getvalue().rstrip()}"

            # Print the last term_height - 1 lines of the history to avoid terminal problems upon clearing again.
            # However, if we're the final print, we no longer need to clear, so we should print all lines.
            if not final:
                split_idx = nth_rfind(text, "\n", term_height)
                if split_idx != -1:
                    text = text[split_idx + 1 :]

            to_print.write(text)

            print_fun(to_print.getvalue(), end="", flush=True, file=stdout_proxy.original_stdout)

            num_newlines_printed = text.count("\n")

        async def live_print_task():
            try:
                while True:
                    clear_and_print(final=False)
                    await asyncio.sleep(1.0 / common.SPINNER_FPS)
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(live_print_task())
        try:
            yield task
        finally:
            task.cancel()
            try:
                await task
            finally:
                clear_and_print(final=True)


class StdoutProxy:
    def __init__(self):
        self._buffer: StringIO = StringIO()

        self._stdout = sys.stdout
        self._stderr = sys.stderr

        self._output: TextIO = sys.stdout

    def __enter__(self) -> "StdoutProxy":
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args: object) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def write(self, data: str) -> int:
        self._buffer.write(data)
        return len(data)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        return self._buffer.getvalue()

    @property
    def empty(self) -> bool:
        return self._buffer.tell() == 0

    @property
    def original_stdout(self) -> TextIO | None:
        return self._stdout or sys.__stdout__

    # Attributes for compatibility with sys.__stdout__:

    def fileno(self) -> int:
        return self._output.fileno()

    def isatty(self) -> bool:
        stdout = self._stdout
        return stdout.isatty() if stdout is not None else False

    @property
    def encoding(self) -> str:
        return self._stdout.encoding

    @property
    def errors(self) -> str:
        return "strict"
