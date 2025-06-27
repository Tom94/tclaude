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

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable
from typing import Callable, cast, override

from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    WordCompleter,
)
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout

from . import common, logging_config
from .commands import Command, CommandCallback, get_commands
from .common import FileMetadata
from .session import ChatSession
from .spinner import SPINNER_FPS


def create_prompt_key_bindings():
    bindings = KeyBindings()

    @bindings.add("c-d")
    def _(event: KeyPressEvent):
        if not event.app.current_buffer.text:
            event.app.current_buffer.text = " "
        event.app.exit(exception=EOFError, style="class:aborting")

    @bindings.add("c-c")
    def _(event: KeyPressEvent):
        if not event.app.current_buffer.text:
            event.app.current_buffer.text = " "
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add("enter")
    def _(event: KeyPressEvent):
        complete_state = event.app.current_buffer.complete_state
        if complete_state is not None and complete_state.current_completion is not None:
            event.app.current_buffer.apply_completion(complete_state.current_completion)
            event.app.current_buffer.insert_text(" ")
            return

        if not event.app.current_buffer.text:
            # Hide placeholder text when the user presses enter.
            event.app.current_buffer.text = " "
        event.app.current_buffer.validate_and_handle()

    @bindings.add("\\", "enter")
    def _(event: KeyPressEvent):
        event.app.current_buffer.newline()

    @bindings.add("c-p")
    def _(event: KeyPressEvent):
        if event.app.current_buffer.complete_state is not None:
            event.app.current_buffer.complete_previous()
            return

        event.app.current_buffer.history_backward()

    @bindings.add("c-n")
    def _(event: KeyPressEvent):
        if event.app.current_buffer.complete_state is not None:
            event.app.current_buffer.complete_next()
            return

        event.app.current_buffer.history_forward()

    return bindings


class CommandCompleter(Completer):
    def __init__(self, options: dict[str, Completer | CommandCallback], ignore_case: bool = True) -> None:
        self.options: dict[str, Completer | CommandCallback] = options
        self.ignore_case: bool = ignore_case

    @override
    def __repr__(self) -> str:
        return f"CommandCompleter({self.options!r}, ignore_case={self.ignore_case!r})"

    @classmethod
    def from_nested_dict(cls, data: dict[str, Command]) -> CommandCompleter:
        options: dict[str, Completer | CommandCallback] = {}
        for key, value in data.items():
            if isinstance(value, Completer):
                options[key] = value
            elif isinstance(value, dict):
                options[key] = cls.from_nested_dict(value)
            else:
                assert inspect.iscoroutinefunction(value)
                options[key] = cast(CommandCallback, value)

        return cls(options)

    @override
    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        # Split document.
        text = document.text_before_cursor.lstrip()
        stripped_len = len(document.text_before_cursor) - len(text)

        # If there is a space, check for the first term, and use a
        # subcompleter.
        if " " in text:
            terms = text.split(" ")
            first_term = terms[0] if terms else ""
            completer = self.options.get(first_term)

            # If we have a sub completer, use this for the completions.
            if isinstance(completer, Completer):
                remaining_text = text[len(first_term) :].lstrip()
                move_cursor = len(text) - len(remaining_text) + stripped_len

                new_document = Document(
                    remaining_text,
                    cursor_position=document.cursor_position - move_cursor,
                )

                yield from completer.get_completions(new_document, complete_event)

        # No space in the input: behave exactly like `WordCompleter`.
        else:
            completer = WordCompleter(list(self.options.keys()), ignore_case=self.ignore_case, sentence=True)
            yield from completer.get_completions(document, complete_event)


async def terminal_prompt(
    lprompt: Callable[[str], str],
    rprompt: Callable[[str], str],
    prompt_session: PromptSession[str],
    session: ChatSession,
    user_input: str = "",
) -> str:
    key_bindings = create_prompt_key_bindings()

    # Ensure we don't have stray remaining characters from user typing before the prompt was ready.
    print(common.ANSI_BEGINNING_OF_LINE, end="", flush=False)

    prefix = ""

    def update_prefix():
        nonlocal prefix
        if logging_config.did_print_since_prompt:
            prefix = "\n"
            logging_config.did_print_since_prompt = False

    update_prefix()

    async def animate_prompts():
        while True:
            await asyncio.sleep(1 / SPINNER_FPS)
            update_prefix()
            prompt_session.message = ANSI(common.prompt_style(lprompt(prefix)))
            prompt_session.rprompt = ANSI(common.prompt_style(rprompt(prefix)))

    def on_files_changed(files: dict[str, FileMetadata]):
        prompt_session.completer = CommandCompleter.from_nested_dict(get_commands(files))

    animate_task = asyncio.create_task(animate_prompts())
    try:
        async with session.on_files_changed.temp_subscribe(on_files_changed):
            with patch_stdout(raw=True):
                user_input = await prompt_session.prompt_async(
                    ANSI(common.prompt_style(lprompt(prefix))),
                    rprompt=ANSI(common.prompt_style(rprompt(prefix))),
                    vi_mode=True,
                    cursor=ModalCursorShapeConfig(),
                    multiline=True,
                    wrap_lines=True,
                    prompt_continuation=ANSI(f"{common.prompt_style(common.CHEVRON_CONTINUATION)} "),
                    placeholder=ANSI(common.gray_style(common.HELP_TEXT)),
                    key_bindings=key_bindings,
                    refresh_interval=1 / SPINNER_FPS,
                    handle_sigint=False,
                    default=user_input,
                    accept_default=user_input != "",
                    completer=CommandCompleter.from_nested_dict(get_commands(session.uploaded_files)),
                    complete_while_typing=Condition(lambda: prompt_session.app.current_buffer.text.startswith("/")),
                )
    finally:
        _ = animate_task.cancel()
        try:
            await animate_task
        except asyncio.CancelledError:
            pass

    user_input = user_input.strip()
    return user_input
