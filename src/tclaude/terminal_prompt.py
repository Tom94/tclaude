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
from typing import Callable

from prompt_toolkit import ANSI, PromptSession
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.patch_stdout import patch_stdout

from . import common, logging
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
        if not event.app.current_buffer.text:
            # Hide placeholder text when the user presses enter.
            event.app.current_buffer.text = " "
        event.app.current_buffer.validate_and_handle()

    @bindings.add("\\", "enter")
    def _(event: KeyPressEvent):
        event.app.current_buffer.newline()

    @bindings.add("c-p")
    def _(event: KeyPressEvent):
        event.app.current_buffer.history_backward()

    @bindings.add("c-n")
    def _(event: KeyPressEvent):
        event.app.current_buffer.history_forward()

    return bindings


async def terminal_prompt(
    lprompt: Callable[[str], str],
    rprompt: Callable[[str], str],
    prompt_session: PromptSession[str],
    user_input: str = "",
) -> str:
    key_bindings = create_prompt_key_bindings()

    # Ensure we don't have stray remaining characters from user typing before the prompt was ready.
    print(common.ANSI_BEGINNING_OF_LINE, end="", flush=False)

    prefix = ""

    def update_prefix():
        nonlocal prefix
        if logging.did_print_since_prompt:
            prefix = "\n"
            logging.did_print_since_prompt = False

    update_prefix()

    async def animate_prompts():
        try:
            while True:
                await asyncio.sleep(1 / SPINNER_FPS)
                update_prefix()
                prompt_session.message = ANSI(common.prompt_style(lprompt(prefix)))
                prompt_session.rprompt = ANSI(common.prompt_style(rprompt(prefix)))
        except asyncio.CancelledError:
            pass

    animate_task = asyncio.create_task(animate_prompts())
    try:
        with patch_stdout(raw=True):
            user_input = await prompt_session.prompt_async(
                ANSI(common.prompt_style(lprompt(prefix))),
                rprompt=ANSI(common.prompt_style(rprompt(prefix))),
                vi_mode=True,
                cursor=ModalCursorShapeConfig(),
                multiline=True,
                wrap_lines=True,
                placeholder=ANSI(common.gray_style(common.HELP_TEXT)),
                key_bindings=key_bindings,
                refresh_interval=1 / SPINNER_FPS,
                handle_sigint=False,
                default=user_input,
                accept_default=user_input != "",
            )
    finally:
        _ = animate_task.cancel()
        await animate_task

    user_input = user_input.strip()
    return user_input
