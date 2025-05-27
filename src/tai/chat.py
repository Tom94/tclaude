#!/usr/bin/env python3

# tai -- Terminal AI
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
import contextlib
import datetime
import json
import os

from prompt_toolkit import PromptSession, print_formatted_text, ANSI
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from typing import Callable

from . import common
from .print import history_to_string
from .prompt import stream_response, TokenCounter


@contextlib.asynccontextmanager
async def live_print(get_to_print: Callable[[], str], transient: bool = True):
    import asyncio
    from io import StringIO

    num_newlines_printed = 0

    def clear_and_print(final: bool):
        nonlocal num_newlines_printed

        to_print = StringIO()

        # Move the cursor up by the number of newlines printed so far, then clear the screen from the cursor down
        if num_newlines_printed > 0:
            to_print.write(f"\033[{num_newlines_printed}F")
        to_print.write("\r\033[J")

        if final and transient:
            print(to_print.getvalue(), end="", flush=True)
            return

        term_height = os.get_terminal_size().lines
        lines = get_to_print().split("\n")

        # Print the last term_height - 1 lines of the history to avoid terminal problems upon clearing again.
        # However, if we're the final print, we no longer need to clear, so we should print all lines.
        if not final:
            if len(lines) >= term_height:
                lines = lines[-(term_height - 1) :]

        to_print.write("\n".join(lines))

        print(to_print.getvalue(), end="", flush=True)
        num_newlines_printed = len(lines) - 1

    async def live_print_task():
        while True:
            clear_and_print(final=False)
            await asyncio.sleep(1.0 / common.SPINNER_FPS)

    task = asyncio.create_task(live_print_task())
    try:
        yield task
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            clear_and_print(final=True)


def create_prompt_key_bindings():
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        event.app.current_buffer.validate_and_handle()

    @bindings.add("\\", "enter")
    def _(event):
        event.app.current_buffer.newline()

    @bindings.add("c-p")
    def _(event):
        event.app.current_buffer.history_backward()

    @bindings.add("c-n")
    def _(event):
        event.app.current_buffer.history_forward()

    return bindings


async def user_prompt(
    lprompt: Callable[[], str],
    rprompt: Callable[[], str],
    prompt_session: PromptSession,
    key_bindings: KeyBindings,
) -> str:
    print(common.ansi("1G"), end="")  # Ensure we don't have stray remaining characters from user typing before the prompt was ready.
    user_input = ""
    while not user_input:
        with patch_stdout():

            async def animate_prompts():
                try:
                    while True:
                        await asyncio.sleep(1 / common.SPINNER_FPS)
                        prompt_session.message = ANSI(common.prompt_style(lprompt()))
                        prompt_session.rprompt = ANSI(common.prompt_style(rprompt()))
                except asyncio.CancelledError:
                    pass

            animate_task = asyncio.create_task(animate_prompts())
            try:
                user_input = await prompt_session.prompt_async(
                    ANSI(common.prompt_style(lprompt())),
                    rprompt=ANSI(common.prompt_style(rprompt())),
                    vi_mode=True,
                    cursor=ModalCursorShapeConfig(),
                    multiline=True,
                    wrap_lines=True,
                    placeholder=ANSI(common.gray_style(common.HELP_TEXT)),
                    key_bindings=key_bindings,
                    refresh_interval=1 / common.SPINNER_FPS,
                )
            finally:
                animate_task.cancel()
                await animate_task

            user_input = user_input.strip()

    return user_input


def should_cache(tokens: TokenCounter, model: str) -> bool:
    """
    We heuristically set a new cache breakpoint when our next prompt (if short ~0 tokens) causes the cost of input to be larger
    than that of cache reads.
    TODO: If we just finished a web search, apparently something messy happens to the cache... should investigate
    """
    tokens_if_short_follow_up = TokenCounter(
        cache_creation_input_tokens=0,
        cache_read_input_tokens=tokens.cache_read + tokens.cache_creation,
        input_tokens=tokens.input + tokens.output,
        output_tokens=0,
    )
    _, cache_read_cost, input_cost, _ = tokens_if_short_follow_up.cost(model)
    return cache_read_cost < input_cost


async def async_chat(args, history: list[dict], user_input: str):
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    # Read system prompt from file if provided
    system_prompt = None
    if args.role:
        system_prompt = common.load_system_prompt(args.role)

    session_name = None
    if args.session:
        session_name = os.path.basename(args.session)

        # If the session is a json file, the session name is the file name without the extension
        stem, ext = os.path.splitext(session_name)
        if ext.lower() == ".json":
            session_name = stem

    initial_history_length = len(history)

    prompt_key_bindings = create_prompt_key_bindings()
    prompt_session = PromptSession()

    for message in history:
        if message.get("role") == "user":
            text = message.get("content", [{}])[0].get("text", "")
            if text:
                prompt_session.history.append_string(text)

    total_tokens = TokenCounter()

    # Initially, don't cache anything. The system prompt is always cached.
    write_cache = False

    # Print the current state of the response. Keep overwriting the same lines since the response is getting incrementally built.
    def history_or_spinner(messages: list[dict]):
        current_message = history_to_string(messages, pretty=True, wrap_width=os.get_terminal_size().columns)
        return current_message if current_message else f"{common.spinner()} "

    autoname_task = None

    def lprompt() -> str:
        return f"{common.CHEVRON} "

    def rprompt() -> str:
        rprompt = f"{total_tokens.total_cost(args.model):.03f}   {common.friendly_model_name(args.model)} "
        if args.role:
            prompt_role = os.path.splitext(os.path.basename(args.role))[0]
            rprompt = f"󱜙 {prompt_role}  {rprompt}"

        if session_name is not None:
            rprompt = f" {session_name}  {rprompt}"
        elif autoname_task is not None:
            rprompt = f" auto-naming {common.spinner()}  {rprompt}"

        return rprompt

    is_user_turn = True
    while True:
        if is_user_turn:
            if not user_input:
                try:
                    user_input = await user_prompt(lprompt, rprompt, prompt_session, prompt_key_bindings)
                except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                    break

            history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
            user_input = ""
        else:
            # Either, the response was paused before (stop_reason == "pause_turn") or we are providing tool results (stop_reason == "tool_use").
            pass

        partial = {"messages": []}
        try:
            async with live_print(lambda: history_or_spinner(partial["messages"]), transient=False):
                # The response is already printed during streaming, so we don't need to print it again
                messages, tokens, call_again = await stream_response(
                    model=args.model,
                    history=history,
                    max_tokens=args.max_tokens,
                    enable_web_search=not args.no_web_search,  # Web search is enabled by default
                    enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
                    system_prompt=system_prompt,
                    enable_thinking=args.thinking,
                    thinking_budget=args.thinking_budget,
                    write_cache=write_cache,
                    on_response_update=lambda m, _: partial.update({"messages": m}),
                )

                is_user_turn = not call_again
        except (KeyboardInterrupt, asyncio.CancelledError, Exception) as e:
            if is_user_turn:
                history.pop()
            is_user_turn = True

            if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                print("\n\nResponse cancelled.\n")
            else:
                print(f"\n\nUnexpected error: {e}. Please try again.\n")

            continue

        history.extend(messages)
        total_tokens += tokens

        # Automatically determine whether we should put a cache breakpoint into the next prompt
        write_cache = should_cache(tokens, args.model)

        print("\n")
        if args.verbose:
            tokens.print_tokens()
            tokens.print_cost(args.model)
            if write_cache:
                print("Next prompt will be cached.\n")

        # Start a background task to auto-name the session if it is not already named
        if session_name is None:
            if autoname_task is None and is_user_turn:
                autoname_prompt = (
                    "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you."
                )

                autoname_history = history.copy() + [{"role": "user", "content": [{"type": "text", "text": autoname_prompt}]}]
                autoname_task = asyncio.create_task(
                    stream_response(
                        model=args.model,
                        history=autoname_history,
                        max_tokens=30,
                        enable_web_search=False,
                        system_prompt=system_prompt,
                        enable_thinking=False,
                    )
                )

                def handle_autoname_result(autoname_task: asyncio.Task):
                    nonlocal total_tokens, session_name

                    try:
                        messages, tokens, _ = autoname_task.result()
                        total_tokens += tokens
                        session_name = history_to_string(messages, pretty=False)
                    except (KeyboardInterrupt, asyncio.CancelledError, Exception) as e:
                        if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                            print("Auto-naming cancelled.")
                        else:
                            print(f"Error auto-naming session: {e}.")
                        print("Falling back to time stamp.\n")
                        session_name = datetime.datetime.now().strftime("%H-%M-%S")

                    session_name = session_name.strip().lower()
                    session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
                    session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

                    date = datetime.datetime.now().strftime("%Y-%m-%d")
                    session_name = f"{date}-{session_name}"

                autoname_task.add_done_callback(handle_autoname_result)

    print()

    # If we submitted a user prompt and received a response (at least 2 messages), save the session.
    if len(history) - initial_history_length >= 2:
        # To obtain the path to save to, we follow these rules:
        # 1. If no session name is provided but an autoname task is running, wait for that.
        # 2. If the path does not end with .json, we append .json.
        # 3. If the path does not exist, we prepend the sessions directory.
        # 4. We write the session file regardless of whether the final path exists or not.
        if autoname_task and not autoname_task.done():
            async with live_print(lambda: f"Auto-naming session {common.spinner()} "):
                await autoname_task

        if session_name:
            session_path = session_name
            if not session_path.lower().endswith(".json"):
                session_path += ".json"

            if not os.path.isfile(session_path):
                session_path = os.path.join(args.sessions_dir, session_path)

            with open(session_path, "w") as f:
                json.dump(history, f, indent=2)

            print(f"✓ Saved session to {session_path}")

    if args.verbose:
        total_tokens.print_tokens()
        total_tokens.print_cost(args.model)


def chat(args, history: list[dict], user_input: str):
    asyncio.run(async_chat(args, history, user_input))
