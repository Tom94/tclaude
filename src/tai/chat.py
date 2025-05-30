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
import datetime
import json
import os
import signal

import aiohttp
from prompt_toolkit import PromptSession

from . import common, files
from .common import History, TaiArgs, perror, pinfo, pplain, psuccess
from .json import JSON, get, get_or, get_or_default
from .live_print import live_print
from .print import history_to_string
from .prompt import Response, TokenCounter, stream_response
from .spinner import spinner
from .terminal_prompt import terminal_prompt


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


async def async_chat(session: aiohttp.ClientSession, args: TaiArgs, history: History, user_input: str):
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    # Read system prompt from file if provided
    system_prompt = None
    if args.role:
        system_prompt = common.load_system_prompt(args.role)

    # Deduce session name from the session file if provided
    session_name = None
    if args.session:
        session_name = os.path.basename(args.session)

        # If the session is a json file, the session name is the file name without the extension
        stem, ext = os.path.splitext(session_name)
        if ext.lower() == ".json":
            session_name = stem

    initial_history_length = len(history)

    prompt_session: PromptSession[str] = PromptSession()

    for message in history:
        if get(message, "role", str) == "user":
            content_blocks = get_or_default(message, "content", list[JSON])
            if not content_blocks:
                continue

            text = get_or(content_blocks[0], "text", "")
            if text:
                prompt_session.history.append_string(text)

    if user_input:
        prompt_session.history.append_string(user_input)

    total_tokens = TokenCounter()

    # Initially, don't cache anything. The system prompt is always cached.
    write_cache = False

    # Print the current state of the response. Keep overwriting the same lines since the response is getting incrementally built.
    def history_or_spinner(messages: History):
        current_message = history_to_string(messages, pretty=True, wrap_width=os.get_terminal_size().columns)
        return current_message if current_message else f"{spinner()} "

    autoname_task: asyncio.Task[Response] | None = None

    def lprompt(prefix: str) -> str:
        return f"{prefix}{common.prompt_style(common.CHEVRON)} "

    def rprompt(prefix: str) -> str:
        rprompt = f"{total_tokens.total_cost(args.model):.03f}   {common.friendly_model_name(args.model)} "
        if args.role:
            prompt_role = os.path.splitext(os.path.basename(args.role))[0]
            rprompt = f"󱜙 {prompt_role}  {rprompt}"

        if session_name is not None:
            rprompt = f" {session_name}  {rprompt}"
        elif autoname_task is not None:
            rprompt = f" auto-naming {spinner()}  {rprompt}"

        return f"{prefix}{rprompt}"

    stream_task: asyncio.Task[Response] | None = None
    is_user_turn = True

    # Our repl session is meant to resemble a shell, hence we don't want Ctrl-C to exit but rather cancel the current response, which
    # roughly equates to pressing Ctrl-C in a shell to stop the current command.
    def interrupt_handler(_signum: int, _frame: object):
        if stream_task and not stream_task.done():
            _ = stream_task.cancel()
            return

        # If there's no conversation to cancel, the user likely wants to cancel the autonaming task.
        if autoname_task and not autoname_task.done():
            _ = autoname_task.cancel()
            return

    _ = signal.signal(signal.SIGINT, interrupt_handler)

    while True:
        if is_user_turn:
            if not user_input:
                try:
                    user_input = await terminal_prompt(lprompt, rprompt, prompt_session)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue

            history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
            user_input = ""
        else:
            # Either, the response was paused before (stop_reason == "pause_turn") or we are providing tool results (stop_reason == "tool_use").
            pass

        container = common.get_latest_container(history)
        if container is not None:
            pinfo(f"Reusing code execution container `{container.id}`", end="\n\n")

        partial: Response = Response(messages=[], tokens=TokenCounter(), call_again=False)
        try:
            async with live_print(lambda: history_or_spinner(partial.messages), transient=False):
                stream_task = asyncio.create_task(
                    stream_response(
                        session=session,
                        model=args.model,
                        history=history,
                        max_tokens=args.max_tokens,
                        enable_web_search=not args.no_web_search,  # Web search is enabled by default
                        enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
                        system_prompt=system_prompt,
                        enable_thinking=args.thinking,
                        thinking_budget=args.thinking_budget,
                        write_cache=write_cache,
                        on_response_update=lambda r: partial.__setattr__("messages", r.messages),
                    )
                )

                response = await stream_task

                is_user_turn = not response.call_again
        except (aiohttp.ClientError, asyncio.CancelledError) as e:
            if is_user_turn:
                _ = history.pop()
            is_user_turn = True

            pplain("\n")
            if isinstance(e, asyncio.CancelledError):
                perror("Response cancelled.\n")
            else:
                perror(f"Unexpected error: {e}. Please try again.\n")

            continue
        finally:
            stream_task = None

        history.extend(response.messages)
        total_tokens += response.tokens

        # Automatically determine whether we should put a cache breakpoint into the next prompt
        write_cache = should_cache(response.tokens, args.model)

        pplain("\n")
        if args.verbose:
            response.tokens.print_tokens()
            response.tokens.print_cost(args.model)
            if write_cache:
                pinfo("Next prompt will be cached.\n")

        # Start a background task to auto-name the session if it is not already named
        if session_name is None:
            if autoname_task is None and is_user_turn:
                autoname_prompt = (
                    "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you."
                )

                autoname_history = history.copy() + [{"role": "user", "content": [{"type": "text", "text": autoname_prompt}]}]
                autoname_task = asyncio.create_task(
                    stream_response(
                        session=session,
                        model=args.model,
                        history=autoname_history,
                        max_tokens=30,
                        enable_web_search=False,
                        system_prompt=system_prompt,
                        enable_thinking=False,
                    )
                )

                def handle_autoname_result(autoname_task: asyncio.Task[Response]):
                    nonlocal total_tokens, session_name

                    try:
                        response = autoname_task.result()
                        total_tokens += response.tokens
                        session_name = history_to_string(response.messages, pretty=False)
                    except (aiohttp.ClientError, asyncio.CancelledError) as e:
                        if isinstance(e, asyncio.CancelledError):
                            perror("Auto-naming cancelled. Using timestamp.")
                        else:
                            perror(f"Error auto-naming session: {e}.")
                        session_name = datetime.datetime.now().strftime("%H-%M-%S")

                    session_name = session_name.strip().lower()
                    session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
                    session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

                    date = datetime.datetime.now().strftime("%Y-%m-%d")
                    session_name = f"{date}-{session_name}"
                    psuccess(f"Session named {session_name}")

                autoname_task.add_done_callback(handle_autoname_result)

    pplain()

    # If we submitted a user prompt and received a response (at least 2 messages), save the session.
    if len(history) - initial_history_length >= 2:
        # To obtain the path to save to, we follow these rules:
        # 1. If no session name is provided but an autoname task is running, wait for that.
        # 2. If the path does not end with .json, we append .json.
        # 3. If the path does not exist, we prepend the sessions directory.
        # 4. We write the session file regardless of whether the final path exists or not.
        if autoname_task:
            try:
                if not autoname_task.done():
                    _ = autoname_task.cancel()
                await autoname_task
            except asyncio.CancelledError:
                pass

        if session_name:
            session_path: str = session_name
            if not session_path.lower().endswith(".json"):
                session_path += ".json"

            if not os.path.isfile(session_path):
                session_path = os.path.join(args.sessions_dir, session_path)

            with open(session_path, "w") as f:
                json.dump(history, f, indent=2)

            psuccess(f"Saved session to {session_path}")

    if args.verbose:
        total_tokens.print_tokens()
        total_tokens.print_cost(args.model)


async def async_chat_wrapper(args: TaiArgs, history: History, user_input: str):
    async with aiohttp.ClientSession() as session:
        await async_chat(session, args, history, user_input)


def chat(args: TaiArgs, history: History, user_input: str):
    asyncio.run(async_chat_wrapper(args, history, user_input))
