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
from itertools import chain

import aiohttp
from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output

from . import common
from .common import History, TaiArgs
from .json import JSON
from .live_print import live_print
from .print import history_to_string
from .prompt import (
    Response,
    TokenCounter,
    file_metadata_to_content,
    stream_response,
    upload_file,
    verify_file_uploads,
)
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


def deduce_session_name(session_file: str | None) -> str | None:
    """
    Deduce the session name from the command line arguments. If a session file is provided, use its basename. Otherwise, return None.
    """
    if session_file:
        session_name = os.path.basename(session_file)
        stem, ext = os.path.splitext(session_name)
        if ext.lower() == ".json":
            return stem
        return session_name

    return None


async def gather_file_uploads(tasks: list[asyncio.Task[JSON]]) -> list[JSON]:
    """
    Wait for all file upload tasks to complete and return the results.
    """
    results: JSON = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except aiohttp.ClientError as e:
            logger.opt(exception=e).error(f"Failed to upload file: {e}")
        except asyncio.CancelledError as e:
            logger.opt(exception=e).error("File upload cancelled.")
        except Exception as e:
            logger.opt(exception=e).error(f"Error during file upload: {e}")

    return results


async def async_chat(session: aiohttp.ClientSession, args: TaiArgs, history: History, user_input: str):
    """
    Main function to get user input, and print Anthropic's response.
    """
    system_prompt = common.load_system_prompt(args.role) if args.role else None
    session_name = deduce_session_name(args.session)

    user_messages, uploaded_files = common.process_user_blocks(history)
    file_upload_verification_task = asyncio.create_task(verify_file_uploads(session, history, uploaded_files)) if uploaded_files else None
    file_upload_tasks = [asyncio.create_task(upload_file(session, f, uploaded_files)) for f in args.file if f]

    input = create_input(always_prefer_tty=True)
    output = create_output()

    prompt_session: PromptSession[str] = PromptSession(input=input, output=output)
    for m in user_messages:
        prompt_session.history.append_string(m)

    if user_input:
        prompt_session.history.append_string(user_input)

    total_tokens = TokenCounter()

    def pretty_history_to_string(messages: History, skip_user_text: bool) -> str:
        return history_to_string(messages, pretty=True, wrap_width=os.get_terminal_size().columns, skip_user_text=skip_user_text, uploaded_files=uploaded_files)

    # Print the current state of the response. Keep overwriting the same lines since the response is getting incrementally built.
    def history_or_spinner(messages: History):
        current_message = pretty_history_to_string(messages, skip_user_text=True)
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

        if file_upload_verification_task and not file_upload_verification_task.done():
            rprompt = f" verifying files {spinner()}  {rprompt}"

        num_uploaded_files = sum(1 for m in uploaded_files.values() if m is not None)
        num_uploading = sum(1 for t in file_upload_tasks if not t.done())

        num_total_files = num_uploaded_files + num_uploading

        if num_uploaded_files < num_total_files:
            rprompt = f" {num_uploaded_files}/{num_total_files} files {spinner()}  {rprompt}"
        elif num_uploaded_files > 0:
            rprompt = f" {num_uploaded_files} files  {rprompt}"

        return f"{prefix}{rprompt}"

    stream_task: asyncio.Task[Response] | None = None

    # Not every request is going to be a user turn (where the user inputs text into a prompt). For example, if the response was paused
    # before (stop_reason == "pause_turn") or we are providing tool results (stop_reason == "tool_use"), it isn't the user's turn, but we
    # still need to make a request to the model to continue the conversation. This is what this variable is for.
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

    response: Response | None = None
    while True:
        if is_user_turn:
            try:
                user_input = await terminal_prompt(lprompt, rprompt, prompt_session, user_input)
            except EOFError:
                break
            except KeyboardInterrupt:
                continue
            if not user_input:
                continue

            if file_upload_verification_task:
                async with live_print(lambda: f"Verifying uploaded files {spinner()}"):
                    await file_upload_verification_task

            async with live_print(lambda: f"[{sum(1 for t in file_upload_tasks if t.done())}/{len(file_upload_tasks)}] files uploaded {spinner()}"):
                file_metadata = await gather_file_uploads(file_upload_tasks)
            file_upload_tasks.clear()

            user_content: list[JSON] = [{"type": "text", "text": user_input}]
            user_content.extend(chain.from_iterable(file_metadata_to_content(m) for m in file_metadata if m))
            user_input = ""

            history.append({"role": "user", "content": user_content})

            # This includes things like file uploads, but *not* the user input text itself, which is already printed in the prompt.
            user_history_string = pretty_history_to_string(history[-1:], skip_user_text=True)
            if user_history_string:
                print(user_history_string, end="\n\n")

        container = common.get_latest_container(history)
        write_cache = should_cache(response.tokens, args.model) if response is not None else False

        if args.verbose:
            if container is not None:
                logger.info(f"Reusing code execution container `{container.id}`")

            logger.info(f"write_cache={write_cache}")

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

            print("\n")
            if isinstance(e, asyncio.CancelledError):
                logger.error("Response cancelled.\n")
            else:
                logger.opt(exception=e).error(f"Unexpected error: {e}. Please try again.\n")

            continue
        finally:
            stream_task = None

        history.extend(response.messages)
        total_tokens += response.tokens

        print("\n")
        if args.verbose:
            response.tokens.print_tokens()
            response.tokens.print_cost(args.model)

        # Start a background task to auto-name the session if it is not already named
        if is_user_turn and session_name is None:
            if autoname_task is None and is_user_turn:
                autoname_prompt = "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you."

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
                            logger.error("Auto-naming cancelled. Using timestamp.")
                        else:
                            logger.opt(exception=e).error(f"Error auto-naming session: {e}.")
                        session_name = datetime.datetime.now().strftime("%H-%M-%S")

                    session_name = session_name.strip().lower()
                    session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
                    session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

                    date = datetime.datetime.now().strftime("%Y-%m-%d")
                    session_name = f"{date}-{session_name}"
                    logger.success(f"Session named {session_name}")

                autoname_task.add_done_callback(handle_autoname_result)

    print()

    # If we received at least one response
    if response is not None:
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
            except aiohttp.ClientError:
                pass
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

            logger.success(f"Saved session to {session_path}")

    if args.verbose:
        total_tokens.print_tokens()
        total_tokens.print_cost(args.model)


async def async_chat_wrapper(args: TaiArgs, history: History, user_input: str):
    async with aiohttp.ClientSession() as session:
        await async_chat(session, args, history, user_input)


def chat(args: TaiArgs, history: History, user_input: str):
    asyncio.run(async_chat_wrapper(args, history, user_input))
