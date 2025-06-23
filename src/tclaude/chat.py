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
import json
import logging
import os
import signal
from contextlib import AsyncExitStack
from functools import partial
from itertools import chain

import aiohttp
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output

from . import commands, common
from .common import History, is_valid_metadata
from .config import TClaudeArgs, load_system_prompt
from .json import JSON, get_or
from .live_print import live_print
from .mcp import setup_mcp
from .print import history_to_string
from .prompt import (
    Response,
    file_metadata_to_content,
    stream_response,
)
from .session import ChatSession, deduce_session_name
from .spinner import spinner
from .task_context import TaskAsyncContextManager
from .terminal_prompt import terminal_prompt
from .token_counter import TokenCounter
from .tool_use import get_python_tools

logger = logging.getLogger(__package__)


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


async def gather_file_uploads(tasks: list[asyncio.Task[dict[str, JSON]]]) -> list[dict[str, JSON]]:
    """
    Wait for all file upload tasks to complete and return the results.
    """
    results: list[dict[str, JSON]] = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            result["_input_pending"] = True  # Mark this file as pending being input in the next user message
            results.append(result)
        except aiohttp.ClientError as e:
            logger.exception(f"Failed to upload file: {e}")
        except asyncio.CancelledError:
            logger.exception("File upload cancelled.")
        except Exception as e:
            logger.exception(f"Error during file upload: {e}")

    return results


async def single_prompt(args: TClaudeArgs, config: dict[str, JSON], history: History, user_input: str, print_text_only: bool):
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """

    system_prompt = load_system_prompt(args.role) if args.role else None
    session = ChatSession(
        history=history,
        model=args.model,
        system_prompt=system_prompt,
        role=os.path.splitext(os.path.basename(args.role))[0] if args.role and system_prompt else None,
        name=deduce_session_name(args.session) if args.session else None,
    )

    async with AsyncExitStack() as stack:
        client_session: aiohttp.ClientSession = await stack.enter_async_context(aiohttp.ClientSession())

        async with asyncio.TaskGroup() as tg:
            if session.uploaded_files:
                _ = tg.create_task(session.verify_file_uploads(client_session))
            file_metadata = [tg.create_task(session.upload_file(client_session, f)) for f in args.file]
            mcp = await stack.enter_async_context(setup_mcp(client_session, config))

        user_content: list[JSON] = [{"type": "text", "text": user_input}]
        user_content.extend(chain.from_iterable(file_metadata_to_content(m.result()) for m in file_metadata if m))
        history.append({"role": "user", "content": user_content})
        response_idx = len(history)

        available_tools, tool_definitions = get_python_tools()

        mcp_tools, mcp_tool_definitions = mcp.get_tools()
        available_tools.update(mcp_tools)
        tool_definitions.extend(mcp_tool_definitions)

        while True:
            response = await stream_response(
                session=client_session,
                model=session.model,
                history=history,
                max_tokens=args.max_tokens,
                enable_web_search=not args.no_web_search,  # Web search is enabled by default
                enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
                external_tools_available=available_tools,
                external_tool_definitions=tool_definitions,
                mcp_remote_servers=await mcp.get_remote_server_descs(client_session),
                system_prompt=system_prompt,
                enable_thinking=args.thinking,
                thinking_budget=args.thinking_budget,
            )

            history.extend(response.messages)
            if not response.call_again:
                break

    try:
        print(await history_to_string(history[response_idx:], pretty=False, text_only=print_text_only), end="", flush=True)
    except BrokenPipeError:
        logger.error("Broken pipe. Response could not passed on to the next command in the pipeline.")


async def chat(args: TClaudeArgs, config: dict[str, JSON], history: History, user_input: str):
    """
    Main function to get user input, and print Anthropic's response.
    """

    system_prompt = load_system_prompt(args.role) if args.role else None
    session = ChatSession(
        history=history,
        model=args.model,
        system_prompt=system_prompt,
        role=os.path.splitext(os.path.basename(args.role))[0] if args.role and system_prompt else None,
        name=deduce_session_name(args.session) if args.session else None,
    )

    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(aiohttp.ClientSession())

        file_upload_verification_task = asyncio.create_task(session.verify_file_uploads(client)) if session.uploaded_files else None
        file_upload_tasks = [asyncio.create_task(session.upload_file(client, f)) for f in args.file if f]

        mcp = setup_mcp(client, config)
        mcp_setup = None if mcp.empty else await stack.enter_async_context(TaskAsyncContextManager(mcp))

        input = create_input(always_prefer_tty=True)
        output = create_output()

        prompt_session: PromptSession[str] = PromptSession(input=input, output=output)
        for m in session.user_messages:
            prompt_session.history.append_string(m)

        if user_input:
            prompt_session.history.append_string(user_input)

        async def pretty_history_to_string(messages: History, skip_user_text: bool) -> str:
            return await history_to_string(
                messages, pretty=True, wrap_width=os.get_terminal_size().columns, skip_user_text=skip_user_text, uploaded_files=session.uploaded_files
            )

        # Print the current state of the response. Keep overwriting the same lines since the response is getting incrementally built.
        async def history_or_spinner(response: Response, final: bool):
            current_message = await pretty_history_to_string(response.messages, skip_user_text=True)
            if not final:
                current_message = current_message if current_message else f"{spinner()} "
            return current_message

        def lprompt(prefix: str) -> str:
            return f"{prefix}{common.CHEVRON} "

        def rprompt(prefix: str) -> str:
            rprompt = f"{session.total_tokens.total_cost(session.model):.03f}   {common.friendly_model_name(session.model)} "
            if session.role:
                rprompt = f"󱜙 {session.role}  {rprompt}"

            if session.name is not None:
                rprompt = f" {session.name}  {rprompt}"
            elif session.is_autonaming:
                rprompt = f" auto-naming {spinner()}  {rprompt}"

            if file_upload_verification_task and not file_upload_verification_task.done():
                rprompt = f" verifying files {spinner()}  {rprompt}"

            num_uploaded_files = sum(1 for m in session.uploaded_files.values() if is_valid_metadata(m))
            num_uploading = sum(1 for t in file_upload_tasks if not t.done())

            num_total_files = num_uploaded_files + num_uploading

            if num_uploaded_files < num_total_files:
                rprompt = f" {num_uploaded_files}/{num_total_files} files {spinner()}  {rprompt}"
            elif num_uploaded_files > 0:
                rprompt = f" {num_uploaded_files} files  {rprompt}"

            if mcp_setup is not None and not mcp_setup.done():
                rprompt = f" setting up mcp {spinner()}  {rprompt}"

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
            session.cancel_autoname()

        _ = signal.signal(signal.SIGINT, interrupt_handler)

        available_tools, tool_definitions = get_python_tools()
        logger.debug(f"Available tools: {', '.join(available_tools.keys())}")

        response: Response | None = None
        while True:
            if is_user_turn:
                try:
                    user_input = await terminal_prompt(lprompt, rprompt, prompt_session, session, user_input)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    try:
                        cb = commands.get_callback(user_input.rstrip(), commands.get_commands(session.uploaded_files))
                        async with live_print(lambda _: f"Executing '{user_input}' {spinner()}"):
                            await cb()
                    except ValueError as e:
                        logger.error(f"Could not execute {user_input}: {e}")
                    except EOFError:
                        break
                    user_input = ""
                    continue

                if file_upload_verification_task:
                    async with live_print(lambda _: f"Verifying uploaded files {spinner()}"):
                        await file_upload_verification_task
                    file_upload_verification_task = None

                async with live_print(lambda _: f"[{sum(1 for t in file_upload_tasks if t.done())}/{len(file_upload_tasks)}] files uploaded {spinner()}"):
                    _ = await gather_file_uploads(file_upload_tasks)
                file_upload_tasks.clear()

                files_to_input = [m for m in session.uploaded_files.values() if get_or(m, "_input_pending", False)]
                for metadata in session.uploaded_files.values():
                    _ = metadata.pop("_input_pending", None)

                user_content: list[JSON] = [{"type": "text", "text": user_input}]
                user_content.extend(chain.from_iterable(file_metadata_to_content(m) for m in files_to_input if m))
                user_input = ""

                session.history.append({"role": "user", "content": user_content})

                # This includes things like file uploads, but *not* the user input text itself, which is already printed in the prompt.
                user_history_string = await pretty_history_to_string(session.history[-1:], skip_user_text=True)
                if user_history_string:
                    print(user_history_string, end="\n\n")

            if mcp_setup is not None:
                mcp = await mcp_setup
                mcp_setup = None

                mcp_tools, mcp_tool_definitions = mcp.get_tools()
                available_tools.update(mcp_tools)
                tool_definitions.extend(mcp_tool_definitions)

            container = common.get_latest_container(session.history)
            write_cache = should_cache(response.tokens, args.model) if response is not None else False

            if args.verbose:
                if container is not None:
                    logger.info(f"Reusing code execution container `{container.id}`")

                logger.info(f"write_cache={write_cache}")

            partial_response: Response = Response(messages=[], tokens=TokenCounter(), call_again=False)

            try:
                async with live_print(partial(history_or_spinner, partial_response), transient=False):
                    stream_task = asyncio.create_task(
                        stream_response(
                            session=client,
                            model=args.model,
                            history=session.history,
                            max_tokens=args.max_tokens,
                            enable_web_search=not args.no_web_search,  # Web search is enabled by default
                            enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
                            external_tools_available=available_tools,
                            external_tool_definitions=tool_definitions,
                            mcp_remote_servers=await mcp.get_remote_server_descs(client),
                            system_prompt=session.system_prompt,
                            enable_thinking=args.thinking,
                            thinking_budget=args.thinking_budget,
                            write_cache=write_cache,
                            on_response_update=lambda r: partial_response.__setattr__("messages", r.messages),
                        )
                    )

                    response = await stream_task

                    is_user_turn = not response.call_again
            except (aiohttp.ClientError, asyncio.CancelledError) as e:
                if is_user_turn:
                    _ = session.history.pop()
                is_user_turn = True

                print("\n")
                if isinstance(e, asyncio.CancelledError):
                    logger.error("Response cancelled.\n")
                elif isinstance(e, aiohttp.ClientResponseError):
                    logger.exception(f"Error {e.status}: {e.message}\n")
                else:
                    logger.exception(f"Unexpected error: {e}\n")

                continue
            finally:
                stream_task = None

            session.history.extend(response.messages)
            session.total_tokens += response.tokens

            session.uploaded_files.update(common.get_uploaded_files(response.messages))

            print("\n")
            if args.verbose:
                response.tokens.print_tokens()
                response.tokens.print_cost(args.model)

            # If we're beginning the next user turn, let us spawn a few background tasks to finish processing responses received so far.
            if is_user_turn:
                # Get metadata for new uploaded files for which we don't yet have valid metadata
                if any(not is_valid_metadata(m) for m in session.uploaded_files.values()):
                    assert file_upload_verification_task is None, "File upload verification task should not be running when we have new uploaded files"
                    file_upload_verification_task = asyncio.create_task(session.verify_file_uploads(client))

                # Auto-name the session if it is not already named
                if session.name is None and not session.is_autonaming:
                    session.start_autoname_task(client)

        print()

    # If we received at least one response
    if response is not None:
        # Cancelling the autoname task while it's running will fall back to using the current date and time as the session name.
        # If the task was already completed, we just use the name it set.
        await session.cancel_autoname_with_date_fallback()

        if session.name:
            session_path: str = session.name
            if not session_path.lower().endswith(".json"):
                session_path += ".json"

            if not os.path.isfile(session_path):
                session_path = os.path.join(args.sessions_dir, session_path)

            with open(session_path, "w") as f:
                json.dump(session.history, f, indent=2)

            logger.info(f"[✓] Saved session to {session_path}")

    if args.verbose:
        session.total_tokens.print_tokens()
        session.total_tokens.print_cost(args.model)
