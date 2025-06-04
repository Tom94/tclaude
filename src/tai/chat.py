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

import aiofiles.os
import aiohttp
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output

from . import common, files
from .common import History, TaiArgs, perror, pinfo, pplain, psuccess, pwarning
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


def process_user_blocks(history: History) -> tuple[list[str], dict[str, JSON]]:
    """
    Process the initial history to extract user messages and uploaded files.
    Returns a tuple of:
    - A list of user messages as strings.
    - A dictionary of uploaded files with their file IDs as keys and metadata as values.
    """
    user_messages: list[str] = []
    uploaded_files: dict[str, JSON] = {}

    for message in history:
        if get(message, "role", str) != "user":
            continue

        for content_block in get_or_default(message, "content", list[JSON]):
            match content_block:
                case {"type": "text", "text": str(text)}:
                    user_messages.append(text)
                case {"type": "container_upload", "file_id": str(file_id)} | {
                    "type": "document" | "image",
                    "source": {"file_id": str(file_id)},
                }:
                    uploaded_files[file_id] = {}
                case {"type": "tool_result"}:
                    pass
                case _:
                    pwarning(f"Unknown content block type in user message: {content_block}")

    return user_messages, uploaded_files


async def upload_file(session: aiohttp.ClientSession, file_path: str, uploaded_files: dict[str, JSON]) -> JSON:
    if not await aiofiles.os.path.isfile(file_path):
        perror(f"File {file_path} does not exist or is not a file.")
        return None

    result = await files.upload_file(session, file_path)
    file_id = get(result, "id", str)
    if file_id is not None:
        uploaded_files[file_id] = result
        return result

    perror(f"Failed to upload file {file_path}. No file ID returned.")
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
            perror(f"Failed to upload file: {e}")
        except asyncio.CancelledError:
            perror("File upload cancelled.")

    return results


def erase_invalid_file_content_blocks(history: History, uploaded_files: dict[str, JSON]) -> None:
    """
    Erase all content blocks in the history that have no entry in `uploaded_files`. This is useful when we want to remove file references
    from the history after verifying or processing them.
    """

    def is_valid_block(block: JSON) -> bool:
        match block:
            case {"type": "container_upload", "file_id": file_id}:
                return file_id in uploaded_files
            case {"type": "document" | "image", "source": {"file_id": file_id}}:
                return file_id in uploaded_files
            case _:  # Other block types, like text or tool use are always valid
                return True

    for message in history:
        if get(message, "role", str) != "user":
            continue

        content = get_or_default(message, "content", list[JSON])
        message["content"] = [block for block in content if is_valid_block(block)]


async def verify_file_uploads(session: aiohttp.ClientSession, history: History, uploaded_files: dict[str, JSON]):
    """
    Verifies the uploaded files by checking their metadata. This is useful to ensure that the files are still valid and have not been
    removed or corrupted. The updates the `uploaded_files` dictionary with the metadata of the uploaded files.
    """
    file_upload_verification_task = asyncio.gather(
        *(files.get_file_metadata(session, file_id) for file_id in uploaded_files.keys()), return_exceptions=True
    )

    metadata_list = await file_upload_verification_task
    for metadata in metadata_list:
        match metadata:
            case {"id": str(file_id)}:
                uploaded_files[file_id] = metadata
            case BaseException() as e:
                perror(f"Failed to verify file upload: {e}")
            case _:
                pwarning(f"Unexpected metadata format: {metadata}. Expected a JSON object with an 'id' field.")

    # Remove any files that were not found or had an error
    missing_files = [file_id for file_id, metadata in uploaded_files.items() if not metadata]
    for file_id in missing_files:
        pwarning(f"File ID `{file_id}` is missing. Please re-upload it.")
        del uploaded_files[file_id]

    erase_invalid_file_content_blocks(history, uploaded_files)


def file_metadata_to_content(metadata: JSON) -> list[JSON]:
    """
    Convert a file metadata JSON object to a list of content blocks that can be added to the history.
    """
    content: list[JSON] = []

    type = files.mime_type_to_content_block_type(get_or(metadata, "mime_type", ""))
    id = get(metadata, "id", str)
    if id is None:
        return content

    # Even if the type is invalid, the code execution tool might still be able to handle the file. Always put valid file IDs
    # into the code execution container.
    content.append({"type": "container_upload", "file_id": id})
    if type is None:
        return content

    info: dict[str, JSON] = {"type": type, "source": {"type": "file", "file_id": id}}
    if type == "document":
        info["context"] = "This document was uploaded by the user."
        info["citations"] = {"enabled": True}
        info["title"] = get_or(metadata, "filename", id)

    content.append(info)
    return content


async def build_user_content(
    user_input: str, file_upload_verification_task: asyncio.Task[None] | None, file_upload_tasks: list[asyncio.Task[JSON]]
) -> list[JSON]:
    """
    Build the user content to be sent to the model. This includes the user input text and any file uploads.
    """
    content: list[JSON] = [{"type": "text", "text": user_input}]

    # Ensure file uploads and verifications are done before querying the model.
    if file_upload_verification_task:
        async with live_print(lambda: f"Verifying uploaded files {spinner()}"):
            await file_upload_verification_task

    async with live_print(lambda: f"[{sum(1 for t in file_upload_tasks if t.done())}/{len(file_upload_tasks)}] files uploaded {spinner()}"):
        metadata = await gather_file_uploads(file_upload_tasks)
    file_upload_tasks.clear()

    content.extend(chain.from_iterable(file_metadata_to_content(m) for m in metadata if m))

    return content


async def async_chat(session: aiohttp.ClientSession, args: TaiArgs, history: History, user_input: str):
    """
    Main function to get user input, and print Anthropic's response.
    """
    system_prompt = common.load_system_prompt(args.role) if args.role else None
    session_name = deduce_session_name(args.session)

    user_messages, uploaded_files = process_user_blocks(history)
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
        return history_to_string(
            messages, pretty=True, wrap_width=os.get_terminal_size().columns, skip_user_text=skip_user_text, uploaded_files=uploaded_files
        )

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
            if not user_input:
                try:
                    user_input = await terminal_prompt(lprompt, rprompt, prompt_session)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue

            user_content = await build_user_content(user_input, file_upload_verification_task, file_upload_tasks)
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
                pinfo(f"Reusing code execution container `{container.id}`")

            pinfo(f"write_cache={write_cache}")

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

        pplain("\n")
        if args.verbose:
            response.tokens.print_tokens()
            response.tokens.print_cost(args.model)

        # Start a background task to auto-name the session if it is not already named
        if is_user_turn and session_name is None:
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
            except aiohttp.ClientError as e:
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

            psuccess(f"Saved session to {session_path}")

    if args.verbose:
        total_tokens.print_tokens()
        total_tokens.print_cost(args.model)


async def async_chat_wrapper(args: TaiArgs, history: History, user_input: str):
    async with aiohttp.ClientSession() as session:
        await async_chat(session, args, history, user_input)


def chat(args: TaiArgs, history: History, user_input: str):
    asyncio.run(async_chat_wrapper(args, history, user_input))
