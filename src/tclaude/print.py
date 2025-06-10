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

import argparse
import json
import os
from dataclasses import dataclass
from io import StringIO
from itertools import groupby
from typing import cast

from humanize import naturalsize
from loguru import logger

from . import common, config, logging
from .common import History, escape, wrap_style
from .json import JSON, get, get_or, get_or_default
from .spinner import spinner


def rstrip(io: StringIO) -> StringIO:
    """
    Remove trailing newlines and spaces from the StringIO object.
    """
    pos = io.tell()
    while pos > 0:
        pos = io.seek(pos - 1)
        if not io.read(1).isspace():
            pos = io.seek(pos + 1)  # Move back to the last non-space character
            break

    _ = io.truncate()
    return io


def to_superscript(text: str | int) -> str:
    text = str(text)
    superscript_map = str.maketrans("0123456789+-=(),", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒")
    return text.translate(superscript_map)


def write_system_message(message: JSON, io: StringIO):
    _ = io.write("\n# System prompt\n")
    content = get_or_default(message, "content", list[JSON])
    for content_block in content:
        if get_or(content_block, "type", "") == "text":
            _ = io.write(f"{get_or(content_block, 'text', '<empty>')}\n")


def write_block(heading: str, block_text: str, io: StringIO, pretty: bool, color: str, wrap_width: int):
    _ = io.write(wrap_style(f"╭── {heading}\n", color, pretty=pretty))
    block_text = common.word_wrap(block_text, wrap_width - 2)
    for line in block_text.splitlines():
        _ = io.write(f"{wrap_style('│ ', color, pretty=pretty)}{line}\n")
    _ = io.write(wrap_style("╰─", color, pretty=pretty))


def write_call_block(heading: str, block_text: str, io: StringIO, pretty: bool, wrap_width: int):
    write_block(heading, block_text, io, pretty, color="0;35m", wrap_width=wrap_width)


def write_result_block(heading: str, block_text: str, io: StringIO, pretty: bool, wrap_width: int):
    write_block(heading, block_text, io, pretty, color="0;36m", wrap_width=wrap_width)


def write_error_block(heading: str, block_text: str, io: StringIO, pretty: bool, wrap_width: int):
    write_block(heading, block_text, io, pretty, color="0;91m", wrap_width=wrap_width)


def gather_tool_results(messages: History) -> dict[str, JSON]:
    """
    Find the tool result in the messages by tool ID.
    """
    result: dict[str, JSON] = {}
    for message in messages:
        role = get(message, "role", str)
        if role == "user":
            for content_block in get_or_default(message, "content", list[JSON]):
                tool_use_id = get(content_block, "tool_use_id", str)
                if tool_use_id is not None and get(content_block, "type", str) == "tool_result":
                    result[tool_use_id] = content_block
        elif role == "assistant":
            for content_block in get_or_default(message, "content", list[JSON]):
                tool_use_id = get(content_block, "tool_use_id", str)
                if tool_use_id is not None:
                    result[tool_use_id] = content_block

    return result


def write_tool_result(tool_use: JSON, tool_result: JSON, io: StringIO, pretty: bool, wrap_width: int):
    tool_name = get_or(tool_use, "name", "<unknown>")

    if get(tool_result, "is_error", str):
        error_message = get_or(tool_result, "error", "<unknown>")
        write_error_block("Error", error_message, io, pretty, wrap_width)
        return

    if tool_name == "fetch_url":
        content = get_or(tool_result, "content", "")
        num_lines = content.count("\n") + 1
        result_text = f"Fetched HTML and converted it to {num_lines} lines of markdown text."
    elif tool_name == "web_search":
        results = get_or_default(tool_result, "content", list[JSON])
        result_text = f"Found {len(results)} references. See citations below."
    elif tool_name == "code_execution":
        output = get_or_default(tool_result, "content", dict[str, JSON])

        result_io = StringIO()
        if output:
            return_code = get_or(output, "return_code", 0)
            stdout = get_or(output, "stdout", "")
            stderr = get_or(output, "stderr", "")

            _ = result_io.write(f"Return code: {return_code}")
            if stdout:
                _ = result_io.write(f"\n\n{stdout.strip()}")
            if stderr:
                _ = result_io.write(f"\n\nstderr:\n{stderr.strip()}")

        result_text = result_io.getvalue()
    else:
        content = get(tool_result, "content", list[JSON])
        if content is None:
            content = get(tool_result, "text", str)
            if content is None:
                content = "<unknown>"
            content = [{"type": "text", "text": content}]

        result_io = StringIO()
        for content_block in content:
            if result_io.tell() > 0:
                _ = result_io.write("\n\n")
            match content_block:
                case {"type": "text", "text": str(text)}:
                    _ = result_io.write(text)
                case _:
                    _ = result_io.write(json.dumps(content_block, indent=2, sort_keys=True))

        result_text = result_io.getvalue()

    _ = io.write("\n")
    write_result_block("Result", result_text, io, pretty, wrap_width)


def write_tool_use(tool_use: JSON, tool_results: dict[str, JSON], io: StringIO, pretty: bool, wrap_width: int):
    def check_tool_result(title: str, text: str, tool_id: str) -> tuple[str, str, JSON]:
        tool_result = tool_results.get(tool_id)
        if tool_result is None:
            title += f" {spinner()}"
        else:
            if get(tool_result, "is_error", bool):
                title += " ✗"
            else:
                title += " ✓"
        return title, text, tool_result

    name = get(tool_use, "name", str)
    kind = get(tool_use, "type", str)

    if kind == "tool_use":
        title = f"Tool `{name}`"
    elif kind == "mcp_tool_use":
        server_name = get_or(tool_use, "server_name", "<unknown>")
        title = f"Tool `{name}` via `{server_name}`"
    elif kind == "server_tool_use":
        title = f"Server tool `{name}`"
    else:
        title = f"Unknown tool `{name}`"

    language = None

    input = get_or_default(tool_use, "input", dict[str, JSON])
    if name == "web_search":
        query = get_or(input, "query", "<unknown>")
        text = f"Query: {query}"
        title = "Web search"
    elif name == "code_execution":
        # It's always Python, see https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool
        language = "python"
        text = get_or(input, "code", "<unknown>")
        title = "Code execution"
    else:
        language = "json"
        text = json.dumps(input, indent=2, sort_keys=True)

    # The -2 accounts for the "╭─" and "╰─" indentation
    text = common.word_wrap(text, wrap_width - 2)

    if pretty and language:
        text = common.syntax_highlight(text, language)

    title, text, tool_result = check_tool_result(title, text, get_or(tool_use, "id", ""))
    write_call_block(title, text, io, pretty, wrap_width=0)  # We already wrapped prior to syntax highlighting
    if tool_result is not None:
        write_tool_result(tool_use, tool_result, io, pretty, wrap_width)


def write_user_message(
    message: JSON,
    io: StringIO,
    pretty: bool,
    wrap_width: int,
    skip_user_text: bool,
    uploaded_files: dict[str, JSON] | None,
    text_only: bool,
):
    prompt = f"{common.CHEVRON} "
    if pretty:
        prompt = common.prompt_style(prompt)

    files: dict[str, str] = {}

    for content_block in get_or_default(message, "content", list[JSON]):
        if text_only and get(content_block, "type", str) != "text":
            continue

        match content_block:
            case {"type": "text", "text": str(input)}:
                if not skip_user_text:
                    _ = io.write(f"{prompt}{common.input_style(input) if pretty else input}\n")
            case {"type": "document" | "image", "source": {"file_id": str(file_id)}}:
                files[file_id] = cast(str, content_block["type"])  # We know this is a str because of the match case
            case {"type": "container_upload", "file_id": str(file_id)}:
                # Record container uploads only if their ID isn't already in the files dict. This can happen, at which point we have more
                # precise knowledge about the file type already.
                if file_id not in files:
                    files[file_id] = "container_upload"
            case {"type": "tool_result"}:
                pass  # Tool results are handled in the assistant message's tool_use block
            case _:
                type = get_or(content_block, "type", "<unknown>")
                write_result_block(f"user `{type}`", json.dumps(content_block, indent=2, sort_keys=True), io, pretty, wrap_width)
                _ = io.write("\n\n")

    if files:
        files_io = StringIO()
        for key, group in groupby(sorted(files.items()), key=lambda x: x[1]):
            names = {
                "image": "Images",
                "document": "Documents",
                "container_upload": "Others",
            }

            _ = files_io.write(f"{names[key]}:\n")
            for file_id, _ in group:
                if uploaded_files is not None and file_id in uploaded_files:
                    file_name = get_or(uploaded_files[file_id], "filename", "<unknown>")
                    num_bytes = get_or(uploaded_files[file_id], "size_bytes", 0)
                    _ = files_io.write(f"- {file_name}, {naturalsize(num_bytes)} ({file_id})\n")
                else:
                    # If we don't have the file metadata, just use the ID
                    _ = files_io.write(f"- {file_id}\n")

        write_result_block("Files", files_io.getvalue(), io, pretty, wrap_width)
        _ = io.write("\n\n")


def write_assistant_message(tool_results: dict[str, JSON], message: JSON, io: StringIO, pretty: bool, wrap_width: int, text_only: bool):
    @dataclass(frozen=True)
    class Reference:
        id: int
        title: str
        cited_texts: set[str]

    references: dict[str, Reference] = {}

    def get_reference(key: str, title: str) -> Reference:
        return references.setdefault(key, Reference(id=len(references) + 1, title=title, cited_texts=set()))

    content_blocks = get_or_default(message, "content", list[JSON])

    # Iteration is manual, because text blocks can be split across multiple content blocks and need to have an inner loop.
    i = 0
    while i < len(content_blocks):
        content_block = content_blocks[i]
        block_type = get(content_block, "type", str)

        if text_only and block_type != "text":
            i += 1
            continue

        if block_type == "code_execution_tool_result" or block_type == "web_search_tool_result" or block_type == "mcp_tool_result":
            i += 1
            continue  # These are handled alongside the tool use

        if block_type == "thinking":
            write_call_block("Thinking", get_or(content_block, "thinking", ""), io, pretty, wrap_width)
        elif block_type == "text":
            text_io = StringIO()
            while i < len(content_blocks) and get(content_blocks[i], "type", str) == "text":
                content_block = content_blocks[i]
                _ = text_io.write(get_or(content_block, "text", ""))

                superscripts: set[str] = set()
                citations = get_or_default(content_block, "citations", list[dict[str, JSON]])
                for citation in citations:
                    match citation:
                        case {"type": "web_search_result_location", "url": str(url), "cited_text": str(cited_text)}:
                            title = url
                            page_title = get(citation, "title", str)
                            if page_title:
                                title += f" ({page_title})"

                            reference = get_reference(url, title)
                            reference.cited_texts.add(f"{escape(cited_text)}")
                            superscripts.add(str(reference.id))
                        case {
                            "type": "page_location",
                            "document_title": str(title),
                            "cited_text": str(cited_text),
                            "start_page_number": int(start_page_number),
                            "end_page_number": int(end_page_number),
                        }:
                            reference = get_reference(title, title)
                            end_page_number_incl = end_page_number - 1
                            if start_page_number == end_page_number_incl:
                                reference.cited_texts.add(f"Page {start_page_number}: {escape(cited_text)}")
                            else:
                                reference.cited_texts.add(f"Pages {start_page_number}-{end_page_number_incl}: {escape(cited_text)}")
                        case _:
                            reference = get_reference(get_or(citation, "type", "<unknown>"), "<unknown>")

                    superscripts.add(str(reference.id))

                _ = text_io.write(f"{to_superscript(','.join(sorted(superscripts)))}")
                i += 1

            text = common.word_wrap(rstrip(text_io).getvalue(), wrap_width)
            if pretty:
                text = common.syntax_highlight(text, "md")

            _ = io.write(text)
            i -= 1  # Adjust for the outer loop increment
        elif block_type == "tool_use" or block_type == "server_tool_use" or block_type == "mcp_tool_use":
            write_tool_use(content_block, tool_results, io, pretty, wrap_width)
        elif block_type == "redacted_thinking":
            encrypted_thinking = get_or(content_block, "data", "")
            write_call_block("Redacted thinking", f"{len(encrypted_thinking)} bytes of encrypted thinking data.", io, pretty, wrap_width)
        else:
            write_call_block(f"assistant `{block_type}`", json.dumps(content_block, indent=2, sort_keys=True), io, pretty, wrap_width)

        _ = io.write("\n\n")
        i += 1

    stop_reason = get(message, "stop_reason", str)

    if stop_reason is not None:
        if references:
            references_io = StringIO()
            for _, v in sorted(references.items(), key=lambda x: x[1].id):
                _ = references_io.write(f"{to_superscript(v.id)} {v.title}\n")
                for val in sorted(v.cited_texts):
                    _ = references_io.write(f"   {val}\n")
            write_result_block("References", references_io.getvalue(), io, pretty, wrap_width)
            _ = io.write("\n")

        if stop_reason != "end_turn" and stop_reason != "tool_use" and stop_reason != "pause_turn":
            _ = io.write(f"Response ended prematurely. **Stop reason:** {stop_reason}\n\n")


def history_to_string(
    history: History,
    pretty: bool,
    wrap_width: int = 0,
    skip_user_text: bool = False,
    uploaded_files: dict[str, JSON] | None = None,
    text_only: bool = False,
) -> str:
    tool_results = gather_tool_results(history)

    io = StringIO()
    for message in history:
        role = get(message, "role", str)
        if role == "system":
            write_system_message(message, io)
        elif role == "user":
            write_user_message(message, io, pretty, wrap_width=wrap_width, skip_user_text=skip_user_text, uploaded_files=uploaded_files, text_only=text_only)
        elif role == "assistant":
            write_assistant_message(tool_results, message, io, pretty, wrap_width=wrap_width, text_only=text_only)

    return rstrip(io).getvalue()


def print_decoy_prompt(user_input: str):
    """
    Reproduce the initial prompt that prompt_toolkit will produce. Requires a bit of bespoke formatting to match exactly.
    """
    if user_input:
        print(f"{common.prompt_style(common.CHEVRON)} {user_input}")
        return

    initial_prompt = common.char_wrap(f"  {common.HELP_TEXT}", os.get_terminal_size().columns - 2)
    num_newlines = initial_prompt.count("\n")
    ansi_return: str = "\033[F" * num_newlines + common.ansi("3G")
    print(f"{common.prompt_style(common.CHEVRON)} {common.gray_style(initial_prompt[2:])}{ansi_return}", end="", flush=True)


class PrintArgs(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.session: str
        self.sessions_dir: str = config.default_sessions_dir()
        self.pretty: bool = False
        self.verbose: bool = False


def main():
    """
    Main function to parse arguments, load a JSON file containing conversation history, and print it.
    """
    parser = argparse.ArgumentParser(description="Print conversation history from a JSON file")
    _ = parser.add_argument("session", help="Path to JSON file containing conversation history to print")
    _ = parser.add_argument("--sessions-dir", help="Path to directory for session files")
    _ = parser.add_argument("-p", "--pretty", action="store_true", help="Pretty print the conversation history using bat")
    _ = parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(namespace=PrintArgs())

    logging.setup(args.verbose)

    # Load the history from the JSON file
    try:
        history = common.load_session_if_exists(args.session, args.sessions_dir)
        if not history:
            raise FileNotFoundError(f"No session found at '{args.session}'.")

        wrap_width = os.get_terminal_size().columns if os.isatty(1) else 0

        result = history_to_string(history, pretty=args.pretty, wrap_width=wrap_width)
        print(result, end="", flush=True)
    except json.JSONDecodeError:
        logger.opt(exception=True).error(f"Could not parse JSON file {args.session}")
    except FileNotFoundError:
        logger.opt(exception=True).error(f"Could not find session file {args.session}")


if __name__ == "__main__":
    main()
