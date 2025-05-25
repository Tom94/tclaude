#!/usr/bin/env python3

import argparse
import json
import os
import sys

from io import StringIO

import common
from common import wrap_style


def to_superscript(text: str | int) -> str:
    if isinstance(text, int):
        text = str(text)
    superscript_map = str.maketrans("0123456789+-=(),", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒")
    return text.translate(superscript_map)


def write_system_message(message: dict, io: StringIO):
    io.write("\n# System prompt\n")
    for content_block in message["content"]:
        if content_block.type == "text":
            io.write(f"{content_block.text}\n")


def write_block(heading: str, block_text: str, io: StringIO, pretty: bool, color: str, wrap_width: int | None):
    io.write(wrap_style(f"╭── {heading}\n", color, pretty=pretty))
    block_text = common.word_wrap(block_text, wrap_width - 2 if wrap_width is not None else None)
    for line in block_text.splitlines():
        io.write(f"{wrap_style('│ ', color, pretty=pretty)}{line}\n")
    io.write(wrap_style("╰─", color, pretty=pretty))


def write_call_block(heading: str, block_text: str, io: StringIO, pretty: bool, wrap_width: int | None):
    write_block(heading, block_text, io, pretty, color="0;35m", wrap_width=wrap_width)


def write_result_block(heading: str, block_text: str, io: StringIO, pretty: bool, wrap_width: int | None):
    write_block(heading, block_text, io, pretty, color="0;36m", wrap_width=wrap_width)


def format_python_code(code: str) -> str:
    lines = code.splitlines()
    formatted_lines = []
    for line in lines:
        # Replace tabs with 4 spaces
        formatted_lines.append(line.replace("\t", " " * 4))
    return "\n".join(formatted_lines)


def gather_tool_results(messages: list[dict]) -> dict:
    """
    Find the tool result in the messages by tool ID.
    """
    result = {}
    for message in messages:
        if message.get("role") == "user":
            for content_block in message.get("content", []):
                tool_use_id = content_block.get("tool_use_id")
                if tool_use_id is not None and content_block.get("type") == "tool_result":
                    result[tool_use_id] = content_block
        elif message.get("role") == "assistant":
            for content_block in message.get("content", []):
                tool_use_id = content_block.get("tool_use_id")
                if tool_use_id is not None:
                    result[tool_use_id] = content_block

    return result


def write_tool_result(tool_use: dict, tool_result: dict, io: StringIO, pretty: bool, wrap_width: int | None):
    tool_name = tool_use.get("name", "<unknown>")

    if tool_name == "fetch_url":
        result_text = f"Fetched HTML and converted it to {tool_result.get('content', '').count('\n')} lines of markdown text."
    elif tool_name == "web_search":
        results = tool_result.get("content", [])
        result_text = f"Found {len(results)} references. See citations below."
    elif tool_name == "code_execution":
        output = tool_result.get("content", {})

        result_io = StringIO()
        if output:
            return_code = output.get("return_code", 0)
            stdout = output.get("stdout", "")
            stderr = output.get("stderr", "")

            result_io.write(f"Return code: {return_code}")
            if stdout:
                result_io.write(f"\n\n{stdout.strip()}")
            if stderr:
                result_io.write(f"\n\nstderr:\n{stderr.strip()}")

        result_text = result_io.getvalue()
    else:
        result_text = json.dumps(tool_result.get("content", {}), indent=2, sort_keys=True)

    io.write("\n")
    write_result_block("Result", result_text, io, pretty, wrap_width)


def write_tool_use(tool_use: dict, tool_results: dict, io: StringIO, pretty: bool, wrap_width: int | None):
    def check_tool_result(title: str, text: str, tool_id: str) -> tuple[str, str, dict | None]:
        tool_result = tool_results.get(tool_id)
        if tool_result is None:
            title += f" (running)"
        else:
            is_error = tool_result.get("is_error")
            if is_error:
                title += f" (error)"
            else:
                title += f" (done)"
        return title, text, tool_result

    name = tool_use.get("name")
    kind = tool_use.get("type")

    if kind == "tool_use":
        title = f"Tool `{name}`"
    elif kind == "server_tool_use":
        title = f"Server tool `{name}`"
    else:
        title = f"Unknown tool `{name}`"

    call_wrap_width = wrap_width
    if name == "web_search":
        query = tool_use.get("input", {}).get("query", "<unknown>")
        title = "Web search"
        text = f"Query: {query}"
    elif name == "code_execution":
        # It's always Python, see https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool
        python_code = common.word_wrap(
            tool_use.get("input", {}).get("code", "<unknown>"),
            wrap_width - 2 if wrap_width is not None else None,
        )

        if pretty:
            python_code = common.bat_syntax_highlight(python_code, "python")
            call_wrap_width = None  # pretty printing messes up wrap logic

        title = "Code execution"
        text = python_code
    else:
        text = json.dumps(tool_use.get("input", {}), indent=2, sort_keys=True)
        if pretty:
            text = common.bat_syntax_highlight(text, "json", wrap_width=wrap_width)
            call_wrap_width = None  # pretty printing messes up wrap logic
        title = f"Server tool `{name}`"

    title, text, tool_result = check_tool_result(title, text, tool_use.get("id", ""))
    write_call_block(title, text, io, pretty, call_wrap_width)
    if tool_result is not None:
        write_tool_result(tool_use, tool_result, io, pretty, wrap_width)


def write_user_message(message: dict, io: StringIO, pretty: bool, wrap_width: int | None):
    prompt = f"{common.CHEVRON} "

    for content_block in message.get("content", []):
        kind = content_block.get("type")
        if kind is None or kind == "tool_result":
            continue  # Tool results are handled in their `tool_use` block, not the result block

        if kind == "text":
            input = content_block.get("text", "")
            if pretty:
                prompt = common.prompt_style(prompt)
                input = common.input_style(input)

            io.write(f"{prompt}{input}\n")
        else:
            write_result_block(f"user `{kind}`", json.dumps(content_block, indent=2, sort_keys=True), io, pretty, wrap_width)
            io.write("\n\n")


def write_assistant_message(tool_results: dict, message: dict, io: StringIO, pretty: bool, wrap_width: int | None):
    references = {}

    content_blocks = message.get("content", [])

    # Iteration is manual, because text blocks can be split across multiple content blocks and need to have an inner loop.
    i = 0
    while i < len(content_blocks):
        content_block = content_blocks[i]
        block_type = content_block.get("type")
        if block_type == "code_execution_tool_result" or block_type == "web_search_tool_result":
            i += 1
            continue  # These are handled alongside the tool use

        if block_type == "thinking":
            write_call_block("Thinking", content_block.get("thinking", ""), io, pretty, wrap_width)
        elif block_type == "text":
            text_io = StringIO()
            while i < len(content_blocks) and content_blocks[i].get("type") == "text":
                content_block = content_blocks[i]
                text_io.write(content_block.get("text", ""))

                superscripts = set()
                citations = content_block.get("citations", [])
                for citation in citations:
                    if citation.get("type") != "web_search_result_location":
                        continue
                    url = citation.get("url")
                    if url is None:
                        continue

                    if not url in references:
                        references[url] = {
                            "title": citation.get("title", ""),
                            "cited_texts": set(),
                            "id": len(references) + 1,
                        }

                    reference = references[url]
                    if "cited_text" in citation:
                        reference["cited_texts"].add(citation["cited_text"])

                    superscripts.add(str(reference["id"]))

                text_io.write(f"{to_superscript(','.join(sorted(superscripts)))}")
                i += 1

            text = text_io.getvalue()
            if pretty:
                text = common.bat_syntax_highlight(text, "md", wrap_width=wrap_width)

            io.write(text.strip())
            i -= 1  # Adjust for the outer loop increment
        elif block_type == "tool_use" or block_type == "server_tool_use":
            write_tool_use(content_block, tool_results, io, pretty, wrap_width)
        elif block_type == "redacted_thinking":
            encrypted_thinking = content_block.get("data", "")
            write_call_block(f"Redacted thinking", f"{len(encrypted_thinking)} bytes of encrypted thinking data.", io, pretty, wrap_width)
        else:
            write_call_block(f"assistant `{block_type}`", json.dumps(content_block, indent=2, sort_keys=True), io, pretty, wrap_width)

        io.write(f"\n\n")
        i += 1

    stop_reason = message.get("stop_reason")

    if stop_reason is not None:
        if references:
            references_io = StringIO()
            for k, v in sorted(references.items(), key=lambda x: x[1]["id"]):
                references_io.write(f"{to_superscript(v['id'])} {k} - {v['title']}\n")
                for val in sorted(v["cited_texts"]):
                    references_io.write(f'   "{val}"\n')
            write_result_block("References", references_io.getvalue(), io, pretty, wrap_width)
            io.write("\n")

        if stop_reason != "end_turn" and stop_reason != "tool_use" and stop_reason != "pause_turn":
            io.write(f"Response ended prematurely. **Stop reason:** {stop_reason}\n\n")


def history_to_string(history: list[dict], pretty: bool, wrap_width: int | None = None) -> str:
    tool_results = gather_tool_results(history)

    io = StringIO()
    for message in history:
        if message["role"] == "system":
            write_system_message(message, io)
        elif message["role"] == "user":
            write_user_message(message, io, pretty, wrap_width=wrap_width)
        elif message["role"] == "assistant":
            write_assistant_message(tool_results, message, io, pretty, wrap_width=wrap_width)

    return io.getvalue().strip()


def main():
    """
    Main function to parse arguments, load a JSON file containing conversation history, and print it.
    """
    parser = argparse.ArgumentParser(description="Print conversation history from a JSON file")
    parser.add_argument("path", help="Path to JSON file containing conversation history to print")
    parser.add_argument("-p", "--pretty", action="store_true", help="Pretty print the conversation history using bat")

    args = parser.parse_args()

    # Check if the file exists and has a .json extension
    if not os.path.exists(args.path):
        print(f"Error: File '{args.path}' does not exist.")
        return

    if not args.path.endswith(".json"):
        print(f"Error: File '{args.path}' is not a JSON file.")
        return

    # Load the history from the JSON file
    try:
        with open(args.path, "r") as f:
            history = json.load(f)

        if sys.stdout.isatty():
            wrap_width = os.get_terminal_size().columns - 1
        else:
            wrap_width = None

        result = history_to_string(history, pretty=args.pretty, wrap_width=wrap_width)
        print(result, end="", flush=True)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file '{args.path}'.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
