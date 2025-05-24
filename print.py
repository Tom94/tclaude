#!/usr/bin/env python3

import argparse
import json
import os

from io import StringIO

import common


def to_superscript(text: str | int) -> str:
    if isinstance(text, int):
        text = str(text)
    superscript_map = str.maketrans("0123456789+-=(),", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾˒")
    return text.translate(superscript_map)


def history_to_string(history, pretty, wrap_width: int | None = None):
    io = StringIO()
    for message in history:
        if message["role"] == "system":
            io.write("\n# System prompt\n")
            for content_block in message["content"]:
                if content_block.type == "text":
                    io.write(f"{content_block.text}\n")
        elif message["role"] == "user":
            prompt = f"{common.CHEVRON} "
            input = message["content"][0]["text"]
            if pretty:
                prompt = f"{common.ansi('95m')}{prompt}"
                input = f"{common.ansi('97m')}{input}{common.ansi('0m')}"

            io.write(f"{prompt}{input}\n")
        elif message["role"] == "assistant":
            references = {}

            block_type = None
            for content_block in message["content"]:
                if block_type is not None and block_type != content_block["type"]:
                    io.write(f"\n\n")

                block_type = content_block.get("type")

                if block_type == "thinking":
                    io.write(f"```thinking\n{content_block['thinking']}\n```")
                elif block_type == "text":
                    io.write(f"{content_block['text']}")
                elif block_type == "server_tool_use":
                    name = content_block.get("name")
                    if name == "web_search":
                        io.write(f"Searching the web for `{content_block.get("input", {}).get("query", "<unknown>")}`.")
                    elif name == "code_execution":
                        # It's always Python, see https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool
                        io.write(f"Running the following Python code:\n\n")
                        io.write(f"```python\n{content_block.get("input", {}).get("code", "<unknown>")}\n```")
                    else:
                        io.write(f"Unknown tool: {name}")
                elif block_type == "web_search_tool_result":
                    io.write(f"Found {len(content_block.get("content", []))} results.")
                elif block_type == "code_execution_tool_result":
                    output = content_block.get("content", {})

                    if output:
                        return_code = output.get("return_code", 0)
                        if return_code != 0:
                            io.write(f"Code execution failed with return code {return_code}.")

                        stdout = output.get("stdout", "")
                        if stdout:
                            io.write(f"```stdout\n{stdout.strip()}\n```")

                        stderr = output.get("stderr", "")
                        if stderr:
                            io.write(f"```stderr\n{stderr.strip()}\n```")

                else:
                    io.write(f"```{block_type}\n{content_block}\n```")

                if content_block and content_block.get("type") == "text" and content_block.get("citations"):
                    superscripts = set()
                    for citation in content_block["citations"]:
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

                    io.write(f"{to_superscript(','.join(sorted(superscripts)))}")

            if block_type is not None:
                io.write("\n\n")

            if references:
                for k, v in sorted(references.items(), key=lambda x: x[1]["id"]):
                    io.write(f"{to_superscript(v['id'])} {k} - {v['title']}\n")
                    for val in sorted(v["cited_texts"]):
                        io.write(f'   "{val}"\n')
                io.write("\n")

    result = io.getvalue()
    if pretty:
        result = common.pretty_print_md(result, wrap_width=wrap_width)

    return result.rstrip()


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

        result = history_to_string(history, pretty=args.pretty)
        if args.pretty:
            result = common.pretty_print_md(result)
        print(result, end="", flush=True)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file '{args.path}'.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
