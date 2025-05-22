#!/usr/bin/env python3

import argparse
import json
import os

from io import StringIO

import common


def history_to_string(prompt, history):
    io = StringIO()
    for message in history:
        if message["role"] == "system":
            io.write("\n# System prompt\n")
            for content_block in message["content"]:
                if content_block.type == "text":
                    io.write(f"{content_block.text}\n")
        elif message["role"] == "user":
            io.write(f"{prompt}{message['content']}\n")
        elif message["role"] == "assistant":
            had_thinking = False
            for content_block in message["content"]:
                if content_block["type"] == "thinking":
                    io.write(f"\n# Thought process\n{content_block['thinking']}\n\n")

                    had_thinking = True

            for content_block in message["content"]:
                if content_block["type"] == "text":
                    if had_thinking:
                        io.write(f"\n# Thoughtful response\n{content_block['text']}\n\n")
                    else:
                        io.write(f"{content_block['text']}\n")
                    io.write("\n")

    return io.getvalue().rstrip()


def history_to_pretty_string(prompt, history):
    return common.pretty_print_md(history_to_string(prompt, history))


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

        result = history_to_string(common.PROMPT, history)
        if args.pretty:
            result = common.pretty_print_md(result)
        print(result, end="", flush=True)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file '{args.path}'.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
