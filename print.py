#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import sys
import warnings

from anthropic import Anthropic
from contextlib import nullcontext
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

CONSOLE = Console()


def pretty_print_prompt(prompt):
    CONSOLE.print(f"[bold #efb5f7]{prompt}[/]", end="")


def pretty_print_history(prompt, history):
    for message in history:
        if message["role"] == "system":
            CONSOLE.print("\n# System prompt")
            for content_block in message["content"]:
                if content_block.type == "text":
                    CONSOLE.print(content_block.text)
        elif message["role"] == "user":
            pretty_print_prompt(prompt)
            CONSOLE.print(message["content"])
        elif message["role"] == "assistant":
            had_thinking = False
            for content_block in message["content"]:
                if content_block["type"] == "thinking":
                    CONSOLE.print(Markdown(f"\n# Thought process\n{content_block['thinking']}"))
                    CONSOLE.print()

                    had_thinking = True

            for content_block in message["content"]:
                if content_block["type"] == "text":
                    if had_thinking:
                        CONSOLE.print(Markdown(f"\n# Thoughtful response\n{content_block['text']}\n"))
                    else:
                        CONSOLE.print(Markdown(f"{content_block['text']}"))
                    CONSOLE.print()


def main():
    """
    Main function to parse arguments, load a JSON file containing conversation history, and print it.
    """
    parser = argparse.ArgumentParser(description="Print conversation history from a JSON file")
    parser.add_argument("path", help="Path to JSON file containing conversation history to print")

    args = parser.parse_args()

    # Check if the file exists and has a .json extension
    if not os.path.exists(args.path):
        CONSOLE.print(f"Error: File '{args.path}' does not exist.", style="bold red")
        return

    if not args.path.endswith('.json'):
        CONSOLE.print(f"Error: File '{args.path}' is not a JSON file.", style="bold red")
        return

    # Load the history from the JSON file
    try:
        with open(args.path, "r") as f:
            history = json.load(f)

        pretty_print_history("user> ", history)
    except json.JSONDecodeError:
        CONSOLE.print(f"Error: Could not parse JSON file '{args.path}'.", style="bold red")
    except Exception as e:
        CONSOLE.print(f"Error: {str(e)}", style="bold red")


if __name__ == "__main__":
    main()
