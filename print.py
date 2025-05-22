#!/usr/bin/env python3

import argparse
import json
import os

def print_history(prompt, history):
    for message in history:
        if message["role"] == "system":
            print("\n# System prompt")
            for content_block in message["content"]:
                if content_block.type == "text":
                    print(content_block.text)
        elif message["role"] == "user":
            print(f"{prompt}{message['content']}")
        elif message["role"] == "assistant":
            had_thinking = False
            for content_block in message["content"]:
                if content_block["type"] == "thinking":
                    print(f"\n# Thought process\n{content_block['thinking']}\n")

                    had_thinking = True

            for content_block in message["content"]:
                if content_block["type"] == "text":
                    if had_thinking:
                        print(f"\n# Thoughtful response\n{content_block['text']}\n")
                    else:
                        print(f"{content_block['text']}")
                    print()


def main():
    """
    Main function to parse arguments, load a JSON file containing conversation history, and print it.
    """
    parser = argparse.ArgumentParser(description="Print conversation history from a JSON file")
    parser.add_argument("path", help="Path to JSON file containing conversation history to print")

    args = parser.parse_args()

    # Check if the file exists and has a .json extension
    if not os.path.exists(args.path):
        print(f"Error: File '{args.path}' does not exist.")
        return

    if not args.path.endswith('.json'):
        print(f"Error: File '{args.path}' is not a JSON file.")
        return

    # Load the history from the JSON file
    try:
        with open(args.path, "r") as f:
            history = json.load(f)

        print_history(" ", history)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file '{args.path}'.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
