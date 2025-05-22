#!/usr/bin/env python3

import argparse
import os
import sys
import json
import warnings
import requests

# Suppress the LibreSSL warning from urllib3
warnings.filterwarnings("ignore", category=Warning, message=".*OpenSSL 1.1.1.*")


def get_anthropic_response(user_input, model="claude-3.7-sonnet", session_file=None):
    """
    Send user input to Anthropic API and get the response.

    Args:
        user_input (str): The user's input message
        model (str): The Anthropic model to use
        session_file (str): Path to a session file for conversation history

    Returns:
        str: The response from Anthropic's API
        list: Updated messages history
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    headers = {"x-api-key": api_key, "content-type": "application/json", "anthropic-version": "2023-06-01"}

    # Initialize or load messages history
    messages = []
    if session_file and os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                messages = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse session file {session_file}. Starting new session.")

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    data = {"model": model, "max_tokens": 1000, "messages": messages}

    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)

    if response.status_code != 200:
        error_message = f"Error: {response.status_code} - {response.text}"
        return error_message, messages

    response_data = response.json()
    assistant_message = response_data["content"][0]["text"]

    # Add assistant response to history
    messages.append({"role": "assistant", "content": assistant_message})

    # Save updated history if session file is specified
    if session_file:
        with open(session_file, "w") as f:
            json.dump(messages, f, indent=2)

    return assistant_message, messages


def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("-m", "--model", default="claude-3-7-sonnet-20250219", help="Anthropic model to use (default: claude-3.7-sonnet)")

    args = parser.parse_args()

    # Get user input from arguments or stdin
    if args.input:
        user_input = " ".join(args.input)
    else:
        print("Enter your message (Ctrl+D to submit):")
        user_input = sys.stdin.read().strip()

    if not user_input:
        print("No input provided.")
        return

    try:
        response, _ = get_anthropic_response(user_input, model=args.model, session_file=args.session)
        print("\nAnthropic's response:")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
