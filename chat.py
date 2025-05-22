#!/usr/bin/env python3

import argparse
import os
import sys
import json
import warnings
from anthropic import Anthropic

# Suppress the LibreSSL warning from urllib3
warnings.filterwarnings("ignore", category=Warning, message=".*OpenSSL 1.1.1.*")

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]


def get_anthropic_response(user_input, model="claude-3-7-sonnet-20250219", session_file=None, max_tokens=16384, enable_web_search=False):
    """
    Send user input to Anthropic API and get the response using the Anthropic Python client.

    Args:
        user_input (str): The user's input message
        model (str): The Anthropic model to use
        session_file (str): Path to a session file for conversation history
        max_tokens (int): Maximum number of tokens in the response
        enable_web_search (bool): Whether to enable the web search tool

    Returns:
        str: The response from Anthropic's API
        list: Updated messages history
    """
    # Initialize the Anthropic client
    client = Anthropic()

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

    # Prepare request parameters
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }

    # Add web search tool if enabled
    if enable_web_search:
        web_search_tool = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": MAX_SEARCH_USES
        }

        # Add domain filters if specified
        if ALLOWED_DOMAINS:
            web_search_tool["allowed_domains"] = ALLOWED_DOMAINS
        elif BLOCKED_DOMAINS:
            web_search_tool["blocked_domains"] = BLOCKED_DOMAINS

        params["tools"] = [web_search_tool]

    try:
        # Make the API request using the Anthropic client
        response = client.messages.create(**params)

        # Process the response content
        if enable_web_search:
            # For web search, we need to extract the text and handle citations
            assistant_message = ""
            citations = []

            for content_block in response.content:
                if content_block.type == "text":
                    assistant_message += content_block.text

                    # Handle citations if present
                    if hasattr(content_block, "citations") and content_block.citations:
                        for citation in content_block.citations:
                            if hasattr(citation, "type") and citation.type == "web_search_result_location":
                                citations.append({
                                    "url": citation.url,
                                    "title": citation.title,
                                    "cited_text": citation.cited_text
                                })

            # Add citations to the end of the message if any exist
            if citations:
                assistant_message += "\n\nSources:\n"
                for i, citation in enumerate(citations, 1):
                    assistant_message += f"{i}. {citation['title']} - {citation['url']}\n"
        else:
            # For regular responses without web search
            assistant_message = response.content[0].text

        # Add assistant response to history (store the original response for multi-turn conversations)
        # Convert the Pydantic model to a dictionary using the built-in to_dict() method
        serializable_content = [block.model_dump() for block in response.content]
        messages.append({"role": "assistant", "content": serializable_content})

        # Save updated history if session file is specified
        if session_file:
            with open(session_file, "w") as f:
                json.dump(messages, f, indent=2)

        return assistant_message, messages

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, messages


def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("-m", "--model", default="claude-3-7-sonnet-20250219", help="Anthropic model to use (default: claude-3.7-sonnet)")
    parser.add_argument("--max-tokens", type=int, default=2**14, help="Maximum number of tokens in the response (default: 16384)")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability (enabled by default)")

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
        response, _ = get_anthropic_response(
            user_input,
            model=args.model,
            session_file=args.session,
            max_tokens=args.max_tokens,
            enable_web_search=not args.no_web_search,  # Web search is enabled by default
        )
        print("\nAnthropic's response:")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
