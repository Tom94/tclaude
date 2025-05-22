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

# Initialize the Anthropic client
CLIENT = Anthropic()


def get_anthropic_response(
    user_input,
    model="claude-3-7-sonnet-20250219",
    history=[],
    max_tokens=16384,
    enable_web_search=False,
    system_prompt=None,
    enable_thinking=False,
    thinking_budget=None,
):
    """
    Send user input to Anthropic API and get the response using the Anthropic Python client.
    Uses streaming for incremental output.
    """
    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Prepare request parameters
    params = {"model": model, "max_tokens": max_tokens, "messages": history}

    # Add system prompt if provided
    if system_prompt:
        params["system"] = system_prompt

    # Add extended thinking if enabled
    if enable_thinking:
        thinking_config = {"type": "enabled", "budget_tokens": thinking_budget if thinking_budget else max(1024, max_tokens // 2)}
        params["thinking"] = thinking_config

    # Add web search tool if enabled
    if enable_web_search:
        web_search_tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCH_USES}

        # Add domain filters if specified
        if ALLOWED_DOMAINS:
            web_search_tool["allowed_domains"] = ALLOWED_DOMAINS
        elif BLOCKED_DOMAINS:
            web_search_tool["blocked_domains"] = BLOCKED_DOMAINS

        params["tools"] = [web_search_tool]

    try:
        text_response = ""

        def print_stream(text):
            print(text, end="", flush=True)
            nonlocal text_response
            text_response += text

        with CLIENT.messages.stream(**params) as stream:
            # Track if we're currently in a thinking block
            in_thinking_section = False

            # Process each event in the stream
            for event in stream:
                # Handle different event types
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        if not in_thinking_section:
                            print_stream("\n# Thought process\n")
                            in_thinking_section = True

                        # Print the thinking text
                        print_stream(event.delta.thinking)

                    elif event.delta.type == "text_delta":
                        if in_thinking_section:
                            print_stream("\n\n# Thoughtful response\n")
                            in_thinking_section = False

                        # Print the text and add it to our response
                        print_stream(event.delta.text)

            # Print a newline after the streaming is complete
            if not in_thinking_section:
                print_stream("\n")

            # Get the final message with all content
            final_message = stream.get_final_message()

            citations = []
            if enable_web_search:
                # Extract citations from the response
                for content_block in final_message.content:
                    if content_block.type == "text" and hasattr(content_block, "citations") and content_block.citations:
                        for citation in content_block.citations:
                            if hasattr(citation, "type") and citation.type == "web_search_result_location":
                                citations.append({"url": citation.url, "title": citation.title, "cited_text": citation.cited_text})

            if citations:
                print_stream("\nSources:\n")
                for i, citation in enumerate(citations, 1):
                    print_stream(f"{i}. {citation['title']} - {citation['url']}\n")

            # Add assistant response to history
            # Convert the Pydantic model to a dictionary for JSON serialization
            serializable_content = [block.model_dump() for block in final_message.content]
            history.append({"role": "assistant", "content": serializable_content})

            return text_response, history

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, history


def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("-r", "--role", help="Path to a markdown file containing a system prompt")
    parser.add_argument("-m", "--model", default="claude-3-7-sonnet-20250219", help="Anthropic model to use (default: claude-3.7-sonnet)")
    parser.add_argument("--max-tokens", type=int, default=2**14, help="Maximum number of tokens in the response (default: 16384)")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability (enabled by default)")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    parser.add_argument("--thinking-budget", type=int, help="Number of tokens to allocate for thinking (min 1024)")

    args = parser.parse_args()

    # Get user input from arguments or stdin
    user_input = ""
    is_repl = False
    if args.input:
        user_input = " ".join(args.input)
    elif not sys.stdin.isatty() and not sys.stdin.closed:
        user_input = sys.stdin.read().strip()
    else:
        is_repl = sys.stdin.isatty()

    if not user_input and not is_repl:
        print("No input provided.")
        return

    # Read system prompt from file if provided
    system_prompt = None
    if args.role:
        try:
            with open(args.role, "r") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading system prompt file: {e}")
            return

    # Initialize or load messages history
    history = []
    if args.session and os.path.exists(args.session):
        try:
            with open(args.session, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse session file {args.session}. Starting new session.")
            return

    try:
        while True:
            if is_repl:
                # Get system username
                username = os.getenv("USER") or os.getenv("USERNAME") or "User"
                user_input = input(f"{username}> ").strip()

                if not user_input:
                    continue

            # The response is already printed during streaming, so we don't need to print it again
            # We're ignoring the returned response since it's already been printed
            _, _ = get_anthropic_response(
                user_input,
                model=args.model,
                history=history,
                max_tokens=args.max_tokens,
                enable_web_search=not args.no_web_search,  # Web search is enabled by default
                system_prompt=system_prompt,
                enable_thinking=args.thinking,
                thinking_budget=args.thinking_budget,
            )
            print()

            if not is_repl:
                break
    except KeyboardInterrupt:
        pass

    # Save updated history if session file is specified
    if args.session:
        print(f"\nSaving session as {args.session}...")
        with open(args.session, "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
