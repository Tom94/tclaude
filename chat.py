#!/usr/bin/env python3

import argparse
import asyncio
import datetime
import json
import os
import sys

import common
from print import history_to_string
from prompt import get_anthropic_response, TokenCounter


async def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("-r", "--role", help="Path to a markdown file containing a system prompt")
    parser.add_argument("-m", "--model", default="claude-opus-4-0", help="Anthropic model to use")
    parser.add_argument("--max-tokens", type=int, default=2**14, help="Maximum number of tokens in the response")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability")
    parser.add_argument("--no-code-execution", action="store_true", help="Disable code execution capability")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    parser.add_argument("--thinking-budget", type=int, help="Number of tokens to allocate for thinking (min 1024)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if not sys.stdin.isatty():
        print(f"{sys.argv[0]} should only be run in interactive mode. Use prompt.py otherwise.")
        return

    # Get user input from arguments or stdin
    user_input = ""
    if args.input:
        user_input = " ".join(args.input)

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
                print(history_to_string(history, pretty=True), "\n")
        except json.JSONDecodeError:
            print(f"Error: Could not parse session file {args.session}. Starting new session.")
            return

    total_tokens = TokenCounter()

    received_response = False

    # Initially, don't cache anything. The system prompt is always cached.
    write_cache = False
    running_cost = 0.0

    try:
        while True:
            prompt_prefix = f"${running_cost:.02f} {common.friendly_model_name(args.model)} "
            prompt = common.prompt(prompt_prefix, True)
            if not user_input:
                user_input = input(prompt).strip()
                if not user_input:
                    continue
            else:
                print(f"{prompt}{user_input}")

            num_newlines_printed = 0

            def reprint_current_response(message, tokens):
                nonlocal num_newlines_printed
                # Print the current state of the response. Keep overwriting the same lines since the response is getting incrementally built.
                wrap_width = os.get_terminal_size().columns - 1
                to_print = history_to_string([message], pretty=True, wrap_width=wrap_width)

                # go up num_newlines_printed lines and erase them
                print("\033[F" * num_newlines_printed + "\r", end="")
                num_newlines_printed = to_print.count("\n")

                print(to_print, end="", flush=True)

            # The response is already printed during streaming, so we don't need to print it again
            _, _, tokens = await get_anthropic_response(
                user_input,
                model=args.model,
                history=history,
                max_tokens=args.max_tokens,
                enable_web_search=not args.no_web_search,  # Web search is enabled by default
                enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
                system_prompt=system_prompt,
                enable_thinking=args.thinking,
                thinking_budget=args.thinking_budget,
                write_cache=write_cache,
                on_response_update=reprint_current_response,
            )

            total_tokens += tokens
            running_cost = total_tokens.total_cost(args.model)

            # We heuristically set a new cache breakpoint when our next prompt (if short ~0 tokens) causes the cost of input to be larger
            # than that of cache reads.
            # TODO: If we just finished a web search, apparently something messy happens to the cache... should investigate
            tokens_if_short_follow_up = TokenCounter(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=tokens.cache_read + tokens.cache_creation,
                input_tokens=tokens.input + tokens.output,
                output_tokens=0,
            )
            _, cache_read_cost, input_cost, _ = tokens_if_short_follow_up.cost(args.model)
            write_cache = cache_read_cost < input_cost

            # An empty line between each prompt
            print()
            print()

            if args.verbose:
                tokens.print_tokens()
                tokens.print_cost(args.model)
                if write_cache:
                    print("Next prompt will be cached.")
                print()

            received_response = True
            user_input = ""
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass

    if received_response:
        # Save updated history if session file is specified
        session_name = args.session
        if session_name is None:
            print("Auto-naming session file...")
            _, message, tokens = await get_anthropic_response(
                "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you.",
                model=args.model,
                history=history.copy(),  # Using a copy ensures we don't modify the original history
                max_tokens=30,
                enable_web_search=False,
                system_prompt=system_prompt,
                enable_thinking=False,
            )

            total_tokens += tokens

            session_name = history_to_string("", [message])

            session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
            session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

            date = datetime.datetime.now().strftime("%Y-%m-%d")
            session_name = f"{date}-{session_name}.json"

        print(f"Saving session as {session_name}...")
        with open(session_name, "w") as f:
            json.dump(history, f, indent=2)

        if args.verbose:
            total_tokens.print_tokens()
            total_tokens.print_cost(args.model)

        print(f"Total cost: ${total_tokens.total_cost(args.model):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
