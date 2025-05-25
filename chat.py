#!/usr/bin/env python3

import common

# Print prompt before doing anything else to hide startup delay (which is caused by importing dependencies).
HELP_TEXT = common.wrap_style("Type your message and hit Enter. Ctrl-C to exit, ESC for Vi mode, \\-Enter for newline.", "38;5;245m")
print(f"{common.prompt_style(common.CHEVRON)} {HELP_TEXT}{common.ansi('3G')}", end="", flush=True)

import datetime
import json
import os
import sys

from io import StringIO

from print import history_to_string
from prompt import stream_response, TokenCounter

from prompt_toolkit import PromptSession, print_formatted_text, ANSI, HTML
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.key_binding import KeyBindings


def create_prompt_key_bindings():
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        event.app.current_buffer.validate_and_handle()

    @bindings.add("\\", "enter")
    def _(event):
        event.app.current_buffer.newline()

    @bindings.add("c-p")
    def _(event):
        event.app.current_buffer.history_backward()

    @bindings.add("c-n")
    def _(event):
        event.app.current_buffer.history_forward()

    return bindings


def user_prompt(lprompt: str, rprompt: str, prompt_session: PromptSession, key_bindings: KeyBindings) -> str:
    print(f"\r", end="")  # Ensure we don't have stray remaining characters from user typing before the prompt was ready.
    user_input = ""
    while not user_input:
        user_input = prompt_session.prompt(
            ANSI(common.prompt_style(lprompt)),
            rprompt=ANSI(common.prompt_style(rprompt)),
            vi_mode=True,
            cursor=ModalCursorShapeConfig(),
            multiline=True,
            placeholder=ANSI(HELP_TEXT),
            key_bindings=key_bindings,
        ).strip()

    return user_input


def should_cache(tokens: TokenCounter, model: str) -> bool:
    """
    We heuristically set a new cache breakpoint when our next prompt (if short ~0 tokens) causes the cost of input to be larger
    than that of cache reads.
    TODO: If we just finished a web search, apparently something messy happens to the cache... should investigate
    """
    tokens_if_short_follow_up = TokenCounter(
        cache_creation_input_tokens=0,
        cache_read_input_tokens=tokens.cache_read + tokens.cache_creation,
        input_tokens=tokens.input + tokens.output,
        output_tokens=0,
    )
    _, cache_read_cost, input_cost, _ = tokens_if_short_follow_up.cost(model)
    return cache_read_cost < input_cost


def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    args = common.parse_args()

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

    prompt_key_bindings = create_prompt_key_bindings()
    prompt_session = PromptSession()

    # Initialize or load messages history
    history = []
    if args.session and os.path.exists(args.session):
        try:
            with open(args.session, "r") as f:
                history = json.load(f)
                for message in history:
                    if message.get("role") == "user":
                        text = message.get("content", [{}])[0].get("text", "")
                        if text:
                            prompt_session.history.append_string(text)

                print(history_to_string(history, pretty=True))
        except json.JSONDecodeError:
            print(f"Error: Could not parse session file {args.session}. Starting new session.")
            return

    initial_history_length = len(history)

    total_tokens = TokenCounter()

    # Initially, don't cache anything. The system prompt is always cached.
    write_cache = False

    # Print the current state of the response. Keep overwriting the same lines since the response is getting incrementally built.
    num_newlines_printed = 0

    def reprint_current_response(messages: list[dict], _: TokenCounter, limit_to_terminal_height: bool = True):
        nonlocal num_newlines_printed

        to_print = StringIO()
        to_print.write("\033[F" * num_newlines_printed)
        to_print.write("\r")

        term_width = os.get_terminal_size().columns
        term_height = os.get_terminal_size().lines

        # Print the last term_height - 1 lines of the history to avoid terminal problems
        current_message = history_to_string(messages, pretty=True, wrap_width=term_width - 1)
        lines = current_message.split("\n")

        if limit_to_terminal_height:
            if len(lines) >= term_height:
                lines = lines[-(term_height - 1) :]

        for line in lines:
            to_print.write(f"\033[K{line}\n")

        print(to_print.getvalue().rstrip(), end="", flush=True)
        num_newlines_printed = len(lines) - 1

    is_user_turn = True
    try:
        while True:
            num_newlines_printed = 0

            if is_user_turn:
                if not user_input:
                    lprompt = f"{common.CHEVRON} "
                    rprompt = f"{total_tokens.total_cost(args.model):.03f}   {common.friendly_model_name(args.model)} "
                    user_input = user_prompt(lprompt, rprompt, prompt_session, prompt_key_bindings)
                else:
                    prompt_session.history.append_string(user_input)
                    print_formatted_text(ANSI(common.prompt_style(f"{common.CHEVRON} {user_input}")))

                history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
                user_input = ""
            else:
                # Either, the response was paused before (stop_reason == "pause_turn") or we are providing tool results (stop_reason == "tool_use").
                pass

            def on_response_interrupt():
                nonlocal is_user_turn
                if is_user_turn:
                    history.pop()
                is_user_turn = True

            try:
                # The response is already printed during streaming, so we don't need to print it again
                messages, tokens, call_again = stream_response(
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

                is_user_turn = not call_again
            except KeyboardInterrupt:
                print("\n\nResponse interrupted by user.\n")
                on_response_interrupt()
                continue
            except Exception as e:
                print(f"\n\nUnexpected error: {e}\nPlease try again.\n")
                on_response_interrupt()
                continue

            history.extend(messages)
            total_tokens += tokens

            # Final print of the response that doesn't have a line limit (because we no longer have to overwrite it)
            reprint_current_response(messages, tokens, limit_to_terminal_height=False)

            # Automatically determine whether we should put a cache breakpoint into the next prompt
            write_cache = should_cache(tokens, args.model)

            print("\n")
            if args.verbose:
                tokens.print_tokens()
                tokens.print_cost(args.model)
                if write_cache:
                    print("Next prompt will be cached.\n")
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass

    print()

    # If we submitted a user prompt and received a response (at least 2 messages), save the session
    if len(history) - initial_history_length >= 2:
        session_path = args.session
        if session_path is None:
            print("Auto-naming session... ", end="", flush=True)
            autoname_prompt = (
                "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you."
            )

            history.append({"role": "user", "content": [{"type": "text", "text": autoname_prompt}]})
            try:
                messages, tokens, _ = stream_response(
                    model=args.model,
                    history=history,
                    max_tokens=30,
                    enable_web_search=False,
                    system_prompt=system_prompt,
                    enable_thinking=False,
                )

                total_tokens += tokens
                session_name = history_to_string(messages, pretty=False)
                print("\r", end="", flush=True)
            except KeyboardInterrupt:
                print("interrupted by user.")
                print("Falling back to time stamp.")
                session_name = datetime.datetime.now().strftime("%H-%M-%S")
            except Exception as e:
                print(f"error auto-naming session: {e}")
                print(f"Falling back to time stamp.")
                session_name = datetime.datetime.now().strftime("%H-%M-%S")
            finally:
                history.pop()

            session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
            session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

            date = datetime.datetime.now().strftime("%Y-%m-%d")
            session_name = f"{date}-{session_name}.json"
            session_path = os.path.join(args.sessions_dir, session_name)

        with open(session_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"✓ Saved session to {session_path}")

        if args.verbose:
            total_tokens.print_tokens()
            total_tokens.print_cost(args.model)


if __name__ == "__main__":
    main()
