#!/usr/bin/env python3

import asyncio
import common
import datetime
import json
import os

from io import StringIO
from prompt_toolkit import PromptSession, print_formatted_text, ANSI
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from print import history_to_string
from prompt import stream_response, TokenCounter


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


async def user_prompt(lprompt: str, rprompt: str, prompt_session: PromptSession, key_bindings: KeyBindings) -> str:
    print(common.ansi("1G"), end="")  # Ensure we don't have stray remaining characters from user typing before the prompt was ready.
    user_input = ""
    while not user_input:
        with patch_stdout():
            user_input = await prompt_session.prompt_async(
                ANSI(common.prompt_style(lprompt)),
                rprompt=ANSI(common.prompt_style(rprompt)),
                vi_mode=True,
                cursor=ModalCursorShapeConfig(),
                multiline=True,
                wrap_lines=True,
                placeholder=ANSI(common.gray_style(common.HELP_TEXT)),
                key_bindings=key_bindings,
            )

    return user_input.strip()


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


async def async_main(args, history: list[dict]):
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    # Get user input from arguments or stdin
    user_input = ""
    if args.input:
        user_input = " ".join(args.input)

    # Read system prompt from file if provided
    system_prompt = None
    if args.role:
        system_prompt = common.load_system_prompt(args.role)

    session_name = None
    if args.session:
        session_name = os.path.splitext(os.path.basename(args.session))[0]

    initial_history_length = len(history)

    prompt_key_bindings = create_prompt_key_bindings()
    prompt_session = PromptSession()

    for message in history:
        if message.get("role") == "user":
            text = message.get("content", [{}])[0].get("text", "")
            if text:
                prompt_session.history.append_string(text)

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

        current_message = history_to_string(messages, pretty=True, wrap_width=term_width)

        lines = current_message.split("\n")

        if limit_to_terminal_height:
            # Print the last term_height - 1 lines of the history to avoid terminal problems
            if len(lines) >= term_height:
                lines = lines[-(term_height - 1) :]

        for line in lines:
            to_print.write(f"\033[K{line}\n")

        print(to_print.getvalue().rstrip(), end="", flush=True)
        num_newlines_printed = len(lines) - 1

    async def handle_autoname_result(autoname_task: asyncio.Task) -> str:
        nonlocal total_tokens

        try:
            messages, tokens, _ = await autoname_task
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

        session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
        session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

        date = datetime.datetime.now().strftime("%Y-%m-%d")
        return f"{date}-{session_name}.json"

    is_user_turn = True
    autoname_task = None
    try:
        while True:
            num_newlines_printed = 0

            if is_user_turn:
                if not user_input:
                    lprompt = f"{common.CHEVRON} "
                    rprompt = f"{total_tokens.total_cost(args.model):.03f}   {common.friendly_model_name(args.model)} "
                    if args.role:
                        prompt_role = os.path.splitext(os.path.basename(args.role))[0]
                        rprompt = f"󱜙 {prompt_role}  {rprompt}"
                    if session_name:
                        rprompt = f" {session_name}  {rprompt}"

                    user_input = await user_prompt(lprompt, rprompt, prompt_session, prompt_key_bindings)
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
                messages, tokens, call_again = await stream_response(
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

            if not messages:
                print("\n\nNo response received. Please try again.\n")
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

            # Start a background task to auto-name the session if it is not already named
            if session_name is None:
                if autoname_task is None and is_user_turn:
                    print(common.gray_style("Auto-naming session in the background...\n"))
                    autoname_prompt = (
                        "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you."
                    )

                    autoname_history = history.copy() + [{"role": "user", "content": [{"type": "text", "text": autoname_prompt}]}]
                    autoname_task = asyncio.create_task(
                        stream_response(
                            model=args.model,
                            history=autoname_history,
                            max_tokens=30,
                            enable_web_search=False,
                            system_prompt=system_prompt,
                            enable_thinking=False,
                        )
                    )

                if autoname_task is not None and autoname_task.done():
                    session_name = await handle_autoname_result(autoname_task)

    except KeyboardInterrupt:
        pass
    except EOFError:
        pass

    print()

    # If we submitted a user prompt and received a response (at least 2 messages), save the session
    if len(history) - initial_history_length >= 2:
        session_path = args.session

        if session_path is None:
            if autoname_task is not None:
                session_name = await handle_autoname_result(autoname_task)

            if session_name is not None:
                session_path = os.path.join(args.sessions_dir, session_name)

        if session_path is not None:
            with open(session_path, "w") as f:
                json.dump(history, f, indent=2)

            print(f"✓ Saved session to {session_path}")

            if args.verbose:
                total_tokens.print_tokens()
                total_tokens.print_cost(args.model)


def main(args, history: list[dict]):
    asyncio.run(async_main(args, history))


if __name__ == "__main__":
    args = common.parse_args()

    history = common.load_session(args.session) if args.session else []
    if history:
        print(history_to_string(history, pretty=True, wrap_width=os.get_terminal_size().columns))

    main(args, history)
