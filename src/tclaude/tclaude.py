# tclaude -- Claude in the terminal
#
# Copyright (C) 2025 Thomas Müller <contact@tom94.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import logging
import os
import sys

from . import common, logging_config
from .config import load_config, parse_tclaude_args
from .print import history_to_string

logger = logging.getLogger(__package__)


def read_user_input(input: list[str]) -> str:
    user_input = ""
    if not sys.stdin.isatty() and not sys.stdin.closed:
        user_input = sys.stdin.read().strip()

    if input:
        if user_input:
            user_input += "\n\n"
        user_input += " ".join(input)

    return user_input


async def fzf_sessions(sessions_dir: str) -> str:
    # Find session files in the sessions directory
    session_files = sorted((f for f in os.listdir(sessions_dir) if os.path.isfile(os.path.join(sessions_dir, f)) and f.endswith(".json")), reverse=True)

    try:
        opts = ["--preview", "tclaude -p -s {}"]

        # If the user did not customize their fzf, we set some tclaude-specific defaults.
        if "FZF_DEFAULT_OPTS" not in os.environ:
            opts.extend(
                [
                    "--color=hl:12,hl+:12,prompt:5,query:7,pointer:5,info:244,spinner:5,header:7,marker:12",
                    "--bind=ctrl-d:preview-down,ctrl-u:preview-up",
                    "--prompt= ",
                    "--preview-window=60%",
                    "--height=40%",
                    "--layout=reverse",
                ]
            )

        process = await asyncio.create_subprocess_exec(
            "fzf",
            *opts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        output, error = await process.communicate(input="\n".join(session_files).encode())
        if error:
            print(f"Error running fzf: {error.decode().strip()}", file=sys.stderr)
            sys.exit(1)

        return output.decode().strip()
    except FileNotFoundError:
        print("fzf is not installed. Please install it to use session selection.", file=sys.stderr)
        sys.exit(1)


async def async_main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "Set the ANTHROPIC_API_KEY environment variable to your API key to use tclaude.\nYou can get an API key at https://console.anthropic.com/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    wrap_width = common.get_wrap_width()

    args = parse_tclaude_args()
    logging_config.setup(verbose=args.verbose)

    logger.debug(f"Logging setup complete: verbose={args.verbose}")

    if args.session == "fzf":
        args.session = await fzf_sessions(args.sessions_dir)

    config = load_config(args.config)

    history = common.load_session_if_exists(args.session, args.sessions_dir) if args.session else []
    if args.print_history:
        print(await history_to_string(history, pretty=True, wrap_width=wrap_width), flush=True)
        return

    user_input = read_user_input(args.input)

    # If stdout is not a terminal, execute in single prompt mode. No interactive chat; only print the response (not history)
    if not sys.stdout.isatty():
        if not user_input:
            print("No input provided.", file=sys.stderr)
            sys.exit(1)

        from . import chat

        await chat.single_prompt(args, config, history, user_input, print_text_only=True)
        return

    if history:
        print(await history_to_string(history, pretty=True, wrap_width=wrap_width), end="\n\n")

    # We print a decoy prompt to reduce the perceived startup delay. Importing .chat takes as much as hundreds of milliseconds (!), so we
    # want to show the user something immediately.
    if not user_input:
        common.print_decoy_prompt("", wrap_width)

    from . import chat

    await chat.chat(args, config, history, user_input)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
