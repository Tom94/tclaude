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

import os
import sys

from . import common, logging
from .config import parse_tclaude_args, load_config
from .print import history_to_string, print_decoy_prompt


def read_user_input(input: list[str]) -> str:
    user_input = ""
    if not sys.stdin.isatty() and not sys.stdin.closed:
        user_input = sys.stdin.read().strip()

    if input:
        if user_input:
            user_input += "\n\n"
        user_input += " ".join(input)

    return user_input


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "Set the ANTHROPIC_API_KEY environment variable to your API key to use tclaude.\nYou can get an API key at https://console.anthropic.com/settings/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_tclaude_args()
    logging.setup(verbose=args.verbose)

    config = load_config(args.config)

    history = common.load_session_if_exists(args.session, args.sessions_dir) if args.session else []
    user_input = read_user_input(args.input)

    # If stdout is not a terminal, execute in single prompt mode. No interactive chat; only print the response (not history)
    if not os.isatty(1):
        if not user_input:
            print("No input provided.", file=sys.stderr)
            sys.exit(1)

        from . import chat

        chat.single_prompt(args, config, history, user_input, print_text_only=True)
        return

    if history:
        print(history_to_string(history, pretty=True, wrap_width=os.get_terminal_size().columns), end="\n\n")

    # We print a decoy prompt to reduce the perceived startup delay. Importing .chat takes as much as hundreds of milliseconds (!), so we
    # want to show the user something immediately.
    if not user_input:
        print_decoy_prompt("")

    from . import chat

    chat.chat(args, config, history, user_input)


if __name__ == "__main__":
    main()
