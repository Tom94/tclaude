# tai -- Terminal AI
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
from .print import history_to_string, print_decoy_prompt


def main():
    logging.setup()

    if not "ANTHROPIC_API_KEY" in os.environ:
        print("Set the ANTHROPIC_API_KEY environment variable to your API key to use tai.", file=sys.stderr)
        print("You can get an API key at https://console.anthropic.com/settings/keys", file=sys.stderr)
        sys.exit(1)

    # If stdout is not a terminal, execute in prompt mode. No interactive chat; no progressive printing; no history.
    if not os.isatty(1):
        from . import prompt

        prompt.prompt(print_text_only=True)
        return

    args = common.parse_tai_args()
    history = common.load_session_if_exists(args.session, args.sessions_dir) if args.session else []
    if history:
        print(history_to_string(history, pretty=True, wrap_width=os.get_terminal_size().columns), end="\n\n")

    # We print a decoy prompt to reduce the perceived startup delay. Importing .chat takes as much as hundreds of milliseconds (!), so we
    # want to show the user something immediately.
    user_input = common.read_user_input(args.input)
    if not user_input:
        print_decoy_prompt("")

    from . import chat

    chat.chat(args, history, user_input)


if __name__ == "__main__":
    main()
