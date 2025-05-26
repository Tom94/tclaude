#!/usr/bin/env python3

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

from . import common
from .print import history_to_string, print_decoy_prompt


def main():
    """
    Print prompt and load/print history before importing any other modules to hide startup delay of slower imports.
    This makes a huge difference in perceived responsiveness when launching the interactive CLI.
    """
    if not os.isatty(0):
        print(f"chat.py should only be run in interactive mode. Use prompt.py otherwise.")
        exit(1)

    args = common.parse_args()
    history = common.load_session_if_exists(args.session, args.sessions_dir) if args.session else []
    if history:
        print(history_to_string(history, pretty=True, wrap_width=os.get_terminal_size().columns), end="\n\n")

    print_decoy_prompt()

    from . import chat

    chat.main_with_args_and_history(args, history)


if __name__ == "__main__":
    main()
