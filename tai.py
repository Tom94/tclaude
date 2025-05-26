#!/usr/bin/env python3

import os
import common

from print import history_to_string, print_decoy_prompt


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

    import chat

    chat.main_with_args_and_history(args, history)


if __name__ == "__main__":
    main()
