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

import argparse
import os
import sys
from typing import Callable, Optional

CHEVRON = ""
HELP_TEXT = "Type your message and hit Enter. Ctrl-D to exit, ESC for Vi mode, \\-Enter for newline."


def ansi(cmd: str) -> str:
    return f"\033[{cmd}"


ANSI_MID_GRAY = ansi("0;38;5;245m")
ANSI_BOLD_YELLOW = ansi("1;33m")
ANSI_BOLD_BRIGHT_RED = ansi("1;91m")


def wrap_style(msg: str, cmd: str, pretty=True) -> str:
    if pretty:
        return f"{ansi(cmd)}{msg}{ansi('0m')}"
    return msg


def prompt_style(msg: str) -> str:
    return wrap_style(msg, "0;35m")  # magenta


def gray_style(msg: str) -> str:
    return wrap_style(msg, "38;5;245m")  # gray


def input_style(msg: str) -> str:
    return wrap_style(msg, "1m")  # bold


def pplain(*args, **kwargs):
    """
    Print plain messages to stdout. Use this instead of print() to ensure that the output is formatted correctly.
    """
    print(gray_style(" ".join(map(str, args))), **kwargs)


def pinfo(*args, **kwargs):
    """
    Print info messages to stdout. Use this instead of print() to ensure that the output is formatted correctly.
    """
    pplain("[i]", *args, **kwargs)


def psuccess(*args, **kwargs):
    """
    Print success messages to stdout. Use this instead of print() to ensure that the output is formatted correctly.
    """
    pplain("[✓]", *args, **kwargs)


def pwarning(*args, **kwargs):
    """
    Print warning messages to stderr. Use this instead of print() to ensure that the output is formatted correctly.
    """
    pplain(f"[{ANSI_BOLD_YELLOW}w{ANSI_MID_GRAY}]", *args, **kwargs, file=sys.stderr)


def perror(*args, **kwargs):
    """
    Print error messages to stdout. Use this instead of print() to ensure that the output is formatted correctly.
    """
    pplain(f"[{ANSI_BOLD_BRIGHT_RED}e{ANSI_MID_GRAY}]", *args, **kwargs, file=sys.stderr)


def get_config_dir() -> str:
    """
    Get the path to the configuration file.
    """
    if "XDG_CONFIG_HOME" in os.environ:
        config_dir = os.environ["XDG_CONFIG_HOME"]
    else:
        config_dir = os.path.join(os.path.expanduser("~"), ".config")

    return os.path.join(config_dir, "tai")


def default_sessions_dir() -> str:
    """
    Get the default session directory.
    """
    if "TAI_SESSIONS_DIR" in os.environ:
        return os.environ["TAI_SESSIONS_DIR"]
    return "."


def load_session_if_exists(session_name: str, sessions_dir: str) -> list[dict]:
    import json

    if not session_name.lower().endswith(".json"):
        session_name += ".json"

    if not os.path.isfile(session_name):
        candidate = os.path.join(sessions_dir, session_name)
        if os.path.isfile(candidate):
            session_name = candidate
        else:
            return []

    history = []
    try:
        with open(session_name, "r") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not parse session file {session_name}. Starting new session.")

    return history


def load_system_prompt(path: str) -> Optional[str]:
    system_prompt = None
    try:
        with open(path, "r") as f:
            system_prompt = f.read().strip()
    except Exception as e:
        print(f"Error reading system prompt file: {e}")
    return system_prompt


def parse_args():
    default_role = os.path.join(get_config_dir(), "roles", "default.md")
    if not os.path.isfile(default_role):
        default_role = None

    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("--sessions-dir", default=default_sessions_dir(), help="Path to directory for session files")
    parser.add_argument("-r", "--role", default=default_role, help="Path to a markdown file containing a system prompt")
    parser.add_argument("-m", "--model", default="claude-sonnet-4-0", help="Anthropic model to use")
    parser.add_argument("--max-tokens", type=int, default=2**14, help="Maximum number of tokens in the response")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability")
    parser.add_argument("--no-code-execution", action="store_true", help="Disable code execution capability")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    parser.add_argument("--thinking-budget", type=int, help="Number of tokens to allocate for thinking (min 1024)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-v", "--version", action="store_true", help="Print version information and exit")

    args = parser.parse_args()
    if args.version:
        from . import __version__

        print(f"tai — Terminal AI\nversion {__version__}")
        sys.exit(0)

    args.model = deduce_model_name(args.model)
    return args


def deduce_model_name(model: str) -> str:
    if "opus" in model:
        if "3" in model:
            return "claude-3-opus-latest"
        return "claude-opus-4-0"
    elif "sonnet" in model:
        if "3.5" in model:
            return "claude-3-5-sonnet-latest"
        elif "3.7" in model:
            return "claude-3-7-sonnet-latest"
        elif "3" in model:
            return "claude-3-sonnet-latest"
        return "claude-sonnet-4-0"
    elif "haiku" in model:
        return "claude-3-5-haiku-latest"
    return model


def friendly_model_name(model: str) -> str:
    """
    Convert a model name to a more user-friendly format.
    """
    if not model.startswith("claude-"):
        return model

    kind = None
    if "opus" in model:
        kind = "opus"
    elif "sonnet" in model:
        kind = "sonnet"
    elif "haiku" in model:
        kind = "haiku"

    if kind is None:
        return model

    # Double-digit versions first, then single-digit
    version = None
    if "3-7" in model:
        version = "3.7"
    elif "3-5" in model:
        version = "3.5"
    elif "3" in model:
        version = "3.0"
    elif "4" in model:
        version = "4.0"

    return f"{kind} {version}"


def make_check_bat_available() -> Callable[[], bool]:
    is_bat_available = None

    def check_bat_available() -> bool:
        nonlocal is_bat_available
        if is_bat_available is None:
            import subprocess

            try:
                subprocess.run(["bat", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                is_bat_available = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                is_bat_available = False
                pwarning("Install `bat` (https://github.com/sharkdp/bat) to enable syntax highlighting.")

        return is_bat_available

    return check_bat_available


check_bat_available = make_check_bat_available()


def syntax_highlight(string: str, language: str) -> str:
    """
    Turn string pretty by piping it through bat
    """

    if not check_bat_available():
        return string

    import subprocess

    command = ["bat", "--force-colorization", "--italic-text=always", "--paging=never", "--style=plain", f"--language={language}"]

    # Use bat to pretty print the string
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = process.communicate(input=string.encode("utf-8"))

    if process.returncode != 0:
        raise Exception(f"Error: {error.decode('utf-8')}")
    return output.decode("utf-8")


def char_wrap(text: str, wrap_width: int) -> str:
    """
    Wrap text by characters instead of words, preserving indentation.
    """
    if not text or wrap_width is None or wrap_width <= 0:
        return text

    from wcwidth import wcswidth

    def get_width(s):
        """Get display width of string, fallback to len if wcswidth returns None"""
        width = wcswidth(s)
        return width if width is not None else len(s)

    lines = []

    for line in text.split("\n"):
        # Preserve empty lines
        if not line.strip():
            lines.append(line)
            continue

        # Detect indentation of the original line
        stripped_line = line.lstrip()
        indent = line[: len(line) - len(stripped_line)]
        indent_width = get_width(indent)

        # If the line fits within wrap_width, keep it as is
        if get_width(line) <= wrap_width:
            lines.append(line)
            continue

        # Wrap the line by characters while preserving indentation
        current_chunk = ""
        current_width = indent_width

        for char in stripped_line:
            char_width = get_width(char)
            if current_width + char_width > wrap_width and current_chunk:
                lines.append(indent + current_chunk)
                current_chunk = char
                current_width = indent_width + char_width
            else:
                current_chunk += char
                current_width += char_width

        if current_chunk:
            lines.append(indent + current_chunk)

    return "\n".join(lines)


def word_wrap(text: str, wrap_width: int) -> str:
    if not text or wrap_width is None or wrap_width <= 0:
        return text

    from wcwidth import wcswidth

    def get_width(s):
        """Get display width of string, fallback to len if wcswidth returns None"""
        width = wcswidth(s)
        return width if width is not None else len(s)

    lines = []

    for line in text.split("\n"):
        # Preserve empty lines
        if not line.strip():
            lines.append(line)
            continue

        # Detect indentation of the original line
        stripped_line = line.lstrip()
        indent = line[: len(line) - len(stripped_line)]
        indent_width = get_width(indent)

        # If the line fits within wrap_width, keep it as is
        if get_width(line) <= wrap_width:
            lines.append(line)
            continue

        # Wrap the line while preserving indentation
        current_line = []
        words = stripped_line.split()

        for word in words:
            word_width = get_width(word)
            # If a single word is longer than the available width, split it
            available_width = wrap_width - indent_width
            if word_width > available_width and available_width > 0:
                # First, add any current line content
                if current_line:
                    lines.append(indent + " ".join(current_line))
                    current_line = []

                # Split the long word into chunks by character
                current_chunk = ""
                current_chunk_width = 0

                for char in word:
                    char_width = get_width(char)
                    if current_chunk_width + char_width > available_width and current_chunk:
                        lines.append(indent + current_chunk)
                        current_chunk = char
                        current_chunk_width = char_width
                    else:
                        current_chunk += char
                        current_chunk_width += char_width

                # Add the remaining part of the word
                if current_chunk:
                    current_line = [current_chunk]
            else:
                test_line = " ".join(current_line + [word])
                test_line_width = get_width(indent + test_line)
                if test_line_width > wrap_width and current_line:
                    lines.append(indent + " ".join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)

        if current_line:
            lines.append(indent + " ".join(current_line))

    return "\n".join(lines)
