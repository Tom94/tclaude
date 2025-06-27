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

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeAlias, cast

from .json import JSON, get, get_or_default, of_type_or_none

logger = logging.getLogger(__package__)

History: TypeAlias = list[dict[str, JSON]]
FileMetadata: TypeAlias = dict[str, JSON]


CHEVRON = ""
CHEVRON_CONTINUATION = "·"
HELP_TEXT = "Type your message and hit Enter. Ctrl-D to exit, ESC for Vi mode, \\-Enter for newline, /help for commands."


def ansi(cmd: str) -> str:
    return f"\033[{cmd}"


ANSI_HIDE_CURSOR = ansi("?25l")
ANSI_SHOW_CURSOR = ansi("?25h")
ANSI_BEAM_CURSOR = ansi("6 q")

ANSI_MID_GRAY = ansi("0;38;5;245m")
ANSI_BOLD_YELLOW = ansi("1;33m")
ANSI_BOLD_PURPLE = ansi("1;35m")
ANSI_BOLD_CYAN = ansi("1;36m")
ANSI_BOLD_BRIGHT_RED = ansi("1;91m")
ANSI_RESET = ansi("0m")
ANSI_BEGINNING_OF_LINE = ansi("1G")


def print_decoy_prompt(user_input: str, wrap_width: int):
    """
    Reproduce the initial prompt that prompt_toolkit will produce. Requires a bit of bespoke formatting to match exactly.
    """
    if user_input:
        print(f"{prompt_style(CHEVRON)} {user_input}")
        return

    initial_prompt = char_wrap(f"  {HELP_TEXT}", wrap_width - 2)
    num_newlines = initial_prompt.count("\n")
    ansi_return: str = "\033[F" * num_newlines + ansi("3G")
    print(f"{prompt_style(CHEVRON)} {gray_style(initial_prompt[2:])}{ansi_return}", end="", flush=True)


def get_wrap_width() -> int:
    if sys.stdout.isatty():
        return os.get_terminal_size().columns
    if "FZF_PREVIEW_COLUMNS" in os.environ:
        try:
            return int(os.environ["FZF_PREVIEW_COLUMNS"])
        except ValueError:
            pass
    return -1  # No wrapping in non-TTY environments


def wrap_style(msg: str, cmd: str, pretty: bool = True) -> str:
    if pretty:
        return f"{cmd}{msg}{ANSI_RESET}"
    return msg


def prompt_style(msg: str) -> str:
    return wrap_style(msg, ansi("0;35m"))  # magenta


def gray_style(msg: str) -> str:
    return wrap_style(msg, ANSI_MID_GRAY)


def input_style(msg: str) -> str:
    return wrap_style(msg, ansi("1m"))  # bold


def escape(text: str) -> str:
    return repr(text.strip().replace("\n", " ").replace("\r", "").replace("\t", " "))


def get_cache_dir() -> str:
    """
    Get the path to the cache directory.
    """
    if "XDG_CACHE_HOME" in os.environ:
        cache_dir = os.environ["XDG_CACHE_HOME"]
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache")

    return os.path.join(cache_dir, "tclaude")


def get_state_dir() -> str:
    """
    Get the path to the configuration file.
    """
    if "XDG_STATE_HOME" in os.environ:
        config_dir = os.environ["XDG_STATE_HOME"]
    else:
        config_dir = os.path.join(os.path.expanduser("~"), ".local", "state")

    return os.path.join(config_dir, "tclaude")


@dataclass
class Container:
    id: str
    expires_at: datetime


def get_latest_container(messages: History) -> Container | None:
    """
    Get the latest container from the messages history.
    Returns None if no container is found.
    """
    for message in reversed(messages):
        if "container" in message:
            container_data = message["container"]
            id = get(container_data, "id", str)
            expires_at = get(container_data, "expires_at", str)
            if id is None or expires_at is None:
                continue

            expires_at = datetime.fromisoformat(expires_at)

            # Be conservative. If the container is just 1m from expiring, don't use it anymore.
            if expires_at < datetime.now(timezone.utc) + timedelta(minutes=1):
                continue

            return Container(id=id, expires_at=expires_at)

    return None


def is_valid_metadata(metadata: dict[str, JSON]) -> bool:
    return get(metadata, "id", str) is not None


def get_uploaded_files(messages: History) -> dict[str, FileMetadata]:
    """
    Extract uploaded files from history.
    """
    uploaded_files: dict[str, FileMetadata] = {}

    for message in messages:
        for content_block in get_or_default(message, "content", list[JSON]):
            match content_block:
                case {"type": "container_upload", "file_id": str(file_id)} | {
                    "type": "document" | "image",
                    "source": {"file_id": str(file_id)},
                }:
                    # This removes the _input_pending flag if it exists, which is intended because this user message *is* the input.
                    uploaded_files[file_id] = {}
                case {"type": "tool_result", "content": list(tool_content)}:
                    for c in tool_content:
                        match c:
                            case {"type": "image", "source": {"file_id": str(file_id)}}:
                                uploaded_files[file_id] = {}
                            case _:
                                pass
                    pass
                case {"type": "code_execution_tool_result", "content": {"type": "code_execution_result", "content": list(code_exec_content)}}:
                    for c in code_exec_content:
                        match c:
                            case {"type": "code_execution_output", "file_id": str(file_id)}:
                                # Code-execution outputs need to be re-input by the user to make them available to the model.
                                uploaded_files[file_id] = {"_input_pending": True}
                            case _:
                                # TODO: handle other code execution content types
                                pass
                case _:
                    pass

    return uploaded_files


def get_user_messages(history: History) -> list[str]:
    """
    Extract user messages from history.
    """
    user_messages: list[str] = []
    for message in history:
        if get(message, "role", str) != "user":
            continue

        for content_block in get_or_default(message, "content", list[JSON]):
            match content_block:
                case {"type": "text", "text": str(text)}:
                    user_messages.append(text)
                case {"type": "container_upload", "file_id": str()} | {
                    "type": "document" | "image",
                    "source": {"file_id": str()},
                }:
                    pass
                case {"type": "tool_result"}:
                    pass
                case _:
                    logger.warning(f"Unknown content block type in user message: {content_block}")

    return user_messages


def load_session_if_exists(session_name: str, sessions_dir: str) -> History:
    import json

    if not session_name.lower().endswith(".json"):
        session_name += ".json"

    if not os.path.isfile(session_name):
        candidate = os.path.join(sessions_dir, session_name)
        if os.path.isfile(candidate):
            session_name = candidate
        else:
            return []

    history: History = []
    try:
        with open(session_name, "r") as f:
            j = cast(JSON, json.load(f))
            j = of_type_or_none(History, j)
            if j is not None:
                history = j
            else:
                logger.error(f"Session file {session_name} does not contain a valid history (expected a list of dicts).")
    except json.JSONDecodeError:
        logger.exception(f"Could not parse session file {session_name}. Starting new session.")

    return history


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


def make_check_bat_available() -> Callable[[], tuple[bool, list[str]]]:
    is_bat_available = None
    default_opts: list[str] = []

    def check_bat_available() -> tuple[bool, list[str]]:
        nonlocal is_bat_available

        if is_bat_available is None:
            # Default to vs code dark theme if no theme set
            if "BAT_THEME_DARK" not in os.environ:
                default_opts.extend(["--theme-dark", "Visual Studio Dark+"])

            # Nested calls to bat don't allow bat to detect the terminal theme, so we set it explicitly to dark mode if not set system-wide.
            if "BAT_THEME" not in os.environ:
                default_opts.extend(["--theme", "dark"])

            import subprocess

            try:
                _ = subprocess.run(["bat", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                is_bat_available = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                is_bat_available = False
                logger.warning("Install `bat` (https://github.com/sharkdp/bat) to enable syntax highlighting.")

        return is_bat_available, default_opts

    return check_bat_available


check_bat_available = make_check_bat_available()


async def syntax_highlight(string: str, language: str) -> str:
    """
    Turn string pretty by piping it through bat
    """

    is_available, default_opts = check_bat_available()
    if not is_available:
        return string

    import asyncio
    import subprocess

    command = ["bat", "--force-colorization", "--italic-text=always", "--paging=never", "--style=plain", f"--language={language}"]
    command.extend(default_opts)

    # Use bat to pretty print the string. Spawn in new process group to avoid issues with Ctrl-C handling.
    if sys.platform == "win32":
        process = await asyncio.create_subprocess_exec(
            command[0],
            *command[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        process = await asyncio.create_subprocess_exec(
            command[0],
            *command[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid,
        )

    output, error = await process.communicate(input=string.encode("utf-8"))
    if process.returncode != 0:
        raise Exception(f"Error: {error.decode('utf-8')}")
    return output.decode("utf-8")


def char_wrap(text: str, wrap_width: int) -> str:
    """
    Wrap text by characters instead of words, preserving indentation.
    """
    if not text or wrap_width <= 0:
        return text

    from wcwidth import wcswidth  # pyright: ignore

    lines: list[str] = []

    for line in text.split("\n"):
        # Preserve empty lines
        if not line.strip():
            lines.append(line)
            continue

        # Detect indentation of the original line
        stripped_line = line.lstrip()
        indent = line[: len(line) - len(stripped_line)]
        indent_width = wcswidth(indent)

        # If the line fits within wrap_width, keep it as is
        if wcswidth(line) <= wrap_width:
            lines.append(line)
            continue

        # Wrap the line by characters while preserving indentation
        current_chunk = ""
        current_width = indent_width

        for char in stripped_line:
            char_width = wcswidth(char)
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
    from wcwidth import wcswidth  # pyright: ignore

    wrap_indicator = " ↩"
    wrap_width -= wcswidth(wrap_indicator)
    if not text or wrap_width <= 0:
        return text

    lines: list[str] = []

    for line in text.split("\n"):
        # Preserve empty lines
        if not line.strip():
            lines.append(line)
            continue

        # Detect indentation of the original line
        stripped_line = line.lstrip()
        indent = line[: len(line) - len(stripped_line)]
        indent_width = wcswidth(indent)

        # If the line fits within wrap_width, keep it as is
        if wcswidth(line) <= wrap_width:
            lines.append(line)
            continue

        # Wrap the line while preserving indentation
        current_line = []
        words = stripped_line.split()

        for word in words:
            word_width = wcswidth(word)
            # If a single word is longer than the available width, split it
            available_width = wrap_width - indent_width
            if word_width > available_width and available_width > 0:
                # First, add any current line content
                if current_line:
                    lines.append(f"{indent}{' '.join(current_line)}{wrap_indicator}")
                    current_line = []

                # Split the long word into chunks by character
                current_chunk = ""
                current_chunk_width = 0

                for char in word:
                    char_width = wcswidth(char)
                    if current_chunk_width + char_width > available_width and current_chunk:
                        lines.append(f"{indent}{current_chunk}{wrap_indicator}")
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
                test_line_width = wcswidth(indent + test_line)
                if test_line_width > wrap_width and current_line:
                    lines.append(f"{indent}{' '.join(current_line)}{wrap_indicator}")
                    current_line = [word]
                else:
                    current_line.append(word)

        if current_line:
            lines.append(indent + " ".join(current_line))

    return "\n".join(lines)
