#!/usr/bin/env python3

import os

import argparse


CHEVRON = ""


def ansi(cmd: str) -> str:
    return f"\033[{cmd}"


def wrap_style(msg: str, cmd: str, pretty=True) -> str:
    if pretty:
        return f"{ansi(cmd)}{msg}{ansi('0m')}"
    return msg


def prompt_style(msg: str) -> str:
    return wrap_style(msg, "0;35m")  # magenta


def input_style(msg: str) -> str:
    return wrap_style(msg, "1m")  # bold


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


def parse_args():
    default_role = os.path.join(get_config_dir(), "roles", "default.md")

    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("--sessions-dir", default=default_sessions_dir(), help="Path to directory for session files")
    parser.add_argument("-r", "--role", default=default_role, help="Path to a markdown file containing a system prompt")
    parser.add_argument("-m", "--model", default="claude-opus-4-0", help="Anthropic model to use")
    parser.add_argument("--max-tokens", type=int, default=2**14, help="Maximum number of tokens in the response")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability")
    parser.add_argument("--no-code-execution", action="store_true", help="Disable code execution capability")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    parser.add_argument("--thinking-budget", type=int, help="Number of tokens to allocate for thinking (min 1024)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
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


def word_wrap(text: str, wrap_width: int | None) -> str:
    if not text or wrap_width is None or wrap_width <= 0:
        return text

    lines = []

    for line in text.split("\n"):
        # Preserve empty lines
        if not line.strip():
            lines.append(line)
            continue

        # Detect indentation of the original line
        stripped_line = line.lstrip()
        indent = line[: len(line) - len(stripped_line)]

        # If the line fits within wrap_width, keep it as is
        if len(line) <= wrap_width:
            lines.append(line)
            continue

        # Wrap the line while preserving indentation
        current_line = []
        words = stripped_line.split()

        for word in words:
            # If a single word is longer than the available width, split it
            available_width = wrap_width - len(indent)
            if len(word) > available_width and available_width > 0:
                # First, add any current line content
                if current_line:
                    lines.append(indent + " ".join(current_line))
                    current_line = []

                # Split the long word into chunks
                while len(word) > available_width:
                    lines.append(indent + word[:available_width])
                    word = word[available_width:]

                # Add the remaining part of the word
                if word:
                    current_line = [word]
            else:
                test_line = " ".join(current_line + [word])
                if len(indent + test_line) > wrap_width and current_line:
                    lines.append(indent + " ".join(current_line))
                    current_line = [word]
                else:
                    current_line.append(word)

        if current_line:
            lines.append(indent + " ".join(current_line))

    return "\n".join(lines)


def bat_syntax_highlight(string: str, language: str, wrap_width: int | None = None) -> str:
    """
    Turn string pretty by piping it through bat
    """
    try:
        import subprocess

        command = ["bat", "--force-colorization", "--italic-text=always", "--paging=never", "--style=plain", f"--language={language}"]
        if wrap_width is not None:
            command.extend(["--wrap=character", f"--terminal-width={wrap_width}"])

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
    except FileNotFoundError:
        # If bat is not installed, fall back to regular print
        return string
