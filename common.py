#!/usr/bin/env python3

import os

import argparse


CHEVRON = ""


def ansi(cmd: str) -> str:
    return f"\033[{cmd}"


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


def pretty_print_md(string: str, wrap_width: int | None = None) -> str:
    """
    Turn string pretty by piping it through bat
    """
    try:
        import subprocess

        command = ["bat", "--force-colorization", "--italic-text=always", "--paging=never", "--style=plain", "--language=markdown"]
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
