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


import argparse
import logging
import os
import sys
import tomllib

from .json import JSON

logger = logging.getLogger(__package__)


def get_config_dir() -> str:
    """
    Get the path to the configuration file.
    """
    if "XDG_CONFIG_HOME" in os.environ:
        config_dir = os.environ["XDG_CONFIG_HOME"]
    else:
        config_dir = os.path.join(os.path.expanduser("~"), ".config")

    return os.path.join(config_dir, "tclaude")


def default_sessions_dir() -> str:
    """
    Get the default session directory.
    """
    if "TCLAUDE_SESSIONS_DIR" in os.environ:
        return os.environ["TCLAUDE_SESSIONS_DIR"]
    return "."


def load_system_prompt(path: str) -> str | None:
    system_prompt = None
    if not os.path.isfile(path):
        candidate = os.path.join(get_config_dir(), "roles", path)
        if os.path.isfile(candidate):
            path = candidate
    try:
        with open(path, "r") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        logger.exception(f"System prompt file {path} not found.")
    return system_prompt


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


class TClaudeArgs(argparse.Namespace):
    def __init__(self):
        super().__init__()

        default_role = os.path.join(get_config_dir(), "roles", "default.md")
        if not os.path.isfile(default_role):
            default_role = None

        self.input: list[str]

        self.config: str = "tclaude.toml"
        self.file: list[str] = []
        self.max_tokens: int = 2**14  # 16k tokens
        self.model: str = "claude-sonnet-4-0"
        self.no_code_execution: bool = False
        self.no_web_search: bool = False
        self.print_history: bool = False
        self.role: str | None = default_role
        self.session: str | None = None
        self.sessions_dir: str = default_sessions_dir()
        self.thinking: bool = False
        self.thinking_budget: int | None = None
        self.verbose: bool = False
        self.version: bool = False


def parse_tclaude_args():
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    _ = parser.add_argument("input", nargs="*", help="Input text to send to Claude")

    _ = parser.add_argument("--config", help="Path to the configuration file (default: tclaude.toml)")
    _ = parser.add_argument("-f", "--file", action="append", help="Path to a file that should be sent to Claude as input")
    _ = parser.add_argument("--max-tokens", help="Maximum number of tokens in the response (default: 16384)")
    _ = parser.add_argument("-m", "--model", help="Anthropic model to use (default: claude-sonnet-4-0)")
    _ = parser.add_argument("--no-code-execution", action="store_true", help="Disable code execution capability")
    _ = parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability")
    _ = parser.add_argument("-p", "--print_history", help="Print the conversation history only, without prompting.", action="store_true")
    _ = parser.add_argument("-r", "--role", help="Path to a markdown file containing a system prompt (default: default.md)")
    _ = parser.add_argument("-s", "--session", help="Path to session file for conversation history", nargs="?", const="fzf")
    _ = parser.add_argument("--sessions-dir", help="Path to directory for session files (default: current directory)")
    _ = parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    _ = parser.add_argument("--thinking-budget", help="Number of tokens to allocate for thinking (min 1024, default: half of max-tokens)")
    _ = parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    _ = parser.add_argument("-v", "--version", action="store_true", help="Print version information and exit")

    args = parser.parse_args(namespace=TClaudeArgs())
    if args.version:
        from . import __version__

        print(f"tclaude — Claude in the terminal\nversion {__version__}")
        sys.exit(0)

    args.model = deduce_model_name(args.model)
    return args


def load_config(filename: str | None) -> dict[str, JSON]:
    """
    Load the configuration from the tclaude.toml file located in the config directory.
    """
    if filename is None:
        filename = "tclaude.toml"

    if not os.path.isfile(filename):
        filename = os.path.join(get_config_dir(), filename)
        if not os.path.isfile(filename):
            logger.debug(f"Configuration file {filename} not found. Using default configuration.")

            from importlib import resources

            resources_path = resources.files(__package__)
            filename = str(resources_path.joinpath("default-config", "tclaude.toml"))

    try:
        with open(filename, "rb") as f:
            config = tomllib.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {filename}: {e}")
        return {}
