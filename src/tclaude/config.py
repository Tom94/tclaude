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
from dataclasses import dataclass, field, fields, is_dataclass
from typing import cast

from .json import JSON, generic_is_instance

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


def default_role() -> str | None:
    default_role = os.path.join(get_config_dir(), "roles", "default.md")
    if not os.path.isfile(default_role):
        default_role = None

    return default_role


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

        self.input: list[str]

        self.config: str = "tclaude.toml"
        self.version: bool = False
        self.verbose: bool | None = None

        # Configuration overrides (default values are set in TClaudeConfig)
        self.file: list[str] = []
        self.max_tokens: int | None = None
        self.model: str | None = None
        self.no_code_execution: bool | None = None
        self.no_web_search: bool | None = None
        self.print_history: bool | None = None
        self.role: str | None = None
        self.session: str | None = None
        self.sessions_dir: str | None = None
        self.thinking: bool | None = None
        self.thinking_budget: int | None = None


def parse_tclaude_args():
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    _ = parser.add_argument("input", nargs="*", help="Input text to send to Claude")

    _ = parser.add_argument("--config", help="Path to the configuration file (default: tclaude.toml)")
    _ = parser.add_argument("-f", "--file", action="append", help="Path to a file that should be sent to Claude as input")
    _ = parser.add_argument("--max-tokens", help="Maximum number of tokens in the response (default: 16384)")
    _ = parser.add_argument("-m", "--model", help="Anthropic model to use (default: claude-sonnet-4-20250514)")
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

    return args


@dataclass
class McpConfig:
    local_servers: list[dict[str, JSON]] = field(default_factory=list)
    remote_servers: list[dict[str, JSON]] = field(default_factory=list)


@dataclass
class TClaudeConfig:
    max_tokens: int = 2**14  # 16k tokens
    model: str = "claude-sonnet-4-20250514"
    role: str | None = default_role()

    code_execution: bool = True
    web_search: bool = True
    thinking: bool = False
    thinking_budget: int | None = None

    sessions_dir: str = default_sessions_dir()

    mcp: McpConfig = field(default_factory=McpConfig)

    # Expected to come from args, but can *technically* be set in the config file.
    files: list[str] = field(default_factory=list)
    session: str | None = None
    verbose: bool = False

    def apply_args_override(self, args: TClaudeArgs):
        if args.max_tokens is not None:
            self.max_tokens = args.max_tokens
        if args.model is not None:
            self.model = deduce_model_name(args.model)
        if args.role is not None:
            self.role = args.role

        if args.no_code_execution is not None:
            self.code_execution = not args.no_code_execution
        if args.no_web_search is not None:
            self.web_search = not args.no_web_search
        if args.thinking is not None:
            self.thinking = args.thinking
        if args.thinking_budget is not None:
            self.thinking_budget = args.thinking_budget

        if args.sessions_dir is not None:
            self.sessions_dir = args.sessions_dir

        self.files.extend(args.file)
        if args.session is not None:
            self.session = args.session
        if args.verbose is not None:
            self.verbose = args.verbose


def dataclass_from_dict[T](cls: type[T], data: dict[str, JSON]) -> T:
    assert is_dataclass(cls), f"Expected a dataclass type, got {cls}"

    result = cls()
    for f in fields(cls):
        if f.name in data:
            if isinstance(f.type, str):
                raise TypeError(f"field '{f.name}' has an invalid type: {f.type}")

            nested_cls = f.type
            value = data.pop(f.name)

            if is_dataclass(f.type):
                if not generic_is_instance(value, dict[str, JSON]):
                    raise ValueError(f"'{f.name}' must be a dict defining type {f.type}, got {type(value)}")

                setattr(result, f.name, dataclass_from_dict(nested_cls, cast(dict[str, JSON], value)))
            else:
                if not generic_is_instance(value, nested_cls):
                    raise ValueError(f"'{f.name}' must be of type {f.type}, got {type(value)}")

                setattr(result, f.name, value)

    if data:
        extra_keys = ", ".join(data.keys())
        raise ValueError(f"unexpected variables: {extra_keys}")

    return result


def load_config(filename: str | None) -> TClaudeConfig:
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
            config = dataclass_from_dict(TClaudeConfig, tomllib.load(f))
        return config
    except FileNotFoundError as e:
        logger.error(f"Failed to load {filename}: {e}")
    except (tomllib.TOMLDecodeError, ValueError) as e:
        logger.error(f"{filename} is invalid: {e}")

    return TClaudeConfig()
