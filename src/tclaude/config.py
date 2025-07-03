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
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import cast, get_args, get_origin

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


def load_system_prompt(path: str) -> str | None:
    system_prompt = None
    if not os.path.isfile(path):
        candidate = os.path.join(get_config_dir(), "roles", path)
        if os.path.isfile(candidate):
            path = candidate
        elif path == "default.md":
            return None  # Be silent if the default prompt is not found.

    try:
        with open(path, "r") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        logger.exception(f"System prompt file {path} not found.")
    return system_prompt


class TClaudeArgs(argparse.Namespace):
    def __init__(self):
        super().__init__()

        self.input: list[str]

        self.config: str = "tclaude.toml"
        self.version: bool = False
        self.print_history: bool = False

        # Configuration overrides (default values are set in TClaudeConfig)
        self.endpoint: str | None = None
        self.file: list[str] = []
        self.max_tokens: int | None = None
        self.model: str | None = None
        self.no_code_execution: bool | None = None
        self.no_web_search: bool | None = None
        self.role: str | None = None
        self.session: str | None = None
        self.sessions_dir: str | None = None
        self.thinking: bool | None = None
        self.thinking_budget: int | None = None
        self.verbose: bool | None = None


def parse_tclaude_args():
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    _ = parser.add_argument("input", nargs="*", help="Input text to send to Claude")

    _ = parser.add_argument("--config", help="Path to the configuration file (default: tclaude.toml)")
    _ = parser.add_argument("-e", "--endpoint", help="Endpoint to use for the API (default: anthropic). Custom endpoints can be defined in the config file.")
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
class EndpointConfig:
    kind: str  # "anthropic" or "vertex"
    url: str
    api_key: str | None = None

    max_tokens: int | None = None
    model: str | None = None
    code_execution: bool | None = None
    web_search: bool | None = None
    thinking: bool | None = None
    thinking_budget: int | str | None = None

    def __post_init__(self):
        if self.kind not in ("anthropic", "vertex"):
            raise ValueError(f"Invalid endpoint kind: {self.kind}. Expected 'anthropic' or 'vertex'.")

        if self.api_key and self.api_key.startswith("$"):
            tmp = self.api_key[1:]
            if tmp.startswith("(") and tmp.endswith(")"):
                import subprocess

                self.api_key = subprocess.check_output(tmp[1:-1], shell=True).decode("utf-8").strip()
            else:
                if tmp.startswith("{") and tmp.endswith("}"):
                    tmp = tmp[1:-1]

                if tmp not in os.environ:
                    # Better error message for beginners that try to use the default config without having set the environment variable.
                    if self.url == "https://api.anthropic.com/v1" and tmp == "ANTHROPIC_API_KEY":
                        raise ValueError(
                            "Set the ANTHROPIC_API_KEY environment variable to your API key to use tclaude.\nYou can get an API key at https://console.anthropic.com/settings/keys"
                        )

                    raise ValueError(f"API key environment variable '{tmp}' not set.")

                self.api_key = os.getenv(tmp)

    def inherit_prompt_settings(self, config: "TClaudeConfig"):
        if self.max_tokens is None:
            self.max_tokens = config.max_tokens
        if self.model is None:
            self.model = config.model
        if self.code_execution is None:
            self.code_execution = config.code_execution
        if self.web_search is None:
            self.web_search = config.web_search
        if self.thinking is None:
            self.thinking = config.thinking
        if self.thinking_budget is None:
            self.thinking_budget = config.thinking_budget


@dataclass
class TClaudeConfig:
    # Prompt settings (default values that can be overridden by the endpoint config)
    max_tokens: int
    model: str
    role: str

    code_execution: bool
    web_search: bool
    thinking: bool
    thinking_budget: int | str

    # Endpoint config (each with their own prompt settings)
    endpoint: str
    endpoints: dict[str, EndpointConfig]

    # Global settings
    sessions_dir: str
    mcp: McpConfig = field(default_factory=McpConfig)

    # Expected to come from args, but can *technically* be set in the config file.
    files: list[str] = field(default_factory=list)
    session: str | None = None
    verbose: bool = False

    def get_thinking_budget(self) -> int:
        if isinstance(self.thinking_budget, str):
            if self.thinking_budget == "auto":
                if self.max_tokens < 1024:
                    raise ValueError("Auto thinking budget requires max_tokens to be at least 1024.")
                return self.max_tokens // 2
            else:
                raise ValueError(f"Invalid thinking budget: {self.thinking_budget}. Expected 'auto' or an integer.")

        return self.thinking_budget

    def get_endpoint_config(self) -> EndpointConfig:
        if self.endpoint not in self.endpoints:
            raise ValueError(f"Endpoint '{self.endpoint}' not found in configuration. Available endpoints: {list(self.endpoints.keys())}")

        return self.endpoints[self.endpoint]

    def apply_args_override(self, args: TClaudeArgs):
        if args.max_tokens is not None:
            self.max_tokens = args.max_tokens
        if args.model is not None:
            self.model = args.model
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

        if args.endpoint is not None:
            self.endpoint = args.endpoint

        if self.endpoint not in self.endpoints:
            raise ValueError(f"Endpoint '{self.endpoint}' not found in configuration. Available endpoints: {list(self.endpoints.keys())}")

        if args.sessions_dir is not None:
            self.sessions_dir = args.sessions_dir

        self.files.extend(args.file)
        if args.session is not None:
            self.session = args.session
        if args.verbose is not None:
            self.verbose = args.verbose

    def finalize(self):
        self.role = os.path.expanduser(self.role)
        self.sessions_dir = os.path.expanduser(self.sessions_dir)

        self.files = [os.path.expanduser(f) for f in self.files]
        self.session = os.path.expanduser(self.session) if self.session else None

        endpoint = self.get_endpoint_config()
        if endpoint.max_tokens is not None:
            self.max_tokens = endpoint.max_tokens
        if endpoint.model is not None:
            self.model = endpoint.model

        if endpoint.code_execution is not None:
            self.code_execution = endpoint.code_execution
        if endpoint.web_search is not None:
            self.web_search = endpoint.web_search
        if endpoint.thinking is not None:
            self.thinking = endpoint.thinking
        if endpoint.thinking_budget is not None:
            self.thinking_budget = endpoint.thinking_budget


def dataclass_from_dict[T](cls: type[T], data: JSON, name: str = "config") -> T:
    if not is_dataclass(cls):
        origin = get_origin(cls)
        args = get_args(cls)
        if origin is list:
            if not isinstance(data, list):
                raise ValueError(f"{name} must be a list of type {cls}, got {type(data)}")

            U = cast(type, args[0])
            return cast(T, [dataclass_from_dict(U, item, f"{name}[{i}]") for i, item in enumerate(data)])
        elif origin is dict:
            if not isinstance(data, dict):
                raise ValueError(f"{name} must be a dict of type {cls}, got {type(data)}")

            if args[0] is not str:
                raise ValueError(f"{name} must be a dict with string keys, got {args[0]}")

            V = cast(type, args[1])
            return cast(T, {k: dataclass_from_dict(V, v, f"{name}.{k}") for k, v in data.items()})

        if not generic_is_instance(data, cls):
            raise ValueError(f"{name} must be of type {cls}, got {type(data)}")

        return cast(T, data)

    assert is_dataclass(cls), f"Expected a dataclass type, got {cls}"
    if not isinstance(data, dict) or not generic_is_instance(data, dict[str, JSON]):
        raise ValueError(f"Must be of type dict, got {type(data)}")

    result = {}
    for f in fields(cls):
        if f.name not in data and f.default == MISSING and f.default_factory == MISSING:
            raise ValueError(f"{name}: missing required field '{f.name}'")

        if f.name in data:
            if isinstance(f.type, str):
                raise TypeError(f"field '{f.name}' has an invalid type: {f.type}")

            nested_cls = f.type

            value = data.pop(f.name)
            nested_name = f"{name}.{f.name}" if name else f.name
            result[f.name] = dataclass_from_dict(nested_cls, value, nested_name)

    if data:
        extra_keys = ", ".join(data.keys())
        raise ValueError(f"unexpected variables: {extra_keys}")

    return cls(**result)


def deep_update(d: dict[str, JSON], u: Mapping[str, JSON]) -> dict[str, JSON]:
    for k, v in u.items():
        if isinstance(v, Mapping) and (dv := d.get(k)) and isinstance(dv, dict):
            d[k] = deep_update(dv, v)
        else:
            d[k] = v

    return d


def load_config(filename: str) -> TClaudeConfig | None:
    """
    Load the configuration from the tclaude.toml file located in the config directory.
    """

    try:
        from importlib import resources

        resources_path = resources.files(__package__)
        default_config_filename = str(resources_path.joinpath("default-config", "tclaude.toml"))
        logger.debug(f"Loading default config from {default_config_filename}")
        with open(default_config_filename, "rb") as f:
            config_dict = tomllib.load(f)

        if not os.path.isfile(filename):
            filename = os.path.join(get_config_dir(), filename)

        if os.path.isfile(filename):
            logger.debug(f"Loading user configuration from {filename}")
            with open(filename, "rb") as f:
                user_config_dict = tomllib.load(f)

            config_dict = deep_update(config_dict, user_config_dict)

        return dataclass_from_dict(TClaudeConfig, config_dict)
    except FileNotFoundError as e:
        logger.error(f"Failed to load {filename}: {e}")
    except (tomllib.TOMLDecodeError, ValueError) as e:
        logger.error(f"{filename} is invalid: {e}")

    return None
