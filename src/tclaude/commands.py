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
from collections.abc import Awaitable, Mapping
from functools import partial
from io import StringIO
from typing import Callable, TypeAlias

import aiohttp

from . import common, files
from .config import EndpointConfig, TClaudeConfig
from .json import get_or

CommandCallback: TypeAlias = Callable[[], Awaitable[None]]
Command: TypeAlias = Mapping[str, "Command"] | CommandCallback

logger = logging.getLogger(__package__)


async def print_help():
    help_io = StringIO()
    _ = help_io.write("Available commands:\n")
    _ = help_io.write("/help                  Print this help message\n")
    _ = help_io.write(f"/exit                  Quit {__package__}\n")
    _ = help_io.write("/download <filename>   Download a file generated by code execution\n")
    logger.info(common.word_wrap(help_io.getvalue().rstrip(), common.get_wrap_width()))


async def exit():
    raise EOFError


async def download_file(endpoint: EndpointConfig, file_id: str, file_path: str):
    async with aiohttp.ClientSession() as client:
        await files.download_file(client, endpoint, file_id, file_path)

    logger.info(f"[✓] Downloaded '{file_path}' (id={file_id})")


def get_commands(config: TClaudeConfig, uploaded_files: dict[str, common.FileMetadata]) -> dict[str, Command]:
    result: dict[str, Command] = {
        "/help": print_help,
        "/exit": exit,
        "/download": {},
    }

    endpoint = config.get_endpoint_config()

    if uploaded_files:
        file_commands: dict[str, Command] = {}
        for id, metadata in uploaded_files.items():
            if not metadata.get("downloadable", False):
                continue
            filename = get_or(metadata, "filename", "<unknown>")
            file_commands[filename] = partial(download_file, endpoint, id, filename)

        result["/download"] = file_commands

    return result


def get_callback(command: str, commands: Command) -> CommandCallback:
    parts = command.split(" ")
    callback = commands
    for p in parts:
        if not isinstance(callback, Mapping):
            raise ValueError(f"invalid argument '{p}'")
        if p not in callback:
            raise ValueError(f"unknown '{p}'")
        callback = callback[p]

    if isinstance(callback, Mapping):
        raise ValueError("missing argument")

    return callback
