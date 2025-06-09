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

from __future__ import annotations

import os
import sys

import loguru
from loguru import logger

from .common import (
    ANSI_BOLD_BRIGHT_RED,
    ANSI_BOLD_YELLOW,
    ANSI_MID_GRAY,
    ANSI_RESET,
)


def get_log_dir() -> str:
    """
    Get the path to the configuration file.
    """
    if "XDG_STATE_HOME" in os.environ:
        config_dir = os.environ["XDG_STATE_HOME"]
    else:
        config_dir = os.path.join(os.path.expanduser("~"), ".local", "state")

    return os.path.join(config_dir, "tclaude")


did_print_since_prompt = False


def pprint(message: loguru.Message):
    global did_print_since_prompt
    did_print_since_prompt = True

    level = message.record["level"]
    prefix = f"\r\033[2K{ANSI_MID_GRAY}"
    if level.name == "SUCCESS":
        prefix += "[✓] "
    elif level.name == "WARNING":
        prefix += f"[{ANSI_BOLD_YELLOW}w{ANSI_MID_GRAY}] "
    elif level.name == "ERROR":
        prefix += f"[{ANSI_BOLD_BRIGHT_RED}e{ANSI_MID_GRAY}] "
    elif level.name == "CRITICAL":
        prefix += f"[{ANSI_BOLD_BRIGHT_RED}c{ANSI_MID_GRAY}] "
    print(f"{prefix}{message.record['message']}{ANSI_RESET}", file=sys.stdout if message.record["level"].no < 30 else sys.stderr)


def setup():
    """
    Set up the logging configuration for the application. This function configures the logger to print messages in a plain format.
    """
    logger.remove()  # Remove any existing handlers

    _ = logger.add(pprint, level="INFO", format=f"{ANSI_MID_GRAY}{{message}}{ANSI_RESET}", backtrace=False, diagnose=False)

    log_dir = get_log_dir()
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        logger.opt(exception=e).error(f"Error creating log directory {log_dir}: {e}")
        return

    _ = logger.add(
        os.path.join(log_dir, "tclaude.log"),
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
