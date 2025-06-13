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

import atexit
import datetime
import json
import logging
import logging.config
import logging.handlers
import os
import sys
from typing import override

from .common import (
    ANSI_BOLD_BRIGHT_RED,
    ANSI_BOLD_PURPLE,
    ANSI_BOLD_YELLOW,
    ANSI_MID_GRAY,
    ANSI_RESET,
    get_state_dir,
)

logger = logging.getLogger(__package__)
did_print_since_prompt = False


class PrintFormatter(logging.Formatter):
    def __init__(self, *, verbose: bool):
        super().__init__()
        self.verbose: bool = verbose

    @override
    def format(self, record: logging.LogRecord) -> str:
        global did_print_since_prompt
        did_print_since_prompt = True

        level = record.levelno
        prefix = f"\r\033[2K{ANSI_MID_GRAY}"
        if self.verbose:
            td = datetime.timedelta(milliseconds=record.relativeCreated)
            prefix += f"[{td}] "

        if level == logging.DEBUG:
            prefix += f"[{ANSI_BOLD_PURPLE}d{ANSI_MID_GRAY}] "
        elif level == logging.WARNING:
            prefix += f"[{ANSI_BOLD_YELLOW}w{ANSI_MID_GRAY}] "
        elif level == logging.ERROR:
            prefix += f"[{ANSI_BOLD_BRIGHT_RED}e{ANSI_MID_GRAY}] "
        elif level == logging.CRITICAL:
            prefix += f"[{ANSI_BOLD_BRIGHT_RED}c{ANSI_MID_GRAY}] "

        return f"{prefix}{record.getMessage()}{ANSI_RESET}"


class StderrHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    @override
    def emit(self, record: logging.LogRecord):
        log_entry = self.format(record)
        print(log_entry, file=sys.stderr)


class JsonFormatter(logging.Formatter):
    def __init__(self, *, json_keys: dict[str, str]):
        super().__init__()
        self.json_keys: dict[str, str] = json_keys

    @override
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._to_dict(record), default=str)

    def _to_dict(self, record: logging.LogRecord) -> dict[str, str]:
        # First populate a few hardcoded fields, then add any additional fields specified in `json_keys`.
        result = {
            "message": record.getMessage(),
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
        }

        if record.exc_info:
            result["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            result["stack_info"] = self.formatStack(record.stack_info)

        result.update({json_key: getattr(record, msg_key) for json_key, msg_key in self.json_keys.items() if json_key not in result})
        return result


def setup(verbose: bool):
    """
    Set up the logging configuration for the application. This function configures the logger to print messages in a plain format.
    """

    log_dir = get_state_dir()
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory {log_dir}: {e}", file=sys.stderr)
        return

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "print": {
                    "()": f"{__package__}.logging_config.PrintFormatter",
                    "verbose": verbose,
                },
                "json": {
                    "()": f"{__package__}.logging_config.JsonFormatter",
                    "json_keys": {
                        "level": "levelname",
                        "message": "message",
                        "timestamp": "timestamp",
                        "relative_created": "relativeCreated",
                        "logger": "name",
                        "module": "module",
                        "function": "funcName",
                        "line": "lineno",
                        "thread_name": "threadName",
                    },
                },
            },
            "handlers": {
                "console": {
                    "class": f"{__package__}.logging_config.StderrHandler",
                    "formatter": "print",
                    "level": logging.DEBUG if verbose else logging.INFO,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging.DEBUG,
                    "filename": os.path.join(log_dir, f"{__package__}.log.jsonl"),
                    "formatter": "json",
                    "maxBytes": 10 * 1024 * 1024,  # 10 MB
                    "backupCount": 3,
                },
                "queue_handler": {
                    "class": "logging.handlers.QueueHandler",
                    "handlers": ["file"],
                    "respect_handler_level": True,
                },
            },
            "root": {
                "handlers": ["console", "queue_handler"],
                "level": logging.DEBUG,
            },
        }
    )

    queue_handler = logging.getHandlerByName("queue_handler")
    if isinstance(queue_handler, logging.handlers.QueueHandler) and queue_handler.listener is not None:
        queue_handler.listener.start()
        _ = atexit.register(queue_handler.listener.stop)
