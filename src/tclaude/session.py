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

import asyncio
import datetime
import logging
import os

import aiofiles.os
import aiohttp
from humanize import naturalsize

from . import common, files, prompt
from .common import History, is_valid_metadata
from .event import AsyncEventEmitter
from .json import JSON, get, get_or_default
from .print import history_to_string
from .token_counter import TokenCounter

logger = logging.getLogger(__package__)


class ChatSession:
    """
    Represents a chat session with a model.
    Contains the session name and the history of messages.
    """

    def __init__(self, history: History, model: str, system_prompt: str | None = None, role: str | None = None, name: str | None = None):
        self.history: History = history
        self.model: str = model
        self.system_prompt: str | None = system_prompt
        self.role: str | None = role
        self.name: str | None = name

        self.total_tokens: TokenCounter = TokenCounter()
        self.autoname_task: asyncio.Task[None] | None = None

        # Extract metadata from the history, like info about files we previously uploaded and older user messages.
        self.user_messages: list[str] = common.get_user_messages(history)
        self.uploaded_files: dict[str, files.FileMetadata] = common.get_uploaded_files(history)

        # Subscribable events, e.g. when files change or autonaming finishes
        self.on_files_changed: AsyncEventEmitter[[dict[str, files.FileMetadata]]] = AsyncEventEmitter()
        self.on_autoname_finished: AsyncEventEmitter[[str]] = AsyncEventEmitter()

    # -------------------------------------
    # File things
    # -------------------------------------
    def erase_invalid_file_content_blocks(self):
        """
        Erase all content blocks in the history that have no entry in `uploaded_files`. This is useful when we want to remove file references
        from the history after verifying or processing them.
        """

        def is_valid_block(block: JSON) -> bool:
            match block:
                case {"type": "container_upload", "file_id": file_id}:
                    return file_id in self.uploaded_files
                case {"type": "document" | "image", "source": {"file_id": file_id}}:
                    return file_id in self.uploaded_files
                case _:  # Other block types, like text or tool use are always valid
                    return True

        for message in self.history:
            if get(message, "role", str) != "user":
                continue

            content = get_or_default(message, "content", list[JSON])
            message["content"] = [block for block in content if is_valid_block(block)]

    async def verify_file_uploads(self, client_session: aiohttp.ClientSession):
        """
        Verifies the uploaded files by checking their metadata. This is useful to ensure that the files are still valid and have not been
        removed or corrupted. The updates the `uploaded_files` dictionary with the metadata of the uploaded files.
        """
        if not self.uploaded_files:
            return

        get_file_metadata_task = asyncio.gather(
            *(files.get_file_metadata(client_session, file_id) for file_id, m in self.uploaded_files.items() if not is_valid_metadata(m)),
            return_exceptions=True,
        )

        metadata_list = await get_file_metadata_task
        for metadata in metadata_list:
            match metadata:
                case {"id": str(file_id)}:
                    self.uploaded_files[file_id].update(metadata)
                case BaseException() as e:
                    logger.error(f"Failed to verify file upload: {e}")
                case _:
                    logger.warning(f"Unexpected metadata format: {metadata}. Expected a JSON object with an 'id' field.")

        # Remove any files that were not found or had an error
        missing_files = [file_id for file_id, metadata in self.uploaded_files.items() if not is_valid_metadata(metadata)]
        for file_id in missing_files:
            logger.warning(f"File ID `{file_id}` is missing. Please re-upload it.")
            del self.uploaded_files[file_id]

        self.erase_invalid_file_content_blocks()

        downloadable_files = [m for m in self.uploaded_files.values() if is_valid_metadata(m) and m.get("downloadable")]
        if downloadable_files:
            logger.info("Downloadable files are available. Type `/download` to download them.")
            for metadata in downloadable_files:
                match metadata:
                    case {"id": str(file_id), "filename": str(file_name), "size_bytes": int(num_bytes)}:
                        logger.info(f"- {file_name}, {naturalsize(num_bytes)} (id={file_id})")
                    case _:
                        logger.warning(f"Unexpected metadata format for downloadable file: {metadata}")

        await self.on_files_changed.emit(self.uploaded_files)

    async def upload_file(self, client_session: aiohttp.ClientSession, file_path: str) -> dict[str, JSON]:
        if not await aiofiles.os.path.isfile(file_path):
            logger.error(f"File {file_path} does not exist or is not a file.")
            raise FileNotFoundError(f"File {file_path} does not exist or is not a file.")

        result = await files.upload_file(client_session, file_path)
        file_id = get(result, "id", str)
        if file_id is None:
            raise RuntimeError(f"Failed to upload file {file_path}. No file ID returned.")

        self.uploaded_files[file_id] = result

        await self.on_files_changed.emit(self.uploaded_files)
        return result

    # -------------------------------------
    # Autonaming things
    # -------------------------------------
    async def _autoname(self, client_session: aiohttp.ClientSession):
        autoname_prompt = "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you."
        autoname_history = self.history.copy() + [{"role": "user", "content": [{"type": "text", "text": autoname_prompt}]}]

        try:
            response = await prompt.stream_response(
                session=client_session,
                model=self.model,
                history=autoname_history,
                max_tokens=30,
                enable_web_search=False,
                system_prompt=self.system_prompt,
                enable_thinking=False,
            )

            self.total_tokens += response.tokens
            session_name = await history_to_string(response.messages, pretty=False)
        except (aiohttp.ClientError, asyncio.CancelledError) as e:
            if isinstance(e, asyncio.CancelledError):
                logger.error("Auto-naming cancelled. Using timestamp.")
            else:
                logger.exception(f"Error auto-naming session: {e}.")
            session_name = datetime.datetime.now().strftime("%H-%M-%S")

        session_name = session_name.strip().lower()
        session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
        session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

        date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.name = f"{date}-{session_name}"
        logger.info(f"[✓] Session named {self.name}")

        await self.on_autoname_finished.emit(self.name)

    @property
    def is_autonaming(self) -> bool:
        return self.autoname_task is not None and not self.autoname_task.done()

    def start_autoname_task(self, client_session: aiohttp.ClientSession):
        if self.autoname_task is not None:
            logger.warning("Autonaming task already running, not starting a new one.")
            return

        self.autoname_task = asyncio.create_task(self._autoname(client_session))

    def cancel_autoname(self):
        if self.autoname_task is None:
            return

        if not self.autoname_task.done():
            _ = self.autoname_task.cancel()

    async def cancel_autoname_with_date_fallback(self):
        if self.autoname_task is None:
            return

        try:
            if not self.autoname_task.done():
                _ = self.autoname_task.cancel()
            await self.autoname_task
        except aiohttp.ClientError:
            pass
        except asyncio.CancelledError:
            pass


def deduce_session_name(session_file: str | None) -> str | None:
    """
    Deduce the session name from the command line arguments. If a session file is provided, use its basename. Otherwise, return None.
    """
    if session_file:
        session_name = os.path.basename(session_file)
        stem, ext = os.path.splitext(session_name)
        if ext.lower() == ".json":
            return stem
        return session_name

    return None
