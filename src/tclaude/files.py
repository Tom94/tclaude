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
import asyncio
import json
import mimetypes
import os
from collections.abc import Mapping
from typing import cast

import aiofiles
import aiohttp

from . import endpoints
from .common import FileMetadata
from .json import JSON, get, get_or


MAX_FILE_LIST = 1000


def mime_type_to_content_block_type(mime_type: str) -> str | None:
    if mime_type.startswith("image/"):
        return "image"
    elif mime_type.startswith("application/pdf") or mime_type.startswith("text/plain"):
        return "document"
    else:
        return None


async def list_files(session: aiohttp.ClientSession, after_file_id: str | None, num_files: int = 50) -> JSON:
    url, headers = endpoints.get_files_endpoint_anthropic()
    params: dict[str, int | str] = {"limit": num_files}
    if after_file_id:
        params["after_id"] = after_file_id

    async with session.get(url, headers=headers, params=params) as response:
        response.raise_for_status()
        data = cast(JSON, await response.json())

    return data


async def rm_file(session: aiohttp.ClientSession, file_id: str) -> JSON:
    url, headers = endpoints.get_files_endpoint_anthropic()
    url += f"/{file_id}"

    try:
        async with session.delete(url, headers=headers) as response:
            response.raise_for_status()
            data = cast(JSON, await response.json())
    except aiohttp.ClientResponseError as e:
        data = {
            "error": {
                "message": f"Failed to delete file {file_id}: {e.message}",
                "status": e.status,
            }
        }

    return data


async def upload_file(session: aiohttp.ClientSession, file_path: str) -> FileMetadata:
    async with aiofiles.open(file_path, "rb") as file:  # pyright: ignore[reportUnknownMemberType]
        file_data = await file.read()

    mime_type, content_encoding = mimetypes.guess_file_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    return await upload_file_mem(session, file_path, file_data, mime_type, content_encoding)


async def upload_file_base64(
    session: aiohttp.ClientSession, file_path: str, file_data_base64: str, mime_type: str, content_encoding: str | None
) -> FileMetadata:
    import base64

    file_data = base64.b64decode(file_data_base64)
    return await upload_file_mem(session, file_path, file_data, mime_type, content_encoding)


async def upload_file_mem(session: aiohttp.ClientSession, file_path: str, file_data: bytes, mime_type: str, content_encoding: str | None) -> FileMetadata:
    url, headers = endpoints.get_files_endpoint_anthropic()

    try:
        if content_encoding is not None:
            headers["Content-Encoding"] = content_encoding

        form_data = aiohttp.FormData()
        form_data.add_field("file", file_data, filename=os.path.basename(file_path), content_type=mime_type)

        async with session.post(url, headers=headers, data=form_data) as response:
            response.raise_for_status()
            data = cast(FileMetadata, await response.json())
    except aiohttp.ClientResponseError as e:
        data: FileMetadata = {
            "error": {
                "message": f"Failed to upload file {file_path}: {e.message}",
                "status": e.status,
            }
        }

    return data


async def get_file_metadata(session: aiohttp.ClientSession, file_id: str) -> FileMetadata:
    url, headers = endpoints.get_files_endpoint_anthropic()
    url += f"/{file_id}"

    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = cast(FileMetadata, await response.json())
    except aiohttp.ClientResponseError as e:
        raise FileNotFoundError(f"Failed to get metadata for file {file_id}: {e.message}") from e

    return data


async def get_file_metadata_or_none(session: aiohttp.ClientSession, file_id: str) -> FileMetadata | None:
    """
    Get file metadata, returning None if the file does not exist.
    """
    try:
        return await get_file_metadata(session, file_id)
    except FileNotFoundError:
        return None


def get_path_from_metadata(file_id: str, metadata: FileMetadata) -> str:
    filename = get(metadata, "filename", str)
    if filename is None:
        extension = ".txt"
        mime_type = get(metadata, "mime_type", str)
        if mime_type is not None:
            extension = mimetypes.guess_extension(mime_type) or ".txt"
        filename = f"{file_id}{extension}"

    return filename


async def download_file(session: aiohttp.ClientSession, file_id: str, file_path: str):
    url, headers = endpoints.get_files_endpoint_anthropic()
    url += f"/{file_id}/content"

    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()

            async with aiofiles.open(file_path, "wb") as file:  # pyright: ignore[reportUnknownMemberType]
                async for data, _ in response.content.iter_chunks():
                    _ = await file.write(data)
    except aiohttp.ClientResponseError as e:
        raise FileNotFoundError(f"Failed to download file {file_id}: {e.message}") from e


async def download_file_mem(session: aiohttp.ClientSession, file_id: str) -> bytes:
    url, headers = endpoints.get_files_endpoint_anthropic()
    url += f"/{file_id}/content"

    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.read()
    except aiohttp.ClientResponseError as e:
        raise FileNotFoundError(f"Failed to download file {file_id}: {e.message}") from e


class FilesArgs(argparse.Namespace):
    def __init__(self):
        super().__init__()

        self.command: str | None = None

        # ls
        self.after_file_id: str | None = ""
        self.num_files: int = 50

        # rm, upload, and download
        self.files: list[str]


async def async_main():
    """
    Main function to parse arguments, load a JSON file containing conversation history, and print it.
    """
    parser = argparse.ArgumentParser(description="Server file operations for Anthropic AI models")
    subparsers = parser.add_subparsers(dest="command", help="commands")

    ls_parser = subparsers.add_parser("ls", help="List files")
    _ = ls_parser.add_argument("after_file_id", help="File id after which to start listing", nargs="?", type=str)
    _ = ls_parser.add_argument("-n", "--num_files", default=50, help="Number of files to list", type=int)

    rm_parser = subparsers.add_parser("rm", help="Remove files")
    _ = rm_parser.add_argument("files", help="File ids to remove", nargs="+", type=str)

    upload_parser = subparsers.add_parser("upload", help="Upload files")
    _ = upload_parser.add_argument("files", help="Files to upload", nargs="+", type=str)

    upload_parser = subparsers.add_parser("download", help="Download files")
    _ = upload_parser.add_argument("files", help="File ids to download", nargs="+", type=str)

    args = parser.parse_args(namespace=FilesArgs())

    if not args.command:
        parser.print_help()
        return

    async with aiohttp.ClientSession() as session:
        if args.command == "ls":
            print(f"Listing {args.num_files} files after {args.after_file_id or 'start'}")
            files = await list_files(session, args.after_file_id, args.num_files)
            print(f"Files: {json.dumps(files, indent=2)}")
        elif args.command == "rm":
            print("Deleting files:", ", ".join(args.files))
            if not args.files:
                print("No files specified to remove.")
                return

            if "all" in args.files:
                files = await list_files(session, None, MAX_FILE_LIST)
                data = get(files, "data", list[JSON])
                if data:
                    args.files = [str(file["id"]) for file in data if isinstance(file, Mapping) and "id" in file]

            rm_tasks: list[asyncio.Task[JSON]] = []
            async with asyncio.TaskGroup() as tg:
                for file_id in args.files:
                    rm_tasks.append(tg.create_task(rm_file(session, file_id)))

            for task, file in zip(rm_tasks, args.files):
                result = task.result()
                print(f"{file}: {json.dumps(result, indent=2)}")
        elif args.command == "upload":
            print("Uploading files:", ", ".join(args.files))
            if not args.files:
                print("No files specified to upload.")
                return

            upload_tasks: list[asyncio.Task[FileMetadata]] = []
            async with asyncio.TaskGroup() as tg:
                for file_path in args.files:
                    if not os.path.isfile(file_path):
                        print(f"File {file_path} does not exist or is not a file.")
                        continue
                    upload_tasks.append(tg.create_task(upload_file(session, file_path)))

            for task, file in zip(upload_tasks, args.files):
                result = task.result()
                print(f"{file}: {json.dumps(result, indent=2)}")
        elif args.command == "download":
            print("Downloading files:", ", ".join(args.files))
            if not args.files:
                print("No files specified to upload.")
                return

            async def get_filename_and_download(session: aiohttp.ClientSession, file_id: str):
                metadata = await get_file_metadata(session, file_id)
                file_path = get_or(metadata, "filename", f"{file_id}.txt")
                file_path = os.path.basename(file_path)
                await download_file(session, file_id, file_path)

            download_tasks: list[asyncio.Task[None]] = []
            async with asyncio.TaskGroup() as tg:
                for file_id in args.files:
                    download_tasks.append(tg.create_task(get_filename_and_download(session, file_id)))

            print("Finished.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
