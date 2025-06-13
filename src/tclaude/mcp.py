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

import itertools
import logging
import os
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from enum import Enum
from functools import partial
from io import TextIOWrapper
from typing import TextIO, override
from urllib.parse import urlparse

import aiohttp

from .json import JSON, get, get_or, get_or_default
from .task_context import AsyncContextPool
from .tool_use import AvailableTools
from .tools import ToolContentBase64Image, ToolContentText, ToolResult

logger = logging.getLogger(__package__)


class AuthenticationType(Enum):
    NONE = "none"
    OAUTH2 = "oauth2"
    TOKEN = "token"


class ConnectionType(Enum):
    URL = "url"
    STDIN = "stdin"


# Used for forwarding stderr output from stdin MCP servers to our logger.
class LoggingStream(TextIO):
    def __init__(self, name: str):
        self._name: str = name

        self._read_fd: int
        self._write_fd: int
        self._read_fd, self._write_fd = os.pipe()
        self._file: TextIOWrapper = os.fdopen(self._write_fd, "w")

        self._reader: threading.Thread = threading.Thread(target=self._read_loop, daemon=False)
        self._reader.start()

    def _read_loop(self):
        with os.fdopen(self._read_fd, "r") as f:
            for line in iter(f.readline, ""):
                if line.strip():
                    logger.info(f"[mcp:{self._name}] {line.strip()}")

    @override
    def write(self, s: str) -> int:
        return self._file.write(s)

    @override
    def flush(self) -> None:
        self._file.flush()

    @override
    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
            self._reader.join(timeout=1.0)

    @override
    def fileno(self) -> int:
        return self._write_fd


class McpConnection:
    def __init__(self, name: str, connection: object):
        from mcp import ClientSession, Tool

        if not isinstance(connection, ClientSession):
            raise TypeError(f"Expected a ClientSession, got {type(connection).__name__}")

        self.name: str = name

        self._conn: ClientSession = connection
        self._tools: list[Tool] = []

    async def fetch_info(self) -> None:
        """
        Fetch various information about the MCP server, like available tools.
        """
        self._tools = (await self._conn.list_tools()).tools

    def get_tools(self) -> tuple[AvailableTools, list[JSON]]:
        available_tools: AvailableTools = {}
        tool_definitions: list[JSON] = []

        for tool in self._tools:
            if not tool.name or not tool.description or not tool.inputSchema:
                logger.warning(f"Skipping tool '{tool.name}' due to missing name, description, or input schema.")
                continue

            # Make sure different MCP servers don't conflict with each other
            tool_name = f"mcp_{self.name}_{tool.name}"
            available_tools[tool_name] = partial(self.call_tool, tool.name)
            tool_definitions.append(
                {
                    "name": tool_name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            )

        return available_tools, tool_definitions

    async def call_tool(self, tool_name: str, **kwargs: JSON) -> ToolResult:
        from mcp.types import (
            BlobResourceContents,
            EmbeddedResource,
            ImageContent,
            TextContent,
            TextResourceContents,
        )

        content: list[ToolContentText | ToolContentBase64Image] = []

        result = await self._conn.call_tool(tool_name, kwargs)
        for c in result.content:
            match c:
                case TextContent():
                    content.append(ToolContentText(c.text))
                case ImageContent():
                    content.append(ToolContentBase64Image(c.data, c.mimeType))
                case EmbeddedResource():
                    match c.resource:
                        case TextResourceContents():
                            content.append(ToolContentText(c.resource.text))
                        case BlobResourceContents():
                            logger.warning(f"Ignoring embedded resource of type '{c.type}' in tool '{tool_name}'")

        return ToolResult(content, is_error=result.isError)


class McpServerConfig:
    def __init__(self, server: dict[str, JSON]):
        self.type: ConnectionType

        match server:
            case {"url": str(), "command": str()}:
                raise ValueError("MCP server configuration cannot have both 'url' and 'command' keys.")
            case {"url": str(url)}:
                self.type = ConnectionType.URL
                self.url: str = url
                self.authentication: AuthenticationType = AuthenticationType(get_or(server, "authentication", "none"))
                self.authorization_token: str | None = get(server, "authorization_token", str)
                self.oauth_token: dict[str, JSON] | None = None

                if self.authentication not in AuthenticationType:
                    raise ValueError(f"Invalid authentication type '{self.authentication}' for server '{self.name}'.")

                if self.authentication == AuthenticationType.TOKEN and not self.authorization_token:
                    raise ValueError(f"Server '{self.name}' requires an authorization token but none was provided.")

            case {"command": str(command)}:
                self.type = ConnectionType.STDIN
                self.command: str = command
                self.args: list[str] = get_or_default(server, "args", list[str])
                self.env: dict[str, str] | None = get(server, "env", dict[str, str])
                self.cwd: str | None = get(server, "cwd", str)

            case _:
                raise ValueError("MCP server configuration must have either 'url' or 'command' key.")

        self.name: str = get_or(server, "name", "")
        self.tool_configuration: JSON = get(server, "tool_configuration", dict[str, JSON])

        # TODO: send to direct MCP server connections https://modelcontextprotocol.io/docs/concepts/roots
        self.roots: list[str] = get_or_default(server, "roots", list[str])

        if not self.name:
            raise ValueError("MCP server configuration must have a 'name' key with a non-empty string value.")

    async def ensure_auth(self, session: aiohttp.ClientSession) -> None:
        """
        Authenticate the server if it requires authentication.
        """
        if not self.type == ConnectionType.URL or not self.authentication == AuthenticationType.OAUTH2:
            return

        from . import oauth

        # Obtain authorization and token URLs per the MCP spec
        # https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization#2-3-2-authorization-base-url
        parsed = urlparse(self.url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if self.oauth_token:
            if oauth.is_expiring(self.oauth_token):
                logger.debug(f"Refreshing OAuth token for server '{self.name}'")
                self.oauth_token = await oauth.OAuth2Client(session, self.name, base_url).refresh_token(session, self.oauth_token)

            return

        self.oauth_token = await oauth.OAuth2Client(session, self.name, base_url).get_token(session)

    @property
    def is_authenticated(self) -> bool:
        if self.type != ConnectionType.URL or self.authentication != AuthenticationType.OAUTH2:
            return True

        from . import oauth

        return bool(self.oauth_token) and not oauth.is_expired(self.oauth_token)

    @asynccontextmanager
    async def connect(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        async def create_conn(session: ClientSession) -> McpConnection:
            conn = McpConnection(self.name, session)
            await conn.fetch_info()
            return conn

        match self.type:
            case ConnectionType.STDIN:
                server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
                log_stream = LoggingStream(self.name)
                try:
                    async with stdio_client(server_params, errlog=log_stream) as (read, write):
                        async with ClientSession(read_stream=read, write_stream=write) as session:
                            conn = await create_conn(session)
                            yield conn
                finally:
                    log_stream.close()

            case ConnectionType.URL:
                if self.authentication != AuthenticationType.NONE:
                    # TODO: handle authentication for local servers
                    raise ValueError(f"Cannot connect to local server '{self.name}' via URL w/ authentication.")

                async with streamablehttp_client(self.url) as (read, write, _):
                    async with ClientSession(read_stream=read, write_stream=write) as session:
                        conn = await create_conn(session)
                        yield conn

    async def get_remote_server_desc(self, session: aiohttp.ClientSession) -> dict[str, JSON]:
        """
        Get a description of the remote server configuration the Anthropic API expects.
        """
        if self.type != ConnectionType.URL:
            raise ValueError(f"Cannot get remote server description for server '{self.name}' as it is not a URL type.")

        result: dict[str, JSON] = {
            "url": self.url,
            "name": self.name,
            "type": "url",
        }

        match self.authentication:
            case AuthenticationType.NONE:
                pass  # No authentication required, nothing to add
            case AuthenticationType.OAUTH2:
                # Ensures that tokens that need refreshing are refreshed
                await self.ensure_auth(session)
                if self.oauth_token:
                    result["authorization_token"] = get_or(self.oauth_token, "access_token", "")
            case AuthenticationType.TOKEN:
                result["authorization_token"] = self.authorization_token

        return result


class McpServerConfigs:
    def __init__(self, session: aiohttp.ClientSession, remote_servers: list[McpServerConfig], local_servers: list[McpServerConfig]):
        self.remote_servers: list[McpServerConfig] = remote_servers
        self.local_servers: list[McpServerConfig] = local_servers

        self._session: aiohttp.ClientSession = session
        self._conns: list[McpConnection] = []
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._connection_pool: AsyncContextPool = AsyncContextPool()

    @property
    def empty(self) -> bool:
        return not self.remote_servers and not self.local_servers

    async def __aenter__(self) -> McpServerConfigs:
        await self.ensure_auth(self._session)

        if self._conns:
            raise RuntimeError("MCP servers are already connected. Call close() before reconnecting.")

        pool = await self._exit_stack.enter_async_context(AsyncContextPool())
        conns = await pool.add_many(*[s.connect() for s in self.local_servers])
        for conn in conns:
            if isinstance(conn, BaseException):
                logger.error(f"Error connecting to local MCP server: {conn}")
            else:
                self._conns.append(conn)

        if self._conns:
            logger.info(f"[✓] Connected to {len(self._conns)} MCP servers: {', '.join([c.name for c in self._conns])}")

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        await self._exit_stack.aclose()
        self._conns.clear()
        logger.debug("Closed all MCP server connections.")
        return False  # Do not suppress exceptions

    def get_tools(self) -> tuple[AvailableTools, list[JSON]]:
        available_tools: AvailableTools = {}
        tool_definitions: list[JSON] = []

        for conn in self._conns:
            tools, definitions = conn.get_tools()
            available_tools.update(tools)
            tool_definitions.extend(definitions)

        return available_tools, tool_definitions

    async def ensure_auth(self, session: aiohttp.ClientSession) -> None:
        for server in itertools.chain(self.remote_servers, self.local_servers):
            try:
                await server.ensure_auth(session)
            except ValueError as e:
                logger.error(f"Error authenticating MCP server '{server.name}': {e}")

        self.remote_servers = [s for s in self.remote_servers if s.is_authenticated]
        self.local_servers = [s for s in self.local_servers if s.is_authenticated]

        if self.remote_servers:
            logger.info(f"[✓] Authenticated {len(self.remote_servers)} remote MCP servers: {', '.join(s.name for s in self.remote_servers)}")

    async def get_remote_server_descs(self, session: aiohttp.ClientSession) -> list[dict[str, JSON]]:
        return [await server.get_remote_server_desc(session) for server in self.remote_servers if server.url]


def setup_mcp(session: aiohttp.ClientSession, config: dict[str, JSON]) -> McpServerConfigs:
    """
    Get the MCP configuration from the loaded config.
    """
    if "mcp" not in config:
        return McpServerConfigs(session, remote_servers=[], local_servers=[])

    remote_servers: list[McpServerConfig] = []
    for s in get_or_default(config["mcp"], "remote_servers", list[dict[str, JSON]]):
        try:
            server = McpServerConfig(s)
            if server.type != ConnectionType.URL:
                raise ValueError(f"Remote MCP server '{server.name}' must be of type URL.")

            remote_servers.append(server)
        except ValueError as e:
            logger.error(f"Error parsing remote MCP server configuration: {e}")

    local_servers: list[McpServerConfig] = []
    for s in get_or_default(config["mcp"], "local_servers", list[dict[str, JSON]]):
        try:
            local_servers.append(McpServerConfig(s))
        except ValueError as e:
            logger.error(f"Error parsing local MCP server configuration: {e}")

    return McpServerConfigs(session, remote_servers=remote_servers, local_servers=local_servers)
