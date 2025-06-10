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


import itertools
from enum import Enum
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from .json import JSON, get, get_or, get_or_default


class AuthenticationType(Enum):
    NONE = "none"
    OAUTH2 = "oauth2"
    TOKEN = "token"


class McpServerConfig:
    def __init__(self, server: JSON):
        self.url: str = get_or(server, "url", "")
        self.name: str = get_or(server, "name", "")
        self.type: str = get_or(server, "type", "url")
        self.authorization_token: str | None = get(server, "authorization_token", str)
        self.tool_configuration: JSON = get(server, "tool_configuration", dict[str, JSON])
        self.authentication: AuthenticationType = AuthenticationType(get_or(server, "authentication", "none"))

        self.oauth_token: dict[str, JSON] | None = None

        if not self.url:
            raise ValueError("MCP server configuration must have a 'url' key with a non-empty string value.")

        if not self.name:
            raise ValueError("MCP server configuration must have a 'name' key with a non-empty string value.")

        if self.authentication not in AuthenticationType:
            raise ValueError(f"Invalid authentication type '{self.authentication}' for server '{self.name}'.")

        if self.authentication == AuthenticationType.TOKEN and not self.authorization_token:
            raise ValueError(f"Server '{self.name}' requires an authorization token but none was provided.")

    async def ensure_auth(self, session: aiohttp.ClientSession) -> None:
        """
        Authenticate the server if it requires authentication.
        """
        if not self.authentication == AuthenticationType.OAUTH2:
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

    async def get_remote_server_desc(self, session: aiohttp.ClientSession) -> dict[str, JSON]:
        """
        Get a description of the remote server configuration the Anthropic API expects.
        """
        result: dict[str, JSON] = {
            "url": self.url,
            "name": self.name,
            "type": self.type,
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


class McpServersConfig:
    def __init__(self, remote_servers: list[McpServerConfig], local_servers: list[McpServerConfig]):
        self.remote_servers: list[McpServerConfig] = remote_servers
        self.local_servers: list[McpServerConfig] = local_servers

    async def ensure_auth(self, session: aiohttp.ClientSession) -> None:
        for server in itertools.chain(self.remote_servers, self.local_servers):
            try:
                await server.ensure_auth(session)
            except ValueError as e:
                logger.error(f"Error validating MCP server '{server.name}': {e}")
                continue

    async def get_remote_server_descs(self, session: aiohttp.ClientSession) -> list[dict[str, JSON]]:
        return [await server.get_remote_server_desc(session) for server in self.remote_servers if server.url]


async def get_mcp_config(session: aiohttp.ClientSession, config: dict[str, JSON]) -> McpServersConfig:
    """
    Get the MCP configuration from the loaded config.
    """
    if "mcp" not in config:
        return McpServersConfig(remote_servers=[], local_servers=[])

    remote_servers = [McpServerConfig(s) for s in get_or_default(config["mcp"], "remote_servers", list[JSON])]
    local_servers = [McpServerConfig(s) for s in get_or_default(config["mcp"], "local_servers", list[JSON])]

    result = McpServersConfig(remote_servers=remote_servers, local_servers=local_servers)
    await result.ensure_auth(session)
    return result
