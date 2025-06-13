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
import base64
import hashlib
import json
import logging
import os
import secrets
import time
import webbrowser
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable, cast

import aiofiles
import aiofiles.os
import aiohttp
import keyring
import keyring.errors
from aiohttp import web
from multidict import MultiMapping
from oauthlib.oauth2 import WebApplicationClient

from . import __version__, common
from .json import JSON, get, get_or

logger = logging.getLogger(__package__)

SOFTWARE_ID = "f4f90cea-20ab-4778-a1a3-3e8091d2f939"


def get_oauth_state_dir() -> str:
    return os.path.join(common.get_state_dir(), "oauth")


def get_oauth_cache_dir() -> str:
    return os.path.join(common.get_cache_dir(), "oauth")


async def get_server_metadata(session: aiohttp.ClientSession, name: str, server_base_url: str) -> dict[str, JSON] | None:
    metadata_filename = os.path.join(get_oauth_cache_dir(), f"{name}-metadata.json")
    if os.path.isfile(metadata_filename):
        with open(metadata_filename, "r") as f:
            try:
                metadata = cast(dict[str, JSON], json.load(f))
            except json.JSONDecodeError:
                logger.exception(f"Failed to decode JSON from {metadata_filename}. Fetching fresh metadata.")
                metadata = None

        if metadata:
            logger.debug(f"Using cached metadata from {metadata_filename}")
            return metadata

    # OAuth 2.0 Authorization Server Metadata (RFC 8414)
    discovery_url = f"{server_base_url}/.well-known/oauth-authorization-server"

    try:
        async with session.get(discovery_url, timeout=aiohttp.ClientTimeout(5)) as response:
            response.raise_for_status()
            metadata = cast(dict[str, JSON], await response.json())

        logger.debug(f"Fetched metadata from {discovery_url}. Caching.")

        os.makedirs(os.path.dirname(metadata_filename), exist_ok=True)
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
    except aiohttp.ClientError:
        logger.debug(f"Failed to fetch metadata from {discovery_url}. Falling back to default endpoints.", exc_info=True)
        return None


async def load_client_info_from_file(name: str) -> dict[str, JSON] | None:
    client_info_filename = os.path.join(get_oauth_state_dir(), f"{name}-client-info.json")
    if await aiofiles.os.path.isfile(client_info_filename):
        with open(client_info_filename, "r") as f:
            try:
                client_info = cast(dict[str, JSON], json.load(f))
            except json.JSONDecodeError:
                logger.exception(f"Failed to decode JSON from {client_info_filename}. Registering new client.")
                return None

        if client_info:
            logger.debug(f"Using existing client info from {client_info_filename}")
            return client_info

    return None


async def save_client_info_to_file(name: str, client_info: dict[str, JSON]) -> None:
    client_info_filename = os.path.join(get_oauth_state_dir(), f"{name}-client-info.json")
    os.makedirs(os.path.dirname(client_info_filename), exist_ok=True)
    with open(client_info_filename, "w") as f:
        json.dump(client_info, f, indent=2)


async def get_client_info(session: aiohttp.ClientSession, name: str, registration_url: str, redirect_uri: str) -> dict[str, JSON]:
    """
    Register a new OAuth client dynamically using RFC 7591
    """
    client_info = await load_client_info_from_file(name)
    if client_info:
        return client_info

    registration_data = {
        "client_name": "tclaude",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "software_id": SOFTWARE_ID,
        "software_version": __version__,
        "token_endpoint_auth_method": "none",
    }

    try:
        async with session.post(registration_url, json=registration_data) as response:
            response.raise_for_status()
            client_info = cast(dict[str, JSON], await response.json())

        if not client_info or "client_id" not in client_info:
            raise ValueError("Received empty client info from registration endpoint")

        logger.debug(f"Registered new client with ID {client_info['client_id']} at {registration_url}. Caching.")
        await save_client_info_to_file(name, client_info)

        return client_info
    except aiohttp.ClientError as e:
        raise Exception(f"Registration failed: {e}") from e


def get_token_from_keyring(name: str) -> dict[str, JSON] | None:
    service_name = f"tclaude-{name}-oauth2"
    try:
        token_json = keyring.get_password(service_name, "default")
    except keyring.errors.KeyringError as e:
        logger.exception(f"Failed to retrieve token from keyring: {e}. Token will not be used.")
        return None

    if not token_json:
        return None
    try:
        token = cast(dict[str, JSON], json.loads(token_json))
    except json.JSONDecodeError:
        logger.exception(f"Failed to decode token JSON for {name} from keyring.")
        return None

    if "access_token" not in token:
        logger.warning(f"Token for {name} in keyring does not contain access_token.")
        return None

    return token


def save_token_to_keyring(name: str, token: dict[str, JSON]) -> None:
    service_name = f"tclaude-{name}-oauth2"
    try:
        keyring.set_password(service_name, "default", json.dumps(token))
        logger.debug(f"Stored token for {name} in keyring.")
    except keyring.errors.KeyringError as e:
        logger.exception(f"Failed to store token in keyring: {e}. Token will not be persisted.")


def is_expiring(token: dict[str, JSON], tolerance: float = 300) -> bool:
    expires_at = get_or(token, "expires_at", 0.0)
    if expires_at < time.time() + tolerance:  # If we're closer than 5 minutes to expiry, refresh
        return True
    return False


def is_expired(token: dict[str, JSON]) -> bool:
    return is_expiring(token, tolerance=0)


@dataclass
class OAuthEndpoints:
    auth_url: str
    token_url: str
    registration_url: str

    @classmethod
    async def from_base_url(cls, session: aiohttp.ClientSession, name: str, base_url: str) -> OAuthEndpoints:
        metadata = await get_server_metadata(session, name, base_url)
        auth_url = get_or(metadata, "authorization_endpoint", f"{base_url}/authorize")
        token_url = get_or(metadata, "token_endpoint", f"{base_url}/token")
        registration_url = get_or(metadata, "registration_endpoint", f"{base_url}/register")
        return cls(auth_url, token_url, registration_url)


class OAuth2Client:
    def __init__(self, session: aiohttp.ClientSession, name: str, base_url: str, local_port: int = 17993):
        self.name: str = name
        self.base_url: str = base_url
        self.local_port: int = local_port
        self.redirect_uri: str = f"http://localhost:{local_port}"

    def _generate_pkce(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("utf-8").rstrip("=")

        challenge = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(challenge).decode("utf-8").rstrip("=")
        return code_verifier, code_challenge

    async def _start_async_server(self, callback_handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> web.AppRunner:
        """Start async HTTP server for OAuth callback"""
        app = web.Application()
        _ = app.router.add_get("/", callback_handler)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, "localhost", self.local_port)
        await site.start()

        return runner

    async def authenticate(self, session: aiohttp.ClientSession) -> dict[str, JSON]:
        endpoints = await OAuthEndpoints.from_base_url(session, self.name, self.base_url)

        client_info: dict[str, JSON] = await get_client_info(session, self.name, endpoints.registration_url, self.redirect_uri)
        client_id = get(client_info, "client_id", str)
        if not client_id:
            raise ValueError("Client ID not found in client info")

        client = WebApplicationClient(client_id)
        code_verifier, code_challenge = self._generate_pkce()

        # State is a random string generated by the client. The server will return this string when redirecting back to the client to prevent CSRF attacks.
        state: str = secrets.token_urlsafe(32)
        auth_url = client.prepare_request_uri(  # pyright: ignore[reportUnknownMemberType]
            endpoints.auth_url,
            redirect_uri=self.redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method="S256",
            state=state,
        )

        callback_event: asyncio.Future[MultiMapping[str]] = asyncio.Future()

        async def callback_handler(request: web.Request) -> web.Response:
            query = request.query
            callback_event.set_result(query)

            if "error" in query:
                return web.Response(
                    text="<html><body><h1>Authorization failed!</h1><p>Error: " + query["error"] + "</p></body></html>", content_type="text/html"
                )
            elif "code" not in query:
                return web.Response(
                    text="<html><body><h1>Authorization failed!</h1><p>No authorization code in response.</p></body></html>", content_type="text/html"
                )
            else:
                return web.Response(
                    text="<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>", content_type="text/html"
                )

        runner = await self._start_async_server(callback_handler)
        try:
            if webbrowser.open(auth_url):
                logger.info(f"Authenticate MCP server '{self.name}' in your browser.")
            else:
                logger.info(f"Authenticate MCP server '{self.name}' by opening this URL: {auth_url}")

            try:
                callback_result = await callback_event
            except asyncio.TimeoutError:
                raise ValueError("Authentication timeout")
        finally:
            await runner.cleanup()

        # Parse callback URL
        logger.debug(f"Received callback qs: {callback_result}")
        if "error" in callback_result:
            raise ValueError(f"Authorization error: {callback_result['error']}")
        if "code" not in callback_result:
            raise ValueError("Authorization code not found in callback URL")

        code = callback_result["code"]
        returned_state = callback_result.get("state", None)
        if returned_state != state:
            raise ValueError(f"State mismatch: expected {state}, got {returned_state}")

        token_url, headers, body = client.prepare_token_request(  # pyright: ignore[reportUnknownMemberType]
            endpoints.token_url, code=code, state=state, redirect_url=self.redirect_uri, code_verifier=code_verifier
        )

        try:
            async with session.post(token_url, data=body, headers=headers) as response:
                response.raise_for_status()
                token = client.parse_request_body_response(await response.text())  # pyright: ignore[reportUnknownMemberType]
                save_token_to_keyring(self.name, token)
                return token
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to obtain token for {self.name}: {e}") from e

    async def refresh_token(self, session: aiohttp.ClientSession, token: dict[str, JSON]) -> dict[str, JSON]:
        endpoints = await OAuthEndpoints.from_base_url(session, self.name, self.base_url)

        refresh_token = get(token, "refresh_token", str)
        if not refresh_token:
            logger.debug("No refresh token available, re-authenticating...")
            return await self.authenticate(session)

        client_info = await get_client_info(session, self.name, endpoints.registration_url, self.redirect_uri)
        client_id = get(client_info, "client_id", str)
        if not client_id:
            raise ValueError("Client ID not found in client info")

        client = WebApplicationClient(client_id)
        token_url, headers, body = client.prepare_refresh_token_request(endpoints.token_url, refresh_token=refresh_token, client_id=client_id)  # pyright: ignore[reportUnknownMemberType]

        try:
            async with session.post(token_url, data=body, headers=headers) as response:
                response.raise_for_status()
                new_token = client.parse_request_body_response(await response.text())  # pyright: ignore[reportUnknownMemberType]
                save_token_to_keyring(self.name, new_token)
                return new_token
        except aiohttp.ClientError as e:
            logger.exception(f"Failed to refresh token for {self.name}: {e}. Re-authenticating...")
            return await self.authenticate(session)

    async def get_token(self, session: aiohttp.ClientSession) -> dict[str, JSON]:
        token: dict[str, JSON] | None = get_token_from_keyring(self.name)
        if token:
            logger.debug(f"Using cached token for {self.name} from keyring.")
        else:
            logger.debug(f"No cached token found for {self.name} in keyring, authenticating.")
            token = await self.authenticate(session)

        if is_expiring(token):
            logger.debug(f"Token for {self.name} is expiring, refreshing.")
            token = await self.refresh_token(session, token)

        return token
