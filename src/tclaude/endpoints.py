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


import subprocess

from .config import EndpointConfig
from .json import JSON


def get_gcp_access_token() -> str:
    cmd = ["gcloud", "auth", "print-access-token"]
    token = subprocess.check_output(cmd).decode("utf-8").strip()
    return token


def get_messages_endpoint_vertex(model: str, url: str, api_key: str) -> tuple[str, dict[str, str], dict[str, JSON]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{url}/{model}:streamRawPredict"
    params: dict[str, JSON] = {
        "anthropic_version": "vertex-2023-10-16",
    }

    return url, headers, params


def get_messages_endpoint_anthropic(model: str, url: str, api_key: str) -> tuple[str, dict[str, str], dict[str, JSON]]:
    beta_features = [
        "interleaved-thinking-2025-05-14",
        "code-execution-2025-05-22",
        "files-api-2025-04-14",
        "mcp-client-2025-04-04",
        "fine-grained-tool-streaming-2025-05-14",
    ]

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": ",".join(beta_features),
    }

    url = f"{url}/v1/messages"
    params: dict[str, JSON] = {
        "model": model,
    }

    return url, headers, params


def get_messages_endpoint(model: str, endpoint: EndpointConfig) -> tuple[str, dict[str, str], dict[str, JSON]]:
    if endpoint.kind == "anthropic":
        return get_messages_endpoint_anthropic(model, endpoint.url, endpoint.api_key or "")
    elif endpoint.kind == "vertex":
        return get_messages_endpoint_vertex(model, endpoint.url, endpoint.api_key or "")
    else:
        raise ValueError(f"Unsupported messages endpoint kind: {endpoint}")


def get_files_endpoint_anthropic(url: str, api_key: str) -> tuple[str, dict[str, str]]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }

    url = f"{url}/v1/files"

    return url, headers


def get_files_endpoint(endpoint: EndpointConfig) -> tuple[str, dict[str, str]]:
    if endpoint.kind == "anthropic":
        return get_files_endpoint_anthropic(endpoint.url, endpoint.api_key or "")
    else:
        raise ValueError(f"Unsupported files endpoint kind: {endpoint}")
