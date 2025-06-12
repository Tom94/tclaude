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


import os
import subprocess

from .json import JSON

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")
VERTEX_API_PROJECT = os.getenv("VERTEX_API_PROJECT")


def get_gcp_access_token() -> str:
    cmd = ["gcloud", "auth", "print-access-token"]
    token = subprocess.check_output(cmd).decode("utf-8").strip()
    return token


def get_messages_endpoint_vertex(model: str) -> tuple[str, dict[str, str], dict[str, JSON]]:
    if not VERTEX_API_PROJECT:
        raise ValueError("VERTEX_API_PROJECT environment variable must be set")

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {get_gcp_access_token()}",
        "Content-Type": "application/json",
    }

    url = f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_API_PROJECT}/locations/global/publishers/anthropic/models/{model}:streamRawPredict"
    params: dict[str, JSON] = {
        "anthropic_version": "vertex-2023-10-16",
    }

    return url, headers, params


def get_messages_endpoint_anthropic(model: str) -> tuple[str, dict[str, str], dict[str, JSON]]:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

    beta_features = [
        "interleaved-thinking-2025-05-14",
        "code-execution-2025-05-22",
        "files-api-2025-04-14",
        "mcp-client-2025-04-04",
        "fine-grained-tool-streaming-2025-05-14"
    ]

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": ",".join(beta_features),
    }

    url = "https://api.anthropic.com/v1/messages"
    params: dict[str, JSON] = {
        "model": model,
    }

    return url, headers, params


def get_files_endpoint_anthropic() -> tuple[str, dict[str, str]]:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

    # Prepare headers
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }

    url = "https://api.anthropic.com/v1/files"

    return url, headers
