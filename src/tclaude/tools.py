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

"""
All functions in this file will be auto-converted to tool calls for Claude. Do not put any function into this file that should not be
directly callable by Claude.
"""


class ToolContentText:
    def __init__(self, text: str):
        self.text: str = text


class ToolContentBase64Image:
    def __init__(self, data: str, type: str):
        self.data: str = data
        self.type: str = type


class ToolResult:
    def __init__(self, content: list[ToolContentText | ToolContentBase64Image], is_error: bool):
        self.content: list[ToolContentText | ToolContentBase64Image] = content
        self.is_error: bool = is_error


async def fetch_url(url: str) -> ToolResult:
    """
    Fetch the content of a URL and transform it as a markdown string. The raw HTML text is cleaned up by removing script, style, and other
    non-content elements, followed by conversion to markdown format.

    Args:
        url (str): The URL to fetch.

    Returns:
        The content of the URL as a markdown string.
    """
    import aiohttp
    import html2text
    from bs4 import BeautifulSoup

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(5)) as response:
            response.raise_for_status()

            soup = BeautifulSoup(await response.text(), "html.parser")

            # Remove script and style elements
            for script_or_style in soup(["script", "style", "meta", "link", "noscript", "iframe", "embed", "object"]):  # pyright: ignore[reportAny]
                script_or_style.decompose()  # pyright: ignore[reportAny]

            # Get the text content
            cleaned_html = str(soup)

            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.skip_internal_links = True
            h.inline_links = True
            h.wrap_links = False
            h.body_width = 0  # No wrapping
            h.unicode_snob = True  # Use unicode chars
            h.mark_code = True

            markdown = h.handle(cleaned_html)
            return ToolResult([ToolContentText(markdown)], is_error=False)
