#!/usr/bin/env python3

"""
All functions in this file will be auto-converted to tool calls for Claude. Do not put any function into this file that should not be
directly callable by Claude.
"""

import requests
import html2text

from bs4 import BeautifulSoup


def fetch_url(url: str) -> str:
    """
    Fetch the content of a URL and return it as a markdown string. The raw HTML text is cleaned up by removing script, style, and other
    non-content elements, followed by conversion to markdown format.

    :param url: The URL to fetch.
    :return: The content of the URL as a markdown string.
    """
    response = requests.get(url, timeout=5)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Remove script and style elements
    for script_or_style in soup(["script", "style", "meta", "link", "noscript", "iframe", "embed", "object"]):
        script_or_style.decompose()

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
    return markdown
