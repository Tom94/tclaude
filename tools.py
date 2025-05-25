#!/usr/bin/env python3

"""
All functions in this file will be auto-converted to tool calls for Claude. Do not put any function into this file that should not be
directly callable by Claude.
"""

import requests


def fetch_url(url: str) -> str:
    """
    Fetch the content of a URL and return it as a string.

    :param url: The URL to fetch.
    :return: The content of the URL as a string.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text
