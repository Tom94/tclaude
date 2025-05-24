#!/usr/bin/env python3

from setuptools import setup

setup(
    name="tai — Terminal AI",
    version="0.1",
    py_modules=["chat", "print", "prompt", "common"],
    entry_points={
        "console_scripts": [
            "tai=chat:main",
        ],
    },
)
