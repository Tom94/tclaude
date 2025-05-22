#!/usr/bin/env python3


PROMPT = "\033[1;35m \033[0m"


def pretty_print_md(string):
    """
    Turn string pretty by piping it through bat
    """
    try:
        import subprocess

        # Use bat to pretty print the string
        process = subprocess.Popen(
            ["bat", "--color=always", "--paging=never", "--style=plain", "--language=markdown"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output, error = process.communicate(input=string.encode("utf-8"))

        if process.returncode != 0:
            raise Exception(f"Error: {error.decode('utf-8')}")
        return output.decode("utf-8")
    except FileNotFoundError:
        # If bat is not installed, fall back to regular print
        return string
