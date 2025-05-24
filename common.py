#!/usr/bin/env python3


PROMPT = " "


def prompt(prefix: str, pretty: bool) -> str:
    result = f"{prefix}{PROMPT}"
    if pretty:
        # result = f"\033[1;35m{result}\033[0m"
        result = f"\033[35m{result}\033[0m"
    return result


def deduce_model_name(model: str) -> str:
    if "opus" in model:
        if "3" in model:
            return "claude-3-opus-latest"
        return "claude-opus-4-0"
    elif "sonnet" in model:
        if "3.5" in model:
            return "claude-3-5-sonnet-latest"
        elif "3.7" in model:
            return "claude-3-7-sonnet-latest"
        elif "3" in model:
            return "claude-3-sonnet-latest"
        return "claude-sonnet-4-0"
    elif "haiku" in model:
        return "claude-3-5-haiku-latest"
    return model


def friendly_model_name(model: str) -> str:
    """
    Convert a model name to a more user-friendly format.
    """
    if not model.startswith("claude-"):
        return model

    kind = None
    if "opus" in model:
        kind = "opus"
    elif "sonnet" in model:
        kind = "sonnet"
    elif "haiku" in model:
        kind = "haiku"

    if kind is None:
        return model

    # Double-digit versions first, then single-digit
    version = None
    if "3-7" in model:
        version = "3.7"
    elif "3-5" in model:
        version = "3.5"
    elif "3" in model:
        version = "3.0"
    elif "4" in model:
        version = "4.0"

    return f"{kind} {version}"


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
