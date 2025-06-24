#!/usr/bin/env python3

import re
import sys
import os
from datetime import datetime
import subprocess


def get_git_hash():
    """Get short git commit hash."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_current_version(init_file: str) -> str:
    """Extract current version from __init__.py file."""
    with open(init_file, "r") as f:
        content = f.read()

    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {init_file}")

    return match.group(1)


def generate_ci_version(base_version: str) -> str:
    """Generate PEP 440 CI version with format: X.Y.Z.devYYYYMMDDHHMMSS+hash"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    git_hash = get_git_hash()
    # return f"{base_version}.dev{timestamp}+{git_hash}"

    # Local versions (+hash) are not allowed in published packages (even on testpypi), so we omit the suffix
    return f"{base_version}.dev{timestamp}"


def patch_version(init_file: str, new_version: str):
    """Replace __version__ in __init__.py with new version."""
    with open(init_file, "r") as f:
        content = f.read()

    new_content = re.sub(r'(__version__\s*=\s*)["\'][^"\']+["\']', rf'\1"{new_version}"', content)

    with open(init_file, "w") as f:
        _ = f.write(new_content)


def main():
    init_file = os.path.join("src", "tclaude", "__init__.py")

    if not os.path.exists(init_file):
        print(f"Error: {init_file} not found")
        sys.exit(1)

    try:
        base_version = get_current_version(init_file)
        print(f"Current version: {base_version}")

        ci_version = generate_ci_version(base_version)
        print(f"CI version: {ci_version}")

        patch_version(init_file, ci_version)
        print(f"Updated {init_file} with version {ci_version}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
