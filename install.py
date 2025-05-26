#!/usr/bin/env python3

# tai -- Terminal AI
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

import argparse
import os
import subprocess
import shutil
import platform


class PyInstallerBuilder:
    def __init__(self, script_name, app_name):
        self.script_name = script_name
        self.app_name = app_name
        self.platform = platform.system().lower()

    def clean(self):
        """Clean previous builds"""
        for folder in ["build", "dist"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Removed {folder}/")

    def build(self, icon=None):
        """Build the executable"""
        cmd = ["pyinstaller"]
        cmd.append("--console")
        cmd.append("--onefile")

        if icon:
            cmd.extend(["--icon", icon])

        cmd.extend(["--name", self.app_name])
        cmd.append(self.script_name)

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"\n✓ Build successful!")
            print(f"Executable location: dist/{self.app_name}")
        else:
            print(f"\n✗ Build failed with code {result.returncode}")

        return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Build a standalone executable for the TAI chat application.")
    parser.add_argument("--prefix", default="~/.local/bin", help="Prefix for the installation directory (default: ~/local/bin)")
    parser.add_argument("--icon", help="Path to the icon file for the executable (optional)")

    args = parser.parse_args()

    bin_name = "tai"

    builder = PyInstallerBuilder("tai.py", bin_name)
    builder.clean()
    success = builder.build(icon=args.icon)

    if success:
        # Copy binary to the specified prefix
        install_dir = os.path.expanduser(args.prefix)
        if not os.path.exists(install_dir):
            os.makedirs(install_dir)
        bin_path = os.path.join(install_dir, bin_name)
        shutil.move(os.path.join("dist", bin_name), bin_path)
        print(f"Moved executable to: {bin_path}")

        shutil.rmtree("dist")
        os.remove("tai.spec")
        print("Cleaned up build artifacts")


if __name__ == "__main__":
    main()
