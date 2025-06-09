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

import time

SPINNER_FPS = 10  # Frames per second for spinner animation


def spinner() -> str:
    """
    Return a spinner frame based on the index.
    """
    frames = [
        "⠲",
        "⠴",
        "⠦",
        "⠖",
    ]
    # frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"]
    sidx = int(time.perf_counter() * SPINNER_FPS)
    return frames[sidx % len(frames)]
