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

from __future__ import annotations

import logging

logger = logging.getLogger(__package__)


class TokenCounter:
    def __init__(self, cache_creation_input_tokens: int = 0, cache_read_input_tokens: int = 0, input_tokens: int = 0, output_tokens: int = 0):
        self.cache_creation: int = cache_creation_input_tokens
        self.cache_read: int = cache_read_input_tokens
        self.input: int = input_tokens
        self.output: int = output_tokens

    def __add__(self, other: TokenCounter) -> TokenCounter:
        result = TokenCounter()
        result.cache_creation = self.cache_creation + other.cache_creation
        result.cache_read = self.cache_read + other.cache_read
        result.input = self.input + other.input
        result.output = self.output + other.output
        return result

    def cost(self, model: str) -> tuple[float, float, float, float]:
        cost_factor = 1.0
        if "opus" in model:
            cost_factor = 5.0
        elif "haiku" in model:
            cost_factor = 1.0 / 3.75
            if "3-haiku" in model:
                cost_factor *= 0.3

        # See https://docs.anthropic.com/en/docs/about-claude/models/overview#model-pricing
        price_per_minput_cache_creation = 3.75 * cost_factor
        price_per_minput_cache_read = 0.3 * cost_factor
        price_per_minput = 3.0 * cost_factor
        price_per_moutput = 15.0 * cost_factor

        cache_creation_cost = (self.cache_creation / 1000000) * price_per_minput_cache_creation
        cache_read_cost = (self.cache_read / 1000000) * price_per_minput_cache_read
        input_cost = (self.input / 1000000) * price_per_minput
        output_cost = (self.output / 1000000) * price_per_moutput

        return cache_creation_cost, cache_read_cost, input_cost, output_cost

    def total_cost(self, model: str) -> float:
        cache_creation_cost, cache_read_cost, input_cost, output_cost = self.cost(model)
        return cache_creation_cost + cache_read_cost + input_cost + output_cost

    def print_tokens(self):
        logger.info(f"Tokens: cache_creation={self.cache_creation} cache_read={self.cache_read} input={self.input} output={self.output}")

    def print_cost(self, model: str):
        cache_creation_cost, cache_read_cost, input_cost, output_cost = self.cost(model)
        logger.info(f"Cost: cache_creation=${cache_creation_cost:.2f} cache_read=${cache_read_cost:.2f} input=${input_cost:.2f} output=${output_cost:.2f}")
