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

from collections.abc import Mapping, Sequence
from typing import cast

type JSON = Mapping[str, JSON] | Sequence[JSON] | str | int | float | bool | None


def get[T: JSON](d: JSON, key: str, target_type: type[T]) -> T | None:
    if isinstance(d, dict) and key in d:
        v = d[key]
        dummy = target_type()
        # list and dict are special cases where we know that if d if JSON, then v should also be JSON, even if we don't check the generic
        # type. The last clause is sufficient for str, int, float, bool, and None.
        if (
            isinstance(v, Sequence)
            and isinstance(dummy, Sequence)
            or isinstance(v, Mapping)
            and isinstance(dummy, Mapping)
            or isinstance(v, target_type)
        ):
            return cast(T, v)
    return None


def get_or_default[T: JSON](d: JSON, key: str, target_type: type[T]) -> T:
    """
    Get a typed value from a JSON-like dictionary, returning a default value if the key is not present or not the type.
    """
    return get_or(d, key, target_type())


def get_or[T: JSON](d: JSON, key: str, value: T) -> T:
    """
    Get a typed value from a JSON-like dictionary, returning a default value if the key is not present or not the type.
    """
    if isinstance(d, dict) and key in d:
        v = d[key]
        # list and dict are special cases where we know that if d if JSON, then v should also be JSON, even if we don't check the generic
        # type. The last clause is sufficient for str, int, float, bool, and None.
        if (
            (isinstance(v, Sequence) and isinstance(value, Sequence))
            or (isinstance(v, Mapping) and isinstance(value, Mapping))
            or type(v) is type(value)
        ):
            return cast(T, v)
    return value
