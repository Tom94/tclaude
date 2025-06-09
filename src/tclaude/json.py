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

from collections.abc import Mapping, Sequence
from types import UnionType
from typing import TypeAlias, cast, get_args, get_origin

# Using TypeAlias instead of defining a new type such that isinstance(obj, JSON) works as expected.
JSON: TypeAlias = Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None

# Allows for nested generic types, as well as unions. The type taken by `isinstance`.
ClassOrTuple: TypeAlias = type | tuple["ClassOrTuple", ...] | UnionType


def generic_is_instance(obj: JSON, target_type: ClassOrTuple) -> bool:
    """
    Check if an object is an instance of a generic type, including nested types.
    """
    origin = get_origin(target_type)
    if origin is None or target_type is UnionType:
        return isinstance(obj, target_type)

    args = get_args(target_type)
    if origin is list:
        T = cast(ClassOrTuple, args[0])
        return isinstance(obj, list) and all(generic_is_instance(item, T) for item in obj)
    elif origin is dict:
        K = cast(ClassOrTuple, args[0])
        V = cast(ClassOrTuple, args[1])
        return isinstance(obj, dict) and all(generic_is_instance(k, K) and generic_is_instance(v, V) for k, v in obj.items())
    elif origin is UnionType:
        assert args == get_args(JSON), "UnionType should only be used with JSON types"
        return isinstance(obj, (str, int, float, bool, type(None), list, dict))
    else:
        return False


def of_type_or_none[T: JSON](target_type: type[T], obj: JSON) -> T | None:
    """
    Safely cast an object to a target type, returning None if the object is not of the target type.
    This is useful for JSON-like structures where types may not match exactly.
    """
    if generic_is_instance(obj, target_type):
        return cast(T, obj)
    return None


def get[T: JSON](d: JSON, key: str, target_type: type[T]) -> T | None:
    if isinstance(d, dict) and key in d:
        v = d[key]
        return of_type_or_none(target_type, v)
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
    v = get(d, key, type(value))
    return v if v is not None else value
