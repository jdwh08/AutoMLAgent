#####################################################
# AutoMLAgent [OVERKILL TYPING UTILITIES]
# ####################################################
# Jonathan Wang

# ABOUT:
# This process is a POC for automating
# the modelling process.

"""Utility functions for extra typing. Probably overkill."""


#####################################################
### BOARD

# NOTE(jdwh08): isinstance(v, expected_type)
# not working for Union[] types? but yes for | (pipe)
# so if we stick to Python 3.10+ should be fine
# <https://peps.python.org/pep-0604/>

#####################################################
### IMPORTS

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, get_origin

if TYPE_CHECKING:
    from collections.abc import Callable


#####################################################
### CODE
def _shallow_type_check(value: object, expected_type: object) -> bool:
    """Shallow type check for generics: only checks outer type."""
    origin = get_origin(expected_type)
    if origin is not None:
        # e.g., list[int], dict[str, float], MyGeneric[T]
        return isinstance(value, origin)
    return isinstance(value, expected_type)  # type: ignore[reportArgumentType]


def check_kwargs(
    kwargs: dict[str, object],
    func: Callable[..., object],
) -> dict[str, object]:
    """Dynamically type check kwargs against function signature.

    Useful for passing kwargs from outer **kwargs into inner function call.

    Args:
        kwargs (dict[str, object]): Keyword arguments to check.
        func (Callable[..., object]): Function to check against.

    Returns:
        dict[str, object]: Keeps only kwargs which are in function signature,
            or all kwargs if **kwargs
            or all kwargs that match **kwargs type if **kwargs is not Any / object

    Raises:
        TypeError: if any parameter is of different type than function signature

    Notes:
        Use this function when passing from/to outside code or user-supplied info.

        Stop using this dang function for internal kwargs-kwargs passing where
        we control both sides!

    """
    sig = inspect.signature(func)
    params = sig.parameters
    explicit_param_names = {
        k for k, p in params.items() if p.kind != inspect.Parameter.VAR_KEYWORD
    }
    varkw_param = next(
        (p for p in params.values() if p.kind == inspect.Parameter.VAR_KEYWORD), None
    )
    varkw_type = varkw_param.annotation if varkw_param is not None else Any

    accepted_kwargs: dict[str, object] = {}
    filtered_keys: list[str] = []

    for k, v in kwargs.items():
        # Explicit parameters
        if k in explicit_param_names:
            expected_type = params[k].annotation
            if expected_type is not inspect._empty and expected_type is not Any:  # type: ignore[reportPrivateUsage] # noqa: SIM102, SLF001
                if not _shallow_type_check(v, expected_type):
                    msg = (
                        f"Argument '{k}' must be of type {expected_type}, got {type(v)}"
                    )
                    raise TypeError(msg)
            accepted_kwargs[k] = v

        # **kwargs handling
        elif varkw_param is not None:
            # Type-check if **kwargs annotation is not Any or _empty
            if varkw_type is not inspect._empty and varkw_type is not Any:  # type: ignore[reportPrivateUsage] # noqa: SIM102, SLF001
                if not _shallow_type_check(v, varkw_type):
                    msg = (
                        f"Extra kwarg '{k}' must be of type {varkw_type}, got {type(v)}"
                    )
                    raise TypeError(msg)
            accepted_kwargs[k] = v
        else:
            filtered_keys.append(k)

    if filtered_keys:
        msg = f"Filtered out {len(filtered_keys)} kwargs: {', '.join(filtered_keys)}"
        warnings.warn(msg, UserWarning, stacklevel=2)

    return accepted_kwargs
