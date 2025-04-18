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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


#####################################################
### CODE
def check_kwargs(
    kwargs: dict[str, object],
    func: Callable[..., object],
) -> dict[str, object]:
    """Type check kwargs against function signature."""
    func_signature = inspect.signature(func)
    filtered_kwargs: dict[str, object] = {}
    for k, v in kwargs.items():
        if k in func_signature.parameters:
            func_param = func_signature.parameters.get(k)
            if func_param is None:
                msg = f"Parameter '{k}' not found in function signature"
                raise TypeError(msg)

            expected_type = func_param.annotation
            if (
                expected_type is not inspect._empty  # type:ignore[reportPrivateUsage, unused-ignore]  # noqa: SLF001 <it's a used ignore of private usage warning>
                and expected_type is not Any
                and not isinstance(v, expected_type)
            ):
                msg = f"Argument '{k}' must be of type {expected_type}, got {type(v)}"
                raise TypeError(msg)
        filtered_kwargs[k] = v
    return filtered_kwargs
