import warnings
from typing import Any

import pytest

from automlagent.utils.typing_utils import check_kwargs


def func_with_types(a: int, b: str, c: float) -> None:
    pass


def func_with_any(a: Any, b: int) -> None:  # noqa: ANN401
    pass


def func_with_no_annotations(a, b):  # type: ignore[reportUnknownParameterType, reportMissingParameterType] # noqa: ANN001, ANN201
    pass


def func_with_defaults(a: int, b: str = "default"):  # type: ignore[reportUnknownParameterType, reportMissingParameterType] # noqa: ANN201
    pass


def func_no_params():  # type: ignore[reportUnknownParameterType, reportMissingParameterType] # noqa: ANN201
    pass


class TestCheckKwargs:
    def test_passes_correct_kwargs(self) -> None:
        """Should return kwargs unchanged when all types match."""
        kwargs: dict[str, object] = {"a": 1, "b": "s", "c": 2.5}
        result = check_kwargs(kwargs, func_with_types)
        assert result == kwargs

    @pytest.mark.parametrize(
        ("kwargs", "param", "wrong_value"),
        [
            ({"a": "not-an-int", "b": "s", "c": 2.5}, "a", "not-an-int"),
            ({"a": 1, "b": 2, "c": 2.5}, "b", 2),
            ({"a": 1, "b": "s", "c": "not-a-float"}, "c", "not-a-float"),
        ],
    )
    def test_type_mismatch_raises_typeerror(
        self, kwargs: dict[str, object], param: str, wrong_value: object
    ) -> None:
        """Should raise TypeError if kwarg type does not match annotation."""
        with pytest.raises(TypeError) as excinfo:
            check_kwargs(kwargs, func_with_types)
        assert param in str(excinfo.value)
        assert str(type(wrong_value)) in str(excinfo.value)

    def test_param_not_in_signature_skips(self) -> None:
        """Should raise TypeError if param not in function signature."""
        kwargs: dict[str, object] = {"a": 38, "b": "potato :3", "not_a_param": 123}
        filtered = check_kwargs(kwargs, func_with_types)
        assert filtered == {
            "a": 38,
            "b": "potato :3",
        }

    def test_no_type_annotation_accepts_any(self) -> None:
        """Should accept any type for params with no annotation."""
        kwargs: dict[str, object] = {"a": 1, "b": object()}
        result = check_kwargs(kwargs, func_with_no_annotations)  # type: ignore[reportUnknownParameterType, reportMissingParameterType]
        assert result == kwargs

    def test_any_annotation_accepts_any(self) -> None:
        """Should accept any type for params annotated as Any."""
        kwargs: dict[str, object] = {"a": object(), "b": 5}
        result = check_kwargs(kwargs, func_with_any)
        assert result == kwargs

    def test_empty_kwargs(self) -> None:
        """Should return empty dict when kwargs is empty."""
        result = check_kwargs({}, func_with_types)
        assert result == {}

    def test_func_with_no_params(self) -> None:
        """Should return empty dict when function has no parameters."""
        kwargs: dict[str, object] = {"a": 1, "b": 2}
        result = check_kwargs(kwargs, func_no_params)
        assert result == {}

    def test_func_with_defaults(self) -> None:
        """Should accept defaulted parameters when types match."""
        kwargs: dict[str, object] = {"a": 5, "b": "custom"}
        result = check_kwargs(kwargs, func_with_defaults)
        assert result == kwargs
        # b is optional, should also work if omitted
        kwargs = {"a": 5}
        result = check_kwargs(kwargs, func_with_defaults)
        assert result == kwargs

    def test_warns_on_filtered_kwargs(self) -> None:
        """Should warn and list filtered kwargs not accepted by func."""
        kwargs: dict[str, object] = {"a": 1, "b": "s", "c": 3.5, "not_a_param": 42}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_kwargs(kwargs, func_with_defaults)
            assert result == {"a": 1, "b": "s"}
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Filtered out 2 kwargs" in str(w[0].message)
            assert "c" in str(w[0].message)
            assert "not_a_param" in str(w[0].message)

    def test_kwargs_passed_through_when_has_varkwargs(self) -> None:
        """Should not filter or warn about extra kwargs if func accepts **kwargs."""

        def func_with_var_keyword(a: int, **kwargs: object) -> None:
            pass

        kwargs = {"a": 1, "b": "extra", "c": 3.5, "d": object()}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_kwargs(kwargs, func_with_var_keyword)
            # All kwargs should be passed through
            assert result == kwargs
            # No warnings should be issued
            assert len(w) == 0

    def test_type_check_on_explicit_params_with_varkwargs(self) -> None:
        """Type-check explicit params, but allow any type for untyped **kwargs."""

        def func_with_varkwargs(a: int, b: float, **kwargs: object) -> None:
            pass

        # Correct types for a and b, extra kwargs
        kwargs = {"a": 42, "b": 3.14, "c": "string", "d": object()}
        result = check_kwargs(kwargs, func_with_varkwargs)
        assert result == kwargs

        # Incorrect type for explicit param
        bad_kwargs: dict[str, object] = {"a": "not-an-int", "b": 3.14, "c": "string"}
        with pytest.raises(TypeError) as excinfo:
            check_kwargs(bad_kwargs, func_with_varkwargs)
        assert "a" in str(excinfo.value)

    def test_type_check_on_varkwargs(self) -> None:
        """Type-check explicit params, but allow any type for untyped **kwargs."""

        def func_with_varkwargs(a: int, b: float, **kwargs: str) -> None:
            pass

        # Correct types for a and b, extra kwargs
        kwargs = {"a": 42, "b": 3.14, "c": "string", "d": object()}
        with pytest.raises(TypeError) as excinfo:
            check_kwargs(kwargs, func_with_varkwargs)
        assert "a" in str(excinfo.value)
        assert str(type(object())) in str(excinfo.value)

    def test_var_positional_does_not_affect_kwargs(self) -> None:
        """*args in signature does not affect kwargs filtering."""

        def func_with_args_and_kwargs(a: int, *args: object, b: float = 0.0) -> None:
            pass

        kwargs: dict[str, object] = {"a": 1, "b": 2.0, "extra": 99}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_kwargs(kwargs, func_with_args_and_kwargs)
            # 'extra' should be filtered out and warned
            assert result == {"a": 1, "b": 2.0}
            assert len(w) == 1
            assert "extra" in str(w[0].message)

    def test_shallow_type_check_list(self) -> None:
        """Should accept list for list[int] annotation (shallow check)."""

        def func(a: list[int]) -> None:
            pass

        kwargs: dict[str, object] = {"a": [1, 2, 3]}
        result = check_kwargs(kwargs, func)
        assert result == kwargs
        # Should reject non-list
        bad_kwargs: dict[str, object] = {"a": {1, 2, 3}}
        with pytest.raises(TypeError):
            check_kwargs(bad_kwargs, func)

    def test_shallow_type_check_dict(self) -> None:
        """Should accept dict for dict[str, float] annotation (shallow check)."""

        def func(a: dict[str, float]) -> None:
            pass

        kwargs: dict[str, object] = {"a": {"x": 1.0, "y": 2.0}}
        result = check_kwargs(kwargs, func)
        assert result == kwargs
        # Should reject non-dict
        bad_kwargs: dict[str, object] = {"a": [1.0, 2.0]}
        with pytest.raises(TypeError):
            check_kwargs(bad_kwargs, func)

    def test_shallow_type_check_user_generic(self) -> None:
        """Should accept user generic outer type (shallow check)."""
        from collections import deque

        def func(a: deque[int]) -> None:
            pass

        kwargs: dict[str, object] = {"a": deque([1, 2, 3])}
        result = check_kwargs(kwargs, func)
        assert result == kwargs
        # Should reject non-deque
        bad_kwargs: dict[str, object] = {"a": [1, 2, 3]}
        with pytest.raises(TypeError):
            check_kwargs(bad_kwargs, func)
