from __future__ import annotations

import pytest

from dale import errors as dale_errors
from dale.errors import DaleError

ERROR_CLASSES = [
    dale_errors.HandleNotFoundError,
    dale_errors.FieldNotFoundError,
    dale_errors.TypeMismatchError,
    dale_errors.DuplicateKeyError,
    dale_errors.DivisionByZeroError,
    dale_errors.RegistryLimitError,
    dale_errors.ToolCallLimitError,
    dale_errors.LoadError,
    dale_errors.InvalidParamsError,
    dale_errors.OperationNotFoundError,
    dale_errors.InternalError,
]


@pytest.mark.parametrize("error_cls", ERROR_CLASSES)
def test_to_payload_shape(error_cls):
    err = error_cls("something went wrong", details={"field": "x"})
    payload = err.to_payload()
    assert payload == {
        "status": "error",
        "code": error_cls.code,
        "message": "something went wrong",
        "details": {"field": "x"},
    }


@pytest.mark.parametrize("error_cls", ERROR_CLASSES)
def test_payload_never_contains_traceback_or_source_path_markers(error_cls):
    err = error_cls("a generic problem occurred")
    payload_str = str(err.to_payload())
    assert "Traceback" not in payload_str
    assert ".py" not in payload_str
    assert "site-packages" not in payload_str


def test_each_error_class_has_distinct_code():
    codes = [cls.code for cls in ERROR_CLASSES]
    assert len(codes) == len(set(codes))


def test_dale_error_is_exception_subclass():
    assert issubclass(DaleError, Exception)


def test_details_defaults_to_empty_dict():
    err = dale_errors.InternalError("failed")
    assert err.to_payload()["details"] == {}
