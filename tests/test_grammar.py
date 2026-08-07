from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from dale.errors import DivisionByZeroError, TypeMismatchError
from dale.grammar import (
    And,
    Comparison,
    ComputedField,
    ConstRef,
    FieldRef,
    Not,
    NullComparison,
    Or,
    Predicate,
    apply_computed_field,
    matches,
    render_predicate,
    resolve_priority,
)

predicate_adapter = TypeAdapter(Predicate)


@pytest.mark.parametrize(
    "op,value,record,expected",
    [
        ("==", 5, {"x": 5}, True),
        ("==", 5, {"x": 6}, False),
        ("!=", 5, {"x": 6}, True),
        ("<", 5, {"x": 4}, True),
        ("<=", 5, {"x": 5}, True),
        (">", 5, {"x": 6}, True),
        (">=", 5, {"x": 5}, True),
        ("in", [1, 2, 3], {"x": 2}, True),
        ("not_in", [1, 2, 3], {"x": 9}, True),
        ("starts_with", "ab", {"x": "abcdef"}, True),
        ("ends_with", "ef", {"x": "abcdef"}, True),
        ("contains", "cd", {"x": "abcdef"}, True),
    ],
)
def test_comparison_ops(op, value, record, expected):
    comp = Comparison(field="x", op=op, value=value)
    assert matches(record, comp) is expected


def test_missing_field_equality_treated_as_none_not_error():
    comp = Comparison(field="missing", op="==", value=None)
    assert matches({"x": 1}, comp) is True


def test_missing_field_ordering_raises_type_mismatch():
    comp = Comparison(field="missing", op=">", value=5)
    with pytest.raises(TypeMismatchError):
        matches({"x": 1}, comp)


def test_incompatible_type_ordering_raises_type_mismatch():
    comp = Comparison(field="x", op=">", value="not a number")
    with pytest.raises(TypeMismatchError):
        matches({"x": 5}, comp)


def test_is_null_matches_missing_or_none_field():
    comp = NullComparison(field="x", op="is_null")
    assert matches({"x": None}, comp) is True
    assert matches({}, comp) is True
    assert matches({"x": 1}, comp) is False


def test_is_not_null_matches_present_field():
    comp = NullComparison(field="x", op="is_not_null")
    assert matches({"x": 1}, comp) is True
    assert matches({"x": None}, comp) is False
    assert matches({}, comp) is False


def test_null_comparison_rejects_value_field():
    with pytest.raises(ValidationError):
        predicate_adapter.validate_python({"field": "x", "op": "is_null", "value": None})


def test_comparison_rejects_missing_value_for_value_op():
    # `value` stays genuinely required in Comparison's own schema -- adding
    # is_null/is_not_null must not make `==`/etc. silently tolerate omitting
    # it (that was the whole bug NullComparison exists to close).
    with pytest.raises(ValidationError):
        predicate_adapter.validate_python({"field": "x", "op": "=="})


def test_render_null_comparison():
    assert render_predicate(NullComparison(field="x", op="is_null")) == "x IS NULL"
    assert render_predicate(NullComparison(field="x", op="is_not_null")) == "x IS NOT NULL"


def test_predicate_union_disambiguates_null_comparison_from_comparison():
    pred = predicate_adapter.validate_python({"field": "x", "op": "is_null"})
    assert isinstance(pred, NullComparison)

    pred2 = predicate_adapter.validate_python({"field": "x", "op": "==", "value": 1})
    assert isinstance(pred2, Comparison)


def test_and_or_not_composition():
    record = {"x": 10, "y": "hello"}
    pred = And(
        and_=[
            Comparison(field="x", op=">", value=5),
            Or(
                or_=[
                    Comparison(field="y", op="==", value="hello"),
                    Comparison(field="y", op="==", value="world"),
                ]
            ),
        ]
    )
    assert matches(record, pred) is True

    not_pred = Not(not_=Comparison(field="x", op="==", value=10))
    assert matches(record, not_pred) is False


def test_render_predicate_single_comparison():
    pred = Comparison(field="x", op=">", value=5)
    assert render_predicate(pred) == "x > 5"


def test_render_predicate_nested_and_or_not():
    pred = And(
        and_=[
            Comparison(field="in_stock", op="==", value=True),
            Not(
                not_=Or(
                    or_=[
                        Comparison(field="category", op="==", value="Furniture"),
                        Comparison(field="price", op="<", value=10),
                    ]
                )
            ),
        ]
    )
    rendered = render_predicate(pred)
    assert rendered == "(in_stock == True AND NOT (category == 'Furniture' OR price < 10))"
    # Round-trips through re-parsing the rendered structure's source predicate
    # correctly evaluates against a record, i.e. this isn't just cosmetic —
    # the same tree still matches() as expected.
    assert matches({"in_stock": True, "category": "Tools", "price": 25}, pred) is True
    assert matches({"in_stock": True, "category": "Furniture", "price": 25}, pred) is False


def test_predicate_union_disambiguates_from_dict():
    comp = predicate_adapter.validate_python({"field": "x", "op": "==", "value": 1})
    assert isinstance(comp, Comparison)

    andp = predicate_adapter.validate_python(
        {"and": [{"field": "x", "op": "==", "value": 1}]}
    )
    assert isinstance(andp, And)

    orp = predicate_adapter.validate_python({"or": [{"field": "x", "op": "==", "value": 1}]})
    assert isinstance(orp, Or)

    notp = predicate_adapter.validate_python({"not": {"field": "x", "op": "==", "value": 1}})
    assert isinstance(notp, Not)


def test_predicate_rejects_unknown_extra_field():
    with pytest.raises(ValidationError):
        predicate_adapter.validate_python(
            {"field": "x", "op": "==", "value": 1, "extra": True}
        )


def test_predicate_rejects_regex_style_ops():
    # Regex was explicitly rejected (ReDoS risk) in favor of bounded literal
    # ops — confirm the grammar has no "matches"/"regex" operator at all.
    with pytest.raises(ValidationError):
        predicate_adapter.validate_python({"field": "x", "op": "regex", "value": ".*"})


def test_computed_field_arithmetic():
    record = {"price": 10, "cost": 4}
    computed = ComputedField(
        **{
            "as": "margin",
            "op": "subtract",
            "left": FieldRef(field="price"),
            "right": FieldRef(field="cost"),
        }
    )
    assert apply_computed_field(record, computed) == 6


def test_computed_field_with_const():
    record = {"price": 10}
    computed = ComputedField(
        **{
            "as": "doubled",
            "op": "multiply",
            "left": FieldRef(field="price"),
            "right": ConstRef(const=2),
        }
    )
    assert apply_computed_field(record, computed) == 20


def test_computed_field_division_by_zero():
    record = {"a": 10, "b": 0}
    computed = ComputedField(
        **{"as": "r", "op": "divide", "left": FieldRef(field="a"), "right": FieldRef(field="b")}
    )
    with pytest.raises(DivisionByZeroError):
        apply_computed_field(record, computed)


def test_computed_field_missing_operand_raises():
    record = {"a": 10}
    computed = ComputedField(
        **{
            "as": "r",
            "op": "add",
            "left": FieldRef(field="a"),
            "right": FieldRef(field="missing"),
        }
    )
    with pytest.raises(TypeMismatchError):
        apply_computed_field(record, computed)


def test_resolve_priority_picks_highest():
    assert resolve_priority(["Allow", "Deny"], ["Deny", "Allow"]) == "Deny"
    assert resolve_priority(["Allow"], ["Deny", "Allow"]) == "Allow"


def test_resolve_priority_no_match_raises():
    with pytest.raises(ValueError):
        resolve_priority(["Unknown"], ["Deny", "Allow"])


def test_null_comparison_carries_its_history_as_a_comment_not_a_docstring():
    """Pydantic promotes a model's `__doc__` into its JSON-Schema
    `description`, and `NullComparison` is reachable from `Predicate`, which
    the tool schema publishes 3-4 times per request. As a docstring, its
    969-char account of a *rejected* design cost 1,101 tokens on every single
    request — the most expensive comment in the codebase, paid for by the model
    rather than read by anyone.

    Pinned at the source, not only at the schema (see
    tests/test_agent.py::test_null_comparison_ships_no_description), because the
    intent is "this history lives in a comment" — and the obvious, well-meant
    edit that breaks it is someone promoting the comment back to a docstring
    where it looks more at home."""
    assert NullComparison.__doc__ is None
    assert "description" not in NullComparison.model_json_schema()
    # The history itself is still there, in the source, immediately above.
    import inspect

    import dale.grammar

    source = inspect.getsource(dale.grammar)
    assert "a plain `value: Any = None` was tried first" in source
