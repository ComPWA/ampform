import pytest

from expertsystem.reaction import InteractionTypes as IT  # noqa: N817
from expertsystem.reaction import _determine_interaction_types


@pytest.mark.parametrize(
    "description, expected",
    [
        ("all", {IT.STRONG, IT.WEAK, IT.EM}),
        ("EM", {IT.EM}),
        ("electromagnetic", {IT.EM}),
        ("electro-weak", {IT.EM, IT.WEAK}),
        ("ew", {IT.EM, IT.WEAK}),
        ("w", {IT.WEAK}),
        ("strong", {IT.STRONG}),
        ("only strong", {IT.STRONG}),
        ("S", {IT.STRONG}),
        (["e", "s", "w"], {IT.STRONG, IT.WEAK, IT.EM}),
        ("S", {IT.STRONG}),
        ("strong and EM", {IT.STRONG, IT.EM}),
        ("", ValueError),
        ("non-existing", ValueError),
    ],
)
def test_determine_interaction_types(description, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            assert _determine_interaction_types(description)
    else:
        assert _determine_interaction_types(description) == expected
