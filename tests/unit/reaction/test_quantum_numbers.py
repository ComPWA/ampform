# pylint: disable=eval-used, no-self-use
import typing
from copy import deepcopy

import pytest

from expertsystem.reaction.quantum_numbers import Parity, _to_fraction


class TestParity:
    @staticmethod
    def test_init_and_eq():
        parity = Parity(+1)
        assert parity == +1
        assert int(parity) == +1

    @typing.no_type_check  # https://github.com/python/mypy/issues/4610
    @staticmethod
    def test_comparison():
        neg = Parity(-1)
        pos = Parity(+1)
        assert pos > 0
        assert neg < 0
        assert neg < pos
        assert neg <= pos
        assert pos > neg
        assert pos >= neg
        assert pos >= 0
        assert neg <= 0
        assert 0 < pos  # pylint: disable=misplaced-comparison-constant

    @staticmethod
    def test_hash():
        neg = Parity(-1)
        pos = Parity(+1)
        assert {pos, neg, deepcopy(pos)} == {neg, pos}

    @staticmethod
    def test_neg():
        parity = Parity(+1)
        flipped_parity = -parity
        assert flipped_parity.value == -parity.value

    @pytest.mark.parametrize("value", [-1, +1])
    def test_repr(self, value):
        parity = Parity(value)
        from_repr = eval(repr(parity))
        assert from_repr == parity

    @staticmethod
    def test_exceptions():
        with pytest.raises(TypeError):
            Parity(1.2)  # type: ignore
        with pytest.raises(ValueError):
            Parity(0)


@pytest.mark.parametrize(
    "value, render_plus, expected",
    [
        (0, False, "0"),
        (0, True, "0"),
        (-1, False, "-1"),
        (-1, True, "-1"),
        (1, False, "1"),
        (1, True, "+1"),
        (1.0, True, "+1"),
        (0.5, True, "+1/2"),
        (-0.5, True, "-1/2"),
        (+1.5, False, "3/2"),
        (+1.5, True, "+3/2"),
        (-1.5, True, "-3/2"),
    ],
)
def test_to_fraction(value, render_plus: bool, expected: str):
    assert _to_fraction(value, render_plus) == expected
