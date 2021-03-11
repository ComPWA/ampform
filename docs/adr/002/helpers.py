# pylint: disable=import-error

from typing import List

import sympy as sp  # pyright: reportMissingImports=false


class StateTransitionGraph:
    pass


def hankel1(angular_momentum: sp.Symbol, x: sp.Symbol) -> sp.Expr:
    x_squared = x ** 2
    return sp.Piecewise(
        (
            1,
            sp.Eq(angular_momentum, 0),
        ),
        (
            1 + x_squared,
            sp.Eq(angular_momentum, 1),
        ),
        (
            9 + x_squared * (3 + x_squared),
            sp.Eq(angular_momentum, 2),
        ),
        (
            225 + x_squared * (45 + x_squared * (6 + x_squared)),
            sp.Eq(angular_momentum, 3),
        ),
        (
            1575 + x_squared * (135 + x_squared * (10 + x_squared)),
            sp.Eq(angular_momentum, 4),
        ),
    )


def blatt_weisskopf(
    q: sp.Symbol, q_r: sp.Symbol, angular_momentum: sp.Symbol
) -> sp.Expr:
    return sp.sqrt(
        abs(hankel1(angular_momentum, q)) ** 2
        / abs(hankel1(angular_momentum, q_r)) ** 2
    )


def two_body_momentum_squared(
    m_d: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
) -> sp.Expr:
    return (
        (m_d ** 2 - (m_a + m_b) ** 2)
        * (m_d ** 2 - (m_a - m_b) ** 2)
        / (4 * m_d ** 2)
    )


def determine_attached_final_state(  # pylint: disable=unused-argument
    graph: StateTransitionGraph,
    edge_id: int,
) -> List[int]:
    mock = [3, 4]
    return mock
