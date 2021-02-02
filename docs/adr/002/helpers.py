# pylint: disable=import-error

from typing import List

import sympy as sy  # pyright: reportMissingImports=false


class StateTransitionGraph:
    pass


def hankel1(angular_momentum: sy.Symbol, x: sy.Symbol) -> sy.Expr:
    x_squared = x ** 2
    return sy.Piecewise(
        (
            1,
            sy.Eq(angular_momentum, 0),
        ),
        (
            1 + x_squared,
            sy.Eq(angular_momentum, 1),
        ),
        (
            9 + x_squared * (3 + x_squared),
            sy.Eq(angular_momentum, 2),
        ),
        (
            225 + x_squared * (45 + x_squared * (6 + x_squared)),
            sy.Eq(angular_momentum, 3),
        ),
        (
            1575 + x_squared * (135 + x_squared * (10 + x_squared)),
            sy.Eq(angular_momentum, 4),
        ),
    )


def blatt_weisskopf(
    q: sy.Symbol, q_r: sy.Symbol, angular_momentum: sy.Symbol
) -> sy.Expr:
    return sy.sqrt(
        abs(hankel1(angular_momentum, q)) ** 2
        / abs(hankel1(angular_momentum, q_r)) ** 2
    )


def two_body_momentum_squared(
    m_d: sy.Symbol, m_a: sy.Symbol, m_b: sy.Symbol
) -> sy.Expr:
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
