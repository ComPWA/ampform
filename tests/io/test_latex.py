from textwrap import dedent

import pytest
import sympy as sp

from ampform.io import aslatex

a, b, x, y = sp.symbols("a b x y")


def test_complex():
    assert aslatex(1.2 - 5.3j) == "1.2-5.3i"
    assert aslatex(1.2 - 5j) == "1.2-5i"
    assert aslatex(1 + 1j) == "1+1i"


def test_expr():
    x, y, z = sp.symbols("x:z")
    expr = x + y + z
    assert aslatex(expr) == "x + y + z"
    assert aslatex(expr, terms_per_line=0) == "x + y + z"
    assert aslatex(expr, terms_per_line=3) == "x + y + z"

    expected = dedent(R"""
    \begin{array}{l}
      x \\
      \; + \; y \\
      \; + \; z \\
    \end{array}
    """)

    assert aslatex(expr, terms_per_line=1) == expected.strip()
    expected = dedent(R"""
    \begin{array}{l}
      x + y \\
      \; + \; z \\
    \end{array}
    """)
    assert aslatex(expr, terms_per_line=2) == expected.strip()


def test_iterable():
    items = [
        a * x**2 + b,
        3.0,
        2 - 1.3j,
    ]
    iterable = iter(items)
    latex = aslatex(iterable)
    expected = R"""
    \begin{array}{c}
      a x^{2} + b \\
      3.0 \\
      2-1.3i \\
    \end{array}
    """
    assert latex == dedent(expected).strip()


@pytest.mark.parametrize("terms_per_line", [0, 2])
def test_mapping(terms_per_line: int):
    definitions = {
        y: a * x**2 + b,
        a: 3.0,
        b: 2 - 1.3j,
    }
    latex = aslatex(definitions, terms_per_line=terms_per_line)
    expected = R"""
    \begin{array}{rcl}
      y &=& a x^{2} + b \\
      a &=& 3.0 \\
      b &=& 2-1.3i \\
    \end{array}
    """
    assert latex == dedent(expected).strip()

    latex = aslatex(definitions, terms_per_line=1)
    expected = R"""
    \begin{array}{rcl}
      y &=& a x^{2} \\
        &+& b \\
      a &=& 3.0 \\
      b &=& 2-1.3i \\
    \end{array}
    """
    assert latex == dedent(expected).strip()
