from textwrap import dedent

import sympy as sp

from ampform.io import aslatex

a, b, x, y = sp.symbols("a b x y")


def test_complex():
    assert aslatex(1.2 - 5.3j) == "1.2-5.3i"
    assert aslatex(1.2 - 5j) == "1.2-5i"
    assert aslatex(1 + 1j) == "1+1i"


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


def test_mapping():
    definitions = {
        y: a * x**2 + b,
        a: 3.0,
        b: 2 - 1.3j,
    }
    latex = aslatex(definitions)
    expected = R"""
    \begin{array}{rcl}
      y &=& a x^{2} + b \\
      a &=& 3.0 \\
      b &=& 2-1.3i \\
    \end{array}
    """
    assert latex == dedent(expected).strip()
