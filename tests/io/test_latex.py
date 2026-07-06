from textwrap import dedent

import pytest
import sympy as sp
from attrs import asdict

from ampform.decay import IsobarNode, Particle, State
from ampform.io import as_markdown_table, aslatex

a, b, x, y = sp.symbols("a b x y")

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = {"mass": 0, "width": 0}
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1, **dummy_args)
p = Particle("p", latex="p", spin=0.5, parity=+1, **dummy_args)
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1, **dummy_args)
K = Particle("K-", latex="K^-", spin=0, parity=-1, **dummy_args)
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, **dummy_args)


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
    \begin{aligned}
    & x \\
    & \;+\; y \\
    & \;+\; z \\
    \end{aligned}
    """)
    assert aslatex(expr, terms_per_line=1) == expected.strip()

    expected = dedent(R"""
    \begin{aligned}
    & x + y \\
    & \;+\; z \\
    \end{aligned}
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
    \begin{aligned}
      y \;&=\; a x^{2} + b \\
      a \;&=\; 3.0 \\
      b \;&=\; 2-1.3i \\
    \end{aligned}
    """
    assert latex == dedent(expected).strip()

    latex = aslatex(definitions, terms_per_line=1)
    expected = R"""
    \begin{aligned}
      y \;&=\; a x^{2} \\
        \;&+\; b \\
      a \;&=\; 3.0 \\
      b \;&=\; 2-1.3i \\
    \end{aligned}
    """
    assert latex == dedent(expected).strip()


def test_aslatex_particle():
    latex = aslatex(Λ1520)
    assert latex == Λ1520.latex
    latex = aslatex(Λ1520, only_jp=True)
    assert latex == R"\frac{3}{2}^-"
    latex = aslatex(Λ1520, with_jp=True)
    assert latex == Λ1520.latex + R"\left[\frac{3}{2}^-\right]"


def test_aslatex_isobar_node():
    node = IsobarNode(Λ1520, p, K)
    latex = aslatex(node)
    assert latex == R"\left(\Lambda(1520) \to p K^-\right)"
    latex = aslatex(node, with_jp=True)
    expected = R"""
    \left(\Lambda(1520)\left[\frac{3}{2}^-\right] \to p\left[\frac{1}{2}^+\right] K^-\left[0^-\right]\right)
    """.strip()
    assert latex == expected

    node = IsobarNode(Λ1520, p, K, interaction=(2, 1))
    latex = aslatex(node)
    assert latex == R"\left(\Lambda(1520) \xrightarrow[S=1]{L=2} p K^-\right)"


def test_as_markdown_table_particles():
    p_state = State(**asdict(p), index=1)
    k_state = State(**asdict(K), index=2)
    particles = [p_state, k_state, π]
    src = as_markdown_table(particles)
    expected = dedent(R"""
    | index | name | LaTeX | $J^P$ | mass (MeV) | width (MeV) |
    | --- | --- | --- | --- | --- | --- |
    | 1 | `p` | $p$ | $\frac{1}{2}^+$ | 0 | 0 |
    | 2 | `K-` | $K^-$ | $0^-$ | 0 | 0 |
    |   | `π+` | $\pi^+$ | $0^-$ | 0 | 0 |
    """)
    assert src.strip() == expected.strip()
