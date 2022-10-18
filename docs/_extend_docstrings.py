# flake8: noqa
# pylint: disable=import-error,import-outside-toplevel,invalid-name,protected-access
# pyright: reportMissingImports=false
"""Extend docstrings of the API.

This small script is used by ``conf.py`` to dynamically modify docstrings.
"""

import inspect
import logging
import textwrap
from typing import Callable, Dict, Optional, Tuple, Type, Union

import attrs

# sphinx.ext.graphviz does not work well on RTD
import graphviz
import qrules
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ampform.io import aslatex
from ampform.kinematics import FourMomentumSymbol, _ArraySize
from ampform.sympy import NumPyPrintable
from ampform.sympy._array_expressions import ArrayMultiplication

logging.getLogger().setLevel(logging.ERROR)


def extend_docstrings() -> None:
    script_name = __file__.rsplit("/", maxsplit=1)[-1]
    script_name = ".".join(script_name.split(".")[:-1])
    definitions = dict(globals())
    for name, definition in definitions.items():
        module = inspect.getmodule(definition)
        if module is None:
            continue
        if module.__name__ not in {"__main__", script_name}:
            continue
        if not inspect.isfunction(definition):
            continue
        if not name.startswith("extend_"):
            continue
        if name == "extend_docstrings":
            continue
        function_arguments = inspect.signature(definition).parameters
        if len(function_arguments):
            raise ValueError(f"Local function {name} should not have a signature")
        definition()


def extend_BlattWeisskopfSquared() -> None:
    from ampform.dynamics import BlattWeisskopfSquared

    L = sp.Symbol("L", integer=True)
    z = sp.Symbol("z", real=True)
    expr = BlattWeisskopfSquared(L, z)
    _append_latex_doit_definition(expr, deep=True, full_width=True)


def extend_BoostMatrix() -> None:
    from ampform.kinematics import BoostMatrix

    p = FourMomentumSymbol("p", shape=[])
    expr = BoostMatrix(p)
    _append_to_docstring(
        BoostMatrix,
        f"""\n
    This boost operates on a `FourMomentumSymbol` and looks like:

    .. math:: {sp.latex(expr)} = {sp.latex(expr.as_explicit())}
        :class: full-width
        :label: BoostMatrix
    """,
    )
    _append_to_docstring(
        BoostMatrix,
        """
    In `TensorWaves <https://tensorwaves.rtfd.io>`_, this class is expressed in a
    computational backend and it should operate on four-momentum arrays of rank-2. As
    such, this boost matrix becomes a **rank-3** matrix. When using `NumPy
    <https://numpy.org>`_ as backend, the computation looks as follows:
    """,
    )
    _append_code_rendering(
        BoostMatrix(p).doit(),
        use_cse=True,
        docstring_class=BoostMatrix,
    )


def extend_BoostZMatrix() -> None:
    from ampform.kinematics import BoostZMatrix

    beta, n_events = sp.symbols("beta n")
    matrix = BoostZMatrix(beta, n_events)
    _append_to_docstring(
        BoostZMatrix,
        f"""\n
    This boost operates on a `FourMomentumSymbol` and looks like:

    .. math:: {sp.latex(matrix)} = {sp.latex(matrix.as_explicit())}
        :label: BoostZMatrix
    """,
    )
    _append_to_docstring(
        BoostZMatrix,
        """
    In `TensorWaves <https://tensorwaves.rtfd.io>`_, this class is expressed in a
    computational backend and it should operate on four-momentum arrays of rank-2. As
    such, this boost matrix becomes a **rank-3** matrix. When using `NumPy
    <https://numpy.org>`_ as backend, the computation looks as follows:
    """,
    )
    b = sp.Symbol("b")
    _append_code_rendering(
        BoostZMatrix(b).doit(),
        use_cse=True,
        docstring_class=BoostZMatrix,
    )

    from ampform.kinematics import RotationYMatrix, RotationZMatrix

    _append_to_docstring(
        BoostZMatrix,
        """
    Note that this code was generated with :func:`sympy.lambdify
    <sympy.utilities.lambdify.lambdify>` with :code:`cse=True`. The repetition of
    :func:`numpy.ones` is still bothersome, but these sub-nodes is also extracted by
    :func:`sympy.cse <sympy.simplify.cse_main.cse>` if the expression is nested further
    down in an :doc:`expression tree <sympy:tutorials/intro-tutorial/manipulation>`, for
    instance when boosting a `.FourMomentumSymbol` :math:`p` in the :math:`z`-direction:
    """,
    )
    p, beta, phi, theta = sp.symbols("p beta phi theta")
    multiplication = ArrayMultiplication(
        BoostZMatrix(beta, n_events=_ArraySize(p)),
        RotationYMatrix(theta, n_events=_ArraySize(p)),
        RotationZMatrix(phi, n_events=_ArraySize(p)),
        p,
    )
    _append_to_docstring(
        BoostZMatrix,
        f"""\n
    .. math:: {sp.latex(multiplication)}
        :label: boost-in-z-direction

    which in :mod:`numpy` code becomes:
    """,
    )
    _append_code_rendering(
        multiplication.doit(), use_cse=True, docstring_class=BoostZMatrix
    )


def extend_BreakupMomentumSquared() -> None:
    from ampform.dynamics.phasespace import BreakupMomentumSquared

    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = BreakupMomentumSquared(s, m_a, m_b)
    _append_latex_doit_definition(expr, deep=True)


def extend_ComplexSqrt() -> None:
    from ampform.sympy.math import ComplexSqrt

    x = sp.Symbol("x", real=True)
    expr = ComplexSqrt(x)
    _append_to_docstring(
        ComplexSqrt,
        Rf"""
    .. math:: {sp.latex(expr)} = {sp.latex(expr.get_definition())}
        :label: ComplexSqrt
    """,
    )


def extend_compute_third_mandelstam() -> None:
    from ampform.kinematics.phasespace import compute_third_mandelstam

    m0, m1, m2, m3 = sp.symbols("m:4")
    s1, s2 = sp.symbols("sigma1 sigma2")
    expr = compute_third_mandelstam(s1, s2, m0, m1, m2, m3)
    _append_to_docstring(
        compute_third_mandelstam,
        Rf"""

    .. math:: \sigma_3 = {sp.latex(expr)}
        :label: compute_third_mandelstam

    Note that this expression is symmetric in :math:`\sigma_{{1,2,3}}`.
    """,
    )


def extend_EqualMassPhaseSpaceFactor() -> None:
    from ampform.dynamics.phasespace import (
        EqualMassPhaseSpaceFactor,
        PhaseSpaceFactorAbs,
    )

    s, m_a, m_b = sp.symbols(R"s, m_a, m_b")
    expr = EqualMassPhaseSpaceFactor(s, m_a, m_b)
    _append_latex_doit_definition(expr)
    rho_hat = PhaseSpaceFactorAbs(s, m_a, m_b)
    _append_to_docstring(
        EqualMassPhaseSpaceFactor,
        f"""
    with :math:`{sp.latex(rho_hat)}` defined by `.PhaseSpaceFactorAbs`
    :eq:`PhaseSpaceFactorAbs`.
    """,
    )


def extend_EnergyDependentWidth() -> None:
    from ampform.dynamics import EnergyDependentWidth

    _append_to_docstring(
        EnergyDependentWidth,
        """
    With that in mind, the "mass-dependent" width in a
    `.relativistic_breit_wigner_with_ff` becomes:
    """,
    )
    L = sp.Symbol("L", integer=True)
    symbols: Tuple[sp.Symbol, ...] = sp.symbols("s m0 Gamma0 m_a m_b")
    s, m0, w0, m_a, m_b = symbols
    expr = EnergyDependentWidth(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=L,
        meson_radius=1,
    )
    _append_latex_doit_definition(expr)
    _append_to_docstring(
        EnergyDependentWidth,
        R"""
    where :math:`B_L^2` is defined by :eq:`BlattWeisskopfSquared`, :math:`q` is defined
    by :eq:`BreakupMomentumSquared`, and :math:`\rho` is (by default) defined by
    :eq:`PhaseSpaceFactor`.
    """,
    )


def extend_Energy_and_FourMomentumXYZ() -> None:
    from ampform.kinematics import Energy, FourMomentumX, FourMomentumY, FourMomentumZ

    def _extend(component_class: Type[sp.Expr]) -> None:
        _append_to_docstring(component_class, "\n\n")
        p = FourMomentumSymbol("p", shape=[])
        expr = component_class(p)
        _append_latex_doit_definition(expr, inline=True)

    _extend(Energy)
    _extend(FourMomentumX)
    _extend(FourMomentumY)
    _extend(FourMomentumZ)


def extend_EuclideanNorm() -> None:
    from ampform.kinematics import EuclideanNorm

    vector = FourMomentumSymbol("v", shape=[])
    expr = EuclideanNorm(vector)
    _append_to_docstring(type(expr), "\n\n" + 4 * " ")
    _append_latex_doit_definition(expr, deep=False, inline=True)
    _append_code_rendering(expr)


def extend_formulate_form_factor() -> None:
    from ampform.dynamics import formulate_form_factor

    s, m_a, m_b, L, d = sp.symbols("s m_a m_b L d")
    form_factor = formulate_form_factor(s, m_a, m_b, angular_momentum=L, meson_radius=d)
    _append_to_docstring(
        formulate_form_factor,
        f"""
    .. math:: {sp.latex(form_factor)}
        :label: formulate_form_factor
    """,
    )


def extend_Kallen() -> None:
    from ampform.kinematics.phasespace import Kallen

    x, y, z = sp.symbols("x:z")
    expr = Kallen(x, y, z)
    _append_latex_doit_definition(expr)
    _append_to_docstring(
        Kallen,
        """
    .. seealso:: `.BreakupMomentumSquared`
    """,
    )


def extend_Kibble() -> None:
    from ampform.kinematics.phasespace import Kibble

    m0, m1, m2, m3 = sp.symbols("m:4")
    s1, s2, s3 = sp.symbols("sigma1:4")
    expr = Kibble(s1, s2, s3, m0, m1, m2, m3)
    _append_latex_doit_definition(expr)
    _append_to_docstring(
        Kibble,
        R"""
    with :math:`\lambda` defined by :eq:`Kallen`.
    """,
    )


def extend_InvariantMass() -> None:
    from ampform.kinematics import InvariantMass

    p = FourMomentumSymbol("p", shape=[])
    expr = InvariantMass(p)
    _append_latex_doit_definition(expr)


def extend_is_within_phasespace() -> None:
    from ampform.kinematics.phasespace import is_within_phasespace

    m0, m1, m2, m3 = sp.symbols("m:4")
    s1, s2 = sp.symbols("sigma1 sigma2")
    expr = is_within_phasespace(s1, s2, m0, m1, m2, m3)
    _append_to_docstring(
        is_within_phasespace,
        Rf"""

    .. math:: {sp.latex(expr)}
        :label: is_within_phasespace

    with :math:`\phi` defined by :eq:`Kibble`.
    """,
    )


def extend_PhaseSpaceFactor() -> None:
    from ampform.dynamics.phasespace import PhaseSpaceFactor

    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = PhaseSpaceFactor(s, m_a, m_b)
    _append_latex_doit_definition(expr)
    _append_to_docstring(
        PhaseSpaceFactor,
        """
    with :math:`q^2` defined as :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_PhaseSpaceFactorAbs() -> None:
    from ampform.dynamics.phasespace import PhaseSpaceFactorAbs

    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = PhaseSpaceFactorAbs(s, m_a, m_b)
    _append_latex_doit_definition(expr)
    _append_to_docstring(
        PhaseSpaceFactorAbs,
        """
    with :math:`q^2(s)` defined as :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_PhaseSpaceFactorComplex() -> None:
    from ampform.dynamics.phasespace import PhaseSpaceFactorComplex

    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = PhaseSpaceFactorComplex(s, m_a, m_b)
    _append_latex_doit_definition(expr)
    _append_to_docstring(
        PhaseSpaceFactorComplex,
        """
    with :math:`q^2(s)` defined as :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_PhaseSpaceFactorSWave() -> None:
    from ampform.dynamics.phasespace import PhaseSpaceFactorSWave

    s, m_a, m_b = sp.symbols("s m_a m_b")
    expr = PhaseSpaceFactorSWave(s, m_a, m_b)
    _append_latex_doit_definition(expr, full_width=True)


def extend_Phi() -> None:
    from ampform.kinematics import Phi

    p = FourMomentumSymbol("p", shape=[])
    expr = Phi(p)
    _append_latex_doit_definition(expr)


def extend_RotationYMatrix() -> None:
    from ampform.kinematics import RotationYMatrix

    angle, n_events = sp.symbols("alpha n")
    expr = RotationYMatrix(angle, n_events)
    _append_to_docstring(
        RotationYMatrix,
        f"""\n
    The matrix for a rotation over angle :math:`\\alpha` around the :math:`y`-axis
    operating on `FourMomentumSymbol` looks like:

    .. math:: {sp.latex(expr)} = {sp.latex(expr.as_explicit())}
        :label: RotationYMatrix

    See `RotationZMatrix` for the computational code.
    """,
    )


def extend_RotationZMatrix() -> None:
    from ampform.kinematics import RotationZMatrix

    angle, n_events = sp.symbols("alpha n")
    expr = RotationZMatrix(angle, n_events)
    _append_to_docstring(
        RotationZMatrix,
        f"""\n
    The matrix for a rotation over angle :math:`\\alpha` around the :math:`z`-axis
    operating on `FourMomentumSymbol` looks like:

    .. math:: {sp.latex(expr)} = {sp.latex(expr.as_explicit())}
        :label: RotationZMatrix
    """,
    )
    _append_to_docstring(
        RotationZMatrix,
        """
    In `TensorWaves <https://tensorwaves.rtfd.io>`_, this class is expressed in a
    computational backend and it should operate on four-momentum arrays of rank-2. As
    such, this boost matrix becomes a **rank-3** matrix. When using `NumPy
    <https://numpy.org>`_ as backend, the computation looks as follows:
    """,
    )
    a = sp.Symbol("a")
    _append_code_rendering(
        RotationZMatrix(a).doit(),
        use_cse=True,
        docstring_class=RotationZMatrix,
    )
    _append_to_docstring(
        RotationZMatrix,
        """
    See also the note that comes with Equation :eq:`boost-in-z-direction`.
    """,
    )


def extend_Theta() -> None:
    from ampform.kinematics import Theta

    p = FourMomentumSymbol("p", shape=[])
    expr = Theta(p)
    _append_latex_doit_definition(expr)


def extend_ThreeMomentum() -> None:
    from ampform.kinematics import ThreeMomentum

    p = FourMomentumSymbol("p", shape=[])
    expr = ThreeMomentum(p)
    _append_to_docstring(type(expr), "\n\n" + 4 * " ")
    _append_latex_doit_definition(expr, deep=False, inline=True)
    _append_code_rendering(expr)


def extend_chew_mandelstam_s_wave() -> None:
    from ampform.dynamics.phasespace import chew_mandelstam_s_wave

    s, m_a, m_b = sp.symbols("s m_a m_b")
    expr = chew_mandelstam_s_wave(s, m_a, m_b)
    _append_to_docstring(
        chew_mandelstam_s_wave,
        Rf"""

    .. math:: {sp.latex(expr)}
        :class: full-width
        :label: chew_mandelstam_s_wave

    with :math:`q^2(s)` defined as :eq:`BreakupMomentumSquared`.

    .. seealso:: :doc:`compwa-org:report/003`
    """,
    )


def extend_formulate_clebsch_gordan_coefficients() -> None:
    from ampform.helicity import formulate_clebsch_gordan_coefficients

    _append_to_docstring(
        formulate_clebsch_gordan_coefficients,
        __get_graphviz_state_transition_example(
            formalism="canonical-helicity", transition_number=1
        ),
    )


def extend_formulate_wigner_d() -> None:
    from ampform.helicity import formulate_wigner_d

    _append_to_docstring(
        formulate_wigner_d,
        __get_graphviz_state_transition_example("helicity"),
    )


def __get_graphviz_state_transition_example(
    formalism: str, transition_number: int = 0
) -> str:
    reaction = qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=[("gamma", [-1]), "f(0)(980)"],
        formalism=formalism,
    )
    transition = reaction.transitions[transition_number]
    new_interaction = attrs.evolve(
        transition.interactions[0],
        parity_prefactor=None,
    )
    interactions = dict(transition.interactions)
    interactions[0] = new_interaction
    transition = attrs.evolve(transition, interactions=interactions)
    dot = qrules.io.asdot(
        transition,
        render_initial_state_id=True,
        render_node=True,
    )
    for state_id in [0, 1, -1]:
        dot = dot.replace(
            f'label="{state_id}: ',
            f'label="{state_id+2}: ',
        )
    return _graphviz_to_image(dot, indent=4, options={"align": "center"})


def extend_get_boost_chain_suffix() -> None:
    from ampform.helicity.naming import get_boost_chain_suffix

    topologies = qrules.topology.create_isobar_topologies(5)
    dot0, dot1, *_ = tuple(
        map(lambda t: qrules.io.asdot(t, render_resonance_id=True), topologies)
    )
    graphviz0 = _graphviz_to_image(
        dot0,
        indent=8,
        caption=":code:`topologies[0]`",
        label="one-to-five-topology-0",
    )
    graphviz1 = _graphviz_to_image(
        dot1,
        indent=8,
        caption=":code:`topologies[1]`",
        label="one-to-five-topology-1",
    )
    _append_to_docstring(
        get_boost_chain_suffix,
        f"""

    .. grid:: 1 2 2 2
      :gutter: 2

      .. grid-item-card::
        {graphviz0}

      .. grid-item-card::
        {graphviz1}
    """,
    )


def extend_relativistic_breit_wigner() -> None:
    from ampform.dynamics import relativistic_breit_wigner

    s, m0, w0 = sp.symbols("s m0 Gamma0")
    rel_bw = relativistic_breit_wigner(s, m0, w0)
    _append_to_docstring(
        relativistic_breit_wigner,
        f"""
    .. math:: {sp.latex(rel_bw)}
        :label: relativistic_breit_wigner
    """,
    )


def extend_relativistic_breit_wigner_with_ff() -> None:
    from ampform.dynamics import relativistic_breit_wigner_with_ff

    L = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d")
    rel_bw_with_ff = relativistic_breit_wigner_with_ff(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=L,
        meson_radius=d,
    )
    _append_to_docstring(
        relativistic_breit_wigner_with_ff,
        Rf"""
    The general form of a relativistic Breit-Wigner with Blatt-Weisskopf form factor is:

    .. math:: {sp.latex(rel_bw_with_ff)}
        :label: relativistic_breit_wigner_with_ff

    where :math:`\Gamma(s)` is defined by :eq:`EnergyDependentWidth`, :math:`B_L^2` is
    defined by :eq:`BlattWeisskopfSquared`, and :math:`q^2` is defined by
    :eq:`BreakupMomentumSquared`.
    """,
    )


def _append_code_rendering(
    expr: NumPyPrintable,
    use_cse: bool = False,
    docstring_class: Optional[type] = None,
) -> None:
    printer = NumPyPrinter()
    if use_cse:
        args = sorted(expr.free_symbols, key=str)
        func = sp.lambdify(args, expr, cse=True, printer=printer)
        numpy_code = inspect.getsource(func)
    else:
        numpy_code = expr._numpycode(printer)
    import_statements = __print_imports(printer)
    if docstring_class is None:
        docstring_class = type(expr)
    numpy_code = textwrap.dedent(numpy_code)
    numpy_code = textwrap.indent(numpy_code, prefix=8 * " ").strip()
    options = ""
    if max(__get_text_width(import_statements), __get_text_width(numpy_code)) > 90:
        options += ":class: full-width\n"
    appended_text = f"""\n
    .. code-block:: python
        {options}
        {import_statements}
        {numpy_code}
    """
    _append_to_docstring(docstring_class, appended_text)


def __get_text_width(text: str) -> int:
    lines = text.split("\n")
    widths = map(len, lines)
    return max(widths)


def _append_latex_doit_definition(
    expr: sp.Expr,
    deep: bool = False,
    full_width: bool = False,
    inline: bool = False,
) -> None:
    if inline:
        return _append_to_docstring(
            type(expr),
            f":math:`{sp.latex(expr)}={sp.latex(expr.doit(deep=deep))}`",
        )
    latex = _create_latex_doit_definition(expr, deep)
    extras = ""
    if full_width:
        extras = """
        :class: full-width
        """
    return _append_to_docstring(
        type(expr),
        f"""\n
    .. math::
        :label: {type(expr).__name__}{extras}

        {latex}
    """,
    )


def _create_latex_doit_definition(expr: sp.Expr, deep: bool = False) -> str:
    latex = aslatex({expr: expr.doit(deep=deep)})
    return textwrap.indent(latex, prefix=8 * " ")


def _append_to_docstring(class_type: Union[Callable, Type], appended_text: str) -> None:
    assert class_type.__doc__ is not None
    class_type.__doc__ += appended_text


def __print_imports(printer: NumPyPrinter) -> str:
    code = ""
    for module, items in printer.module_imports.items():
        imported_items = ", ".join(sorted(items))
        code += f"from {module} import {imported_items}\n"
    return code


_GRAPHVIZ_COUNTER = 0
_IMAGE_DIR = "_images"


def _graphviz_to_image(  # pylint: disable=too-many-arguments
    dot: str,
    options: Optional[Dict[str, str]] = None,
    format: str = "svg",
    indent: int = 0,
    caption: str = "",
    label: str = "",
) -> str:
    if options is None:
        options = {}
    global _GRAPHVIZ_COUNTER  # pylint: disable=global-statement
    output_file = f"graphviz_{_GRAPHVIZ_COUNTER}"
    _GRAPHVIZ_COUNTER += 1  # pyright: reportConstantRedefinition=false
    graphviz.Source(dot).render(f"{_IMAGE_DIR}/{output_file}", format=format)
    restructuredtext = "\n"
    if label:
        restructuredtext += f".. _{label}:\n"
    restructuredtext += f".. figure:: /{_IMAGE_DIR}/{output_file}.{format}\n"
    for option, value in options.items():
        restructuredtext += f"  :{option}: {value}\n"
    if caption:
        restructuredtext += f"\n  {caption}\n"
    return textwrap.indent(restructuredtext, indent * " ")
