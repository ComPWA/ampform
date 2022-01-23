# flake8: noqa
# pylint: disable=import-error,invalid-name,protected-access
# pyright: reportMissingImports=false
"""Extend docstrings of the API.

This small script is used by ``conf.py`` to dynamically modify docstrings.
"""

import inspect
import logging
import textwrap
from typing import Callable, Dict, Optional, Type, Union

import attr

# sphinx.ext.graphviz does not work well on RTD
import graphviz
import qrules
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ampform.dynamics import (
    BlattWeisskopfSquared,
    BreakupMomentumSquared,
    EnergyDependentWidth,
    PhaseSpaceFactor,
    PhaseSpaceFactorAbs,
    PhaseSpaceFactorAnalytic,
    PhaseSpaceFactorComplex,
    relativistic_breit_wigner,
    relativistic_breit_wigner_with_ff,
)
from ampform.helicity import (
    formulate_clebsch_gordan_coefficients,
    formulate_wigner_d,
)
from ampform.kinematics import (
    BoostZ,
    Energy,
    FourMomentumX,
    FourMomentumY,
    FourMomentumZ,
    InvariantMass,
    Phi,
    RotationY,
    RotationZ,
    Theta,
    ThreeMomentumNorm,
    get_helicity_angle_label,
)
from ampform.sympy._array_expressions import ArraySymbol
from ampform.sympy.math import ComplexSqrt

logging.getLogger().setLevel(logging.ERROR)


def extend_blatt_weisskopf() -> None:
    L = sp.Symbol("L", integer=True)
    z = sp.Symbol("z", real=True)
    expr = BlattWeisskopfSquared(L, z)
    latex = _create_latex_doit_definition(expr, deep=True)
    _update_docstring(
        BlattWeisskopfSquared,
        f"""
    .. math::
        :label: BlattWeisskopfSquared
        :class: full-width

        {latex}
    """,
    )


def extend_boost_z() -> None:
    beta = sp.Symbol("beta")
    expr = BoostZ(beta)
    _update_docstring(
        BoostZ,
        f"""\n
    This boost operates on a `FourMomentumSymbol` and looks like:

    .. math:: {sp.latex(expr)} = {sp.latex(expr.as_explicit())}
        :label: BoostZ
    """,
    )
    b = sp.Symbol("b")
    printer = NumPyPrinter()
    numpy_code = BoostZ(b)._numpycode(printer)
    import_statements = __print_imports(printer)
    _update_docstring(
        BoostZ,
        f"""
    In `TensorWaves <https://tensorwaves.rtfd.io>`_, this class is expressed in
    a computational backend and it should operate on four-momentum arrays of
    rank-2. As such, this boost matrix becomes a **rank-3** matrix. When using
    `NumPy <https://numpy.org>`_ as backend, the computation looks as follows:

    .. code::

        {import_statements}
        {numpy_code}
    """,
    )


def extend_breakup_momentum_squared() -> None:
    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = BreakupMomentumSquared(s, m_a, m_b)
    latex = _create_latex_doit_definition(expr, deep=True)
    _update_docstring(
        BreakupMomentumSquared,
        f"""
    .. math::
        :label: BreakupMomentumSquared

        {latex}
    """,
    )


def extend_complex_sqrt() -> None:
    x = sp.Symbol("x", real=True)
    expr = ComplexSqrt(x)
    _update_docstring(
        ComplexSqrt,
        fR"""
    .. math:: {sp.latex(expr)} = {sp.latex(expr.evaluate())}
        :label: ComplexSqrt
    """,
    )


def extend_energy_dependent_width() -> None:
    L = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b = sp.symbols("s m0 Gamma0 m_a m_b")
    expr = EnergyDependentWidth(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=L,
        meson_radius=1,
    )
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        EnergyDependentWidth,
        fR"""
    With that in mind, the "mass-dependent" width in a
    `.relativistic_breit_wigner_with_ff` becomes:

    .. math::
        :label: EnergyDependentWidth

        {latex}

    where :math:`B_L^2` is defined by :eq:`BlattWeisskopfSquared`, :math:`q` is
    defined by :eq:`BreakupMomentumSquared`, and :math:`\rho` is (by default)
    defined by :eq:`PhaseSpaceFactor`.
    """,
    )


def extend_formulate_wigner_d() -> None:
    _update_docstring(
        formulate_wigner_d,
        __get_graphviz_state_transition_example("helicity"),
    )


def extend_formulate_clebsch_gordan_coefficients() -> None:
    _update_docstring(
        formulate_clebsch_gordan_coefficients,
        __get_graphviz_state_transition_example(
            formalism="canonical-helicity", transition_number=1
        ),
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
    new_interaction = attr.evolve(
        transition.interactions[0],
        parity_prefactor=None,
    )
    interactions = dict(transition.interactions)
    interactions[0] = new_interaction
    transition = attr.evolve(transition, interactions=interactions)
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


def extend_four_momentum_components() -> None:
    def _extend(component_class: Type[sp.Expr]) -> None:
        p = ArraySymbol("p")
        energy = component_class(p)
        _update_docstring(
            component_class,
            f"""\n
            :math:`{sp.latex(energy)}={sp.latex(energy.doit())}`
            """,
        )

    _extend(Energy)
    _extend(FourMomentumX)
    _extend(FourMomentumY)
    _extend(FourMomentumZ)


def extend_get_helicity_angle_label() -> None:
    topologies = qrules.topology.create_isobar_topologies(5)
    dot0, dot1, *_ = tuple(
        map(lambda t: qrules.io.asdot(t, render_resonance_id=True), topologies)
    )
    graphviz0 = _graphviz_to_image(
        dot0,
        indent=6,
        caption=":code:`topologies[0]`",
        label="one-to-five-topology-0",
    )
    graphviz1 = _graphviz_to_image(
        dot1,
        indent=6,
        caption=":code:`topologies[1]`",
        label="one-to-five-topology-1",
    )
    _update_docstring(
        get_helicity_angle_label,
        f"""

    .. panels::
      :body: text-center
      {graphviz0}

      ---
      {graphviz1}
    """,
    )


def extend_invariant_mass() -> None:
    p = ArraySymbol("p")
    expr = InvariantMass(p)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        InvariantMass,
        f"""\n
    .. math::
        :label: InvariantMass

        {latex}
    """,
    )


def extend_phase_space_factor() -> None:
    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = PhaseSpaceFactor(s, m_a, m_b)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        PhaseSpaceFactor,
        f"""

    .. math::
        :label: PhaseSpaceFactor

        {latex}

    with :math:`q^2` defined as :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_phase_space_factor_abs() -> None:
    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = PhaseSpaceFactorAbs(s, m_a, m_b)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        PhaseSpaceFactorAbs,
        fR"""

    .. math::
        :label: PhaseSpaceFactorAbs

        {latex}

    with :math:`q^2(s)` defined as :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_phase_space_factor_analytic() -> None:
    s, m_a, m_b = sp.symbols(R"s, m_a, m_b")
    expr = PhaseSpaceFactorAnalytic(s, m_a, m_b)
    latex = _create_latex_doit_definition(expr)
    rho_hat = PhaseSpaceFactorAbs(s, m_a, m_b)
    _update_docstring(
        PhaseSpaceFactorAnalytic,
        fR"""
    .. math::
        :label: PhaseSpaceFactorAnalytic

        {latex}

    with :math:`{sp.latex(rho_hat)}` defined by `.PhaseSpaceFactorAbs`
    :eq:`PhaseSpaceFactorAbs`.
    """,
    )


def extend_phase_space_factor_complex() -> None:
    s, m_a, m_b = sp.symbols("s, m_a, m_b")
    expr = PhaseSpaceFactorComplex(s, m_a, m_b)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        PhaseSpaceFactorComplex,
        fR"""

    .. math::
        :label: PhaseSpaceFactorComplex

        {latex}

    with :math:`q^2(s)` defined as :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_phi() -> None:
    p = ArraySymbol("p")
    expr = Phi(p)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        Phi,
        f"""\n
    .. math::
        :label: Phi

        {latex}
    """,
    )


def extend_relativistic_breit_wigner() -> None:
    s, m0, w0 = sp.symbols("s m0 Gamma0")
    rel_bw = relativistic_breit_wigner(s, m0, w0)
    _update_docstring(
        relativistic_breit_wigner,
        f"""
    .. math:: {sp.latex(rel_bw)}
        :label: relativistic_breit_wigner
    """,
    )


def extend_relativistic_breit_wigner_with_ff() -> None:
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
    _update_docstring(
        relativistic_breit_wigner_with_ff,
        fR"""
    The general form of a relativistic Breit-Wigner with Blatt-Weisskopf form
    factor is:

    .. math:: {sp.latex(rel_bw_with_ff)}
        :label: relativistic_breit_wigner_with_ff

    where :math:`\Gamma(s)` is defined by :eq:`EnergyDependentWidth`, :math:`B_L^2` is
    defined by :eq:`BlattWeisskopfSquared`, and :math:`q^2` is defined by
    :eq:`BreakupMomentumSquared`.
    """,
    )


def extend_rotation_y() -> None:
    angle = sp.Symbol("alpha")
    expr = RotationY(angle)
    _update_docstring(
        RotationY,
        f"""\n
    The **matrix** for a rotation over angle :math:`\\alpha` around the
    :math:`y`-axis operating on `FourMomentumSymbol` looks like:

    .. math:: {sp.latex(expr)} = {sp.latex(expr.as_explicit())}
        :label: RotationY

    See `RotationZ` for the computational code.
    """,
    )


def extend_rotation_z() -> None:
    angle = sp.Symbol("alpha")
    expr = RotationZ(angle)
    _update_docstring(
        RotationZ,
        f"""\n
    The **matrix** for a rotation over angle :math:`\\alpha` around the
    :math:`y`-axis operating on `FourMomentumSymbol` looks like:

    .. math:: {sp.latex(expr)} = {sp.latex(expr.as_explicit())}
        :label: RotationZ
    """,
    )
    a = sp.Symbol("a")
    printer = NumPyPrinter()
    numpy_code = RotationZ(a)._numpycode(printer)
    import_statements = __print_imports(printer)
    _update_docstring(
        RotationZ,
        f"""
    In `TensorWaves <https://tensorwaves.rtfd.io>`_, this class is expressed in
    a computational backend and it should operate on four-momentum arrays of
    rank-2. As such, this boost matrix becomes a **rank-3** matrix. When using
    `NumPy <https://numpy.org>`_ as backend, the computation looks as follows:

    .. code-block::

        {import_statements}
        {numpy_code}
    """,
    )


def extend_theta() -> None:
    p = ArraySymbol("p")
    expr = Theta(p)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        Theta,
        f"""\n
    .. math::
        :label: Theta

        {latex}
    """,
    )


def extend_three_momentum_norm() -> None:
    p = ArraySymbol("p")
    expr = ThreeMomentumNorm(p)
    latex = _create_latex_doit_definition(expr)
    _update_docstring(
        ThreeMomentumNorm,
        f"""\n
    .. math::
        :label: ThreeMomentumNorm

        {latex}
    """,
    )


def _update_docstring(
    class_type: Union[Callable, Type], appended_text: str
) -> None:
    assert class_type.__doc__ is not None
    class_type.__doc__ += appended_text


def _create_latex_doit_definition(expr: sp.Expr, deep: bool = False) -> str:
    latex = sp.multiline_latex(
        expr, expr.doit(deep=deep), environment="eqnarray"
    )
    return textwrap.indent(latex, prefix=8 * " ")


def __print_imports(printer: NumPyPrinter) -> str:
    code = ""
    for module, items in printer.module_imports.items():
        imported_items = ", ".join(sorted(items))
        code += f"from {module} import {imported_items}\n"
    return code


SCRIPT_NAME = __file__.rsplit("/", maxsplit=1)[-1]
SCRIPT_NAME = ".".join(SCRIPT_NAME.split(".")[:-1])


def insert_math() -> None:
    definitions = dict(globals())
    for name, definition in definitions.items():
        module = inspect.getmodule(definition)
        if module is None:
            continue
        if module.__name__ not in {"__main__", SCRIPT_NAME}:
            continue
        if not inspect.isfunction(definition):
            continue
        if not name.startswith("extend_"):
            continue
        function_arguments = inspect.signature(definition).parameters
        if len(function_arguments):
            raise ValueError(
                f"Local function {name} should not have a signature"
            )
        definition()


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
    _GRAPHVIZ_COUNTER += 1
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
