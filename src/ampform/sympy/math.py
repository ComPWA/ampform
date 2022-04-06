# cspell:ignore cmath compwa csqrt lambdifier
# pylint: disable=no-member, protected-access, unused-argument, W0223
# https://stackoverflow.com/a/22224042
"""A collection of basic math operations, used in `ampform.dynamics`."""
from __future__ import annotations

from typing import overload

import sympy as sp
from sympy.plotting.experimental_lambdify import Lambdifier
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.printer import Printer
from sympy.printing.pycode import PythonCodePrinter

from . import NumPyPrintable, create_expression, make_commutative


@make_commutative
class ComplexSqrt(NumPyPrintable):
    """Square root that returns positive imaginary values for negative input.

    A special version :func:`~sympy.functions.elementary.miscellaneous.sqrt`
    that renders nicely as LaTeX and and can be used as a handle for lambdify
    printers. See :doc:`compwa-org:report/000`, :doc:`compwa-org:report/001`,
    and :doc:`sympy:modules/printing` for how to implement a custom
    :func:`~sympy.utilities.lambdify.lambdify` printer.
    """

    @overload
    def __new__(cls, x: sp.Number, *args, **kwargs) -> sp.Expr:  # type: ignore[misc]
        ...

    @overload
    def __new__(cls, x: sp.Expr, *args, **kwargs) -> ComplexSqrt:
        ...

    def __new__(cls, x, *args, **kwargs):
        x = sp.sympify(x)
        expr = create_expression(cls, x, *args, **kwargs)
        if isinstance(x, sp.Number):
            return expr.get_definition()
        return expr

    def _latex(self, printer: Printer, *args) -> str:
        x = printer._print(self.args[0])
        return Rf"\sqrt[\mathrm{{c}}]{{{x}}}"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        return self.__print_complex(printer)

    def _pythoncode(self, printer: PythonCodePrinter, *args) -> str:
        printer.module_imports["cmath"].add("sqrt as csqrt")
        x = printer._print(self.args[0])
        return (
            f"(((1j*sqrt(-{x}))"
            f" if isinstance({x}, (float, int)) and ({x} < 0)"
            f" else (csqrt({x}))))"
        )

    def __print_complex(self, printer: Printer) -> str:
        expr = self.get_definition()
        return printer._print(expr)

    def get_definition(self) -> sp.Piecewise:
        """Get a symbolic definition for this expression class.

        .. note:: This class is `.NumPyPrintable`, so should not have an
            :meth:`~.UnevaluatedExpression.evaluate` method (in order to block
            :meth:`~sympy.core.basic.Basic.doit`). This method serves as an
            equivalent to that.
        """
        x: sp.Expr = self.args[0]  # type: ignore[assignment]
        return sp.Piecewise(
            (sp.I * sp.sqrt(-x), x < 0),
            (sp.sqrt(x), True),
        )


Lambdifier.builtin_functions_different["ComplexSqrt"] = "sqrt"
