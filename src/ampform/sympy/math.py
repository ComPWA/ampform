"""A collection of basic math operations, used in `ampform.dynamics`."""  # noqa: A005

# cspell:ignore Lambdifier
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, overload

import sympy as sp
from sympy.plotting.experimental_lambdify import Lambdifier

from ampform.sympy import NumPyPrintable

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
if TYPE_CHECKING:
    from sympy.printing.numpy import NumPyPrinter
    from sympy.printing.printer import Printer
    from sympy.printing.pycode import PythonCodePrinter


class ComplexSqrt(NumPyPrintable):
    """Square root that returns positive imaginary values for negative input.

    A special version :func:`~sympy.functions.elementary.miscellaneous.sqrt` that
    renders nicely as LaTeX and and can be used as a handle for lambdify printers. See
    :doc:`compwa-report:000/index`, :doc:`compwa-report:001/index`, and
    :doc:`sympy:modules/printing` for how to implement a custom
    :func:`~sympy.utilities.lambdify.lambdify` printer.
    """

    is_commutative = True
    is_extended_real = True

    @overload
    def __new__(cls, x: sp.Number, *args, **kwargs) -> sp.Expr: ...  # type: ignore[misc]
    @overload
    def __new__(cls, x: sp.Expr, *args, **kwargs) -> ComplexSqrt: ...  # type:ignore[misc]
    @override
    def __new__(cls, x, *args, **kwargs):
        x = sp.sympify(x)
        args = sp.sympify((x, *args))
        expr: ComplexSqrt = sp.Expr.__new__(cls, *args, **kwargs)  # type: ignore[annotation-unchecked]
        if isinstance(x, sp.Number):
            return expr.get_definition()
        return expr

    def _latex(self, printer: Printer, *args) -> str:
        x = printer._print(self.args[0])
        return Rf"\sqrt[\mathrm{{c}}]{{{x}}}"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        return self.__print_complex(printer)

    def _pythoncode(self, printer: PythonCodePrinter, *args) -> str:
        # cspell:ignore csqrt
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
        """Get a symbolic definition for this expression class."""
        x: sp.Expr = self.args[0]  # type: ignore[assignment]
        return sp.Piecewise(
            (sp.I * sp.sqrt(-x), x < 0),
            (sp.sqrt(x), True),
        )


Lambdifier.builtin_functions_different["ComplexSqrt"] = "sqrt"
