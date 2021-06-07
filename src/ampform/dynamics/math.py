"""A collection of basic mathematical operations, used in `ampform.dynamics`."""
# cspell:ignore cmath compwa csqrt lambdifier
# pylint: disable=no-member, protected-access, unused-argument

from typing import Any

import sympy as sp
from sympy.plotting.experimental_lambdify import Lambdifier
from sympy.printing.printer import Printer


class ComplexSqrt(sp.Expr):
    """Square root that returns positive imaginary values for negative input.

    A special version :func:`~sympy.functions.elementary.miscellaneous.sqrt`
    that renders nicely as LaTeX and and can be used as a handle for lambdify
    printers. See :doc:`compwa-org:report/000`, :doc:`compwa-org:report/001`,
    and :doc:`sympy:modules/printing` for how to implement a custom
    :func:`~sympy.utilities.lambdify.lambdify` printer.
    """

    is_commutative = True

    def __new__(cls, x: sp.Expr, *args: Any, **kwargs: Any) -> sp.Expr:
        x = sp.sympify(x)
        expr = sp.Expr.__new__(cls, x, *args, **kwargs)
        if hasattr(x, "free_symbols") and not x.free_symbols:
            return expr.evaluate()
        return expr

    def evaluate(self) -> sp.Expr:
        x = self.args[0]
        if not x.is_real:
            return sp.sqrt(x)
        return self._evaluate_complex(x)

    @staticmethod
    def _evaluate_complex(x: sp.Expr) -> sp.Expr:
        return sp.Piecewise(
            (sp.I * sp.sqrt(-x), x < 0),
            (sp.sqrt(x), True),
        )

    def _latex(self, printer: Printer, *args: Any) -> str:
        x = printer._print(self.args[0])
        return fR"\sqrt[\mathrm{{c}}]{{{x}}}"

    def _numpycode(self, printer: Printer, *args: Any) -> str:
        return self.__print_complex(printer)

    def _pythoncode(self, printer: Printer, *args: Any) -> str:
        printer.module_imports["cmath"].add("sqrt as csqrt")
        x = printer._print(self.args[0])
        return (
            f"(((1j*sqrt(-{x}))"
            f" if isinstance({x}, (float, int)) and ({x} < 0)"
            f" else (csqrt({x}))))"
        )

    def __print_complex(self, printer: Printer) -> str:
        x = self.args[0]
        expr = self._evaluate_complex(x)
        return printer._print(expr)


Lambdifier.builtin_functions_different["ComplexSqrt"] = "sqrt"
