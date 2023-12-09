from __future__ import annotations

import functools
import inspect
import sys
from typing import TYPE_CHECKING, Callable, Iterable, TypeVar, overload

import sympy as sp

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, TypedDict
else:
    from typing import Protocol, TypedDict

if sys.version_info < (3, 11):
    from typing_extensions import ParamSpec, Unpack, dataclass_transform
else:
    from typing import ParamSpec, Unpack, dataclass_transform

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter

ExprClass = TypeVar("ExprClass", bound=sp.Expr)
_P = ParamSpec("_P")
_T = TypeVar("_T")


class SymPyAssumptions(TypedDict, total=False):
    """See https://docs.sympy.org/latest/guides/assumptions.html#predicates."""

    algebraic: bool
    commutative: bool
    complex: bool
    extended_negative: bool
    extended_nonnegative: bool
    extended_nonpositive: bool
    extended_nonzero: bool
    extended_positive: bool
    extended_real: bool
    finite: bool
    hermitian: bool
    imaginary: bool
    infinite: bool
    integer: bool
    irrational: bool
    negative: bool
    noninteger: bool
    nonnegative: bool
    nonpositive: bool
    nonzero: bool
    positive: bool
    rational: bool
    real: bool
    transcendental: bool
    zero: bool


@overload
def unevaluated_expression(cls: type[ExprClass]) -> type[ExprClass]: ...
@overload
def unevaluated_expression(
    *,
    implement_doit: bool = True,
    **assumptions: Unpack[SymPyAssumptions],
) -> Callable[[type[ExprClass]], type[ExprClass]]: ...


@dataclass_transform()  # type: ignore[misc]
def unevaluated_expression(  # type: ignore[misc]
    cls: type[ExprClass] | None = None, *, implement_doit=True, **assumptions
):
    r"""Decorator for defining 'unevaluated' SymPy expressions.

    Unevaluated expressions are handy for defining large expressions that consist of
    several sub-definitions.

    >>> @unevaluated_expression
    ... class MyExpr(sp.Expr):
    ...     x: sp.Symbol
    ...     y: sp.Symbol
    ...     _latex_repr_ = R"z\left({x}, {y}\right)"
    ...
    ...     def evaluate(self) -> sp.Expr:
    ...         x, y = self.args
    ...         return x**2 + y**2
    ...
    >>> a, b = sp.symbols("a b")
    >>> expr = MyExpr(a, b**2)
    >>> sp.latex(expr)
    'z\\left(a, b^{2}\\right)'
    >>> expr.doit()
    a**2 + b**4
    """
    if assumptions is None:
        assumptions = {}
    if not assumptions.get("commutative"):
        assumptions["commutative"] = True

    def decorator(cls: type[ExprClass]) -> type[ExprClass]:
        cls = _implement_new_method(cls)
        if implement_doit:
            cls = _implement_doit(cls)
        if hasattr(cls, "_latex_repr_"):
            cls = _implement_latex_repr(cls)
        _set_assumptions(**assumptions)(cls)
        return cls

    if cls is None:
        return decorator
    return decorator(cls)


@dataclass_transform()
def _implement_new_method(cls: type[ExprClass]) -> type[ExprClass]:
    """Implement the :meth:`__new__` method for dataclass-like SymPy expression classes.

    >>> @_implement_new_method
    ... class MyExpr(sp.Expr):
    ...     a: sp.Symbol
    ...     b: sp.Symbol
    ...
    >>> x, y = sp.symbols("x y")
    >>> expr = MyExpr(x**2, y**2)
    >>> expr.a
    x**2
    >>> expr.args
    (x**2, y**2)
    >>> sp.sqrt(expr)
    sqrt(MyExpr(x**2, y**2))
    """
    attr_names = _get_attribute_names(cls)

    @functools.wraps(cls.__new__)
    @_insert_args_in_signature(attr_names, idx=1)
    def new_method(cls, *args, evaluate: bool = False, **kwargs) -> type[ExprClass]:
        attr_values, kwargs = _get_attribute_values(attr_names, *args, **kwargs)
        attr_values = sp.sympify(attr_values)
        expr = sp.Expr.__new__(cls, *attr_values, **kwargs)
        for name, value in zip(attr_names, args):
            setattr(expr, name, value)
        if evaluate:
            return expr.evaluate()
        return expr

    cls.__new__ = new_method  # type: ignore[method-assign]
    return cls


def _get_attribute_values(attr_names: list, *args, **kwargs) -> tuple[tuple, dict]:
    if len(args) == len(attr_names):
        return args, kwargs
    if len(args) > len(attr_names):
        msg = (
            f"Expecting {len(attr_names)} positional arguments"
            f" ({', '.join(attr_names)}), but got {len(args)}"
        )
        raise ValueError(msg)
    attr_values = list(args)
    remaining_attr_names = attr_names[len(args) :]
    for name in list(remaining_attr_names):
        if name in kwargs:
            attr_values.append(kwargs.pop(name))
            remaining_attr_names.pop(0)
    if remaining_attr_names:
        msg = f"Missing constructor arguments: {', '.join(remaining_attr_names)}"
        raise ValueError(msg)
    return tuple(attr_values), kwargs


class LatexMethod(Protocol):
    def __call__(self, printer: LatexPrinter, *args) -> str: ...


@dataclass_transform()
def _implement_latex_repr(cls: type[_T]) -> type[_T]:
    _latex_repr_: LatexMethod | str | None = getattr(cls, "_latex_repr_", None)
    if _latex_repr_ is None:
        msg = (
            "You need to define a _latex_repr_ str or method in order to decorate an"
            " unevaluated expression with a printer method for LaTeX representation."
        )
        raise NotImplementedError(msg)
    if callable(_latex_repr_):
        cls._latex = _latex_repr_  # type: ignore[attr-defined]
    else:
        attr_names = _get_attribute_names(cls)

        def latex_method(self, printer: LatexPrinter, *args) -> str:
            format_kwargs = {
                name: printer._print(getattr(self, name), *args) for name in attr_names
            }
            return _latex_repr_.format(**format_kwargs)  # type: ignore[union-attr]

        cls._latex = latex_method  # type: ignore[attr-defined]
    return cls


@dataclass_transform()
def _implement_doit(cls: type[ExprClass]) -> type[ExprClass]:
    _check_has_implementation(cls)

    @functools.wraps(cls.doit)
    def doit_method(self, deep: bool = True) -> sp.Expr:
        expr = self.evaluate()
        if deep:
            return expr.doit()
        return expr

    cls.doit = doit_method  # type: ignore[assignment]
    return cls


def _check_has_implementation(cls: type) -> None:
    implementation_method = getattr(cls, "evaluate", None)
    if implementation_method is None:
        msg = "Decorated class must have an evaluate() method"
        raise ValueError(msg)
    if not callable(implementation_method):
        msg = "evaluate() must be a callable method"
        raise TypeError(msg)


def _insert_args_in_signature(
    new_params: Iterable[str] | None = None, idx: int = 0
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    if new_params is None:
        new_params = []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        original_signature = inspect.signature(func)
        original_pars = list(original_signature.parameters.values())
        new_parameters = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in new_params
        ]
        new_parameters = [*original_pars[:idx], *new_parameters, *original_pars[idx:]]
        wrapper.__signature__ = inspect.Signature(
            parameters=new_parameters,
            return_annotation=original_signature.return_annotation,
        )
        return wrapper

    return decorator


def _get_attribute_names(cls: type) -> list[str]:
    """Get the public attributes of a class with dataclass-like semantics.

    >>> class MyClass:
    ...     a: int
    ...     b: int
    ...     _c: int
    ...
    ...     def print(self): ...
    ...
    >>> _get_attribute_names(MyClass)
    ['a', 'b']
    """
    return [v for v in cls.__annotations__ if not callable(v) if not v.startswith("_")]


@dataclass_transform()
def _set_assumptions(
    **assumptions: Unpack[SymPyAssumptions],
) -> Callable[[type[_T]], type[_T]]:
    def class_wrapper(cls: _T) -> _T:
        for assumption, value in assumptions.items():
            setattr(cls, f"is_{assumption}", value)
        return cls

    return class_wrapper
