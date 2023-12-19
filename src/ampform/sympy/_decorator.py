from __future__ import annotations

import functools
import inspect
import sys
from collections import abc
from inspect import isclass
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, TypeVar, overload

import sympy as sp
from attrs import frozen
from sympy.core.basic import _aresame
from sympy.utilities.exceptions import SymPyDeprecationWarning

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
P = ParamSpec("P")
T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


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
        attr_values, hints = _get_attribute_values(cls, attr_names, *args, **kwargs)
        converted_attr_values = _safe_sympify(*attr_values)
        expr = sp.Expr.__new__(cls, *converted_attr_values.sympy, **hints)
        for name, value in zip(attr_names, converted_attr_values.all_args):
            setattr(expr, name, value)
        expr._all_args = converted_attr_values.all_args
        expr._non_sympy_args = converted_attr_values.non_sympy
        if evaluate:
            return expr.evaluate()
        return expr

    cls.__new__ = new_method  # type: ignore[method-assign]
    cls._eval_subs = _eval_subs_method  # type: ignore[method-assign]
    cls._hashable_content = _hashable_content_method  # type: ignore[method-assign]
    cls._xreplace = _xreplace_method  # type: ignore[method-assign]
    return cls


@overload
def _get_hashable_object(obj: type) -> str: ...  # type: ignore[overload-overlap]
@overload
def _get_hashable_object(obj: H) -> H: ...
@overload
def _get_hashable_object(obj: Any) -> str: ...
def _get_hashable_object(obj):
    if isclass(obj):
        return str(obj)
    try:
        hash(obj)
    except TypeError:
        return str(obj)
    return obj


def _get_attribute_values(
    cls: type[ExprClass], attr_names: tuple[str, ...], *args, **kwargs
) -> tuple[tuple, dict[str, Any]]:
    """Extract the attribute values from the constructor arguments.

    Returns a `tuple` of:

    1. the extracted, ordered attributes as requested by :code:`attr_names`,
    2. a `dict` of remaining keyword arguments that can be used hints for the
       constructed :class:`sp.Expr<sympy.core.expr.Expr>` instance.

    An attempt is made to get any missing attributes from the type hints in the class
    definition.
    """
    if len(args) == len(attr_names):
        return args, kwargs
    if len(args) > len(attr_names):
        msg = (
            f"Expecting {len(attr_names)} positional arguments"
            f" ({', '.join(attr_names)}), but got {len(args)}"
        )
        raise ValueError(msg)
    attr_values = list(args)
    remaining_attr_names = list(attr_names[len(args) :])
    for name in list(remaining_attr_names):
        if name in kwargs:
            attr_values.append(kwargs.pop(name))
            remaining_attr_names.pop(0)
        elif hasattr(cls, name):
            default_value = getattr(cls, name)
            attr_values.append(default_value)
            remaining_attr_names.pop(0)
    if remaining_attr_names:
        msg = f"Missing constructor arguments: {', '.join(remaining_attr_names)}"
        raise ValueError(msg)
    return tuple(attr_values), kwargs


def _safe_sympify(*args: Any) -> _ExprNewArumgents:
    all_args = []
    sympy_args = []
    non_sympy_args = []
    for arg in args:
        converted_arg, is_sympy = _try_sympify(arg)
        if is_sympy:
            sympy_args.append(converted_arg)
        else:
            non_sympy_args.append(converted_arg)
        all_args.append(converted_arg)
    return _ExprNewArumgents(
        all_args=tuple(all_args),
        sympy=tuple(sympy_args),
        non_sympy=tuple(non_sympy_args),
    )


def _try_sympify(obj) -> tuple[Any, bool]:
    if isinstance(obj, str):
        return obj, False
    try:
        return sp.sympify(obj), True
    except (TypeError, SymPyDeprecationWarning, sp.SympifyError):
        return obj, False


@frozen
class _ExprNewArumgents:
    all_args: tuple[Any, ...]
    sympy: tuple[sp.Basic, ...]
    non_sympy: tuple[Any, ...]


class LatexMethod(Protocol):
    def __call__(self, printer: LatexPrinter, *args) -> str: ...


@dataclass_transform()
def _implement_latex_repr(cls: type[T]) -> type[T]:
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
) -> Callable[[Callable[P, T]], Callable[P, T]]:
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


def _get_attribute_names(cls: type) -> tuple[str, ...]:
    """Get the public attributes of a class with dataclass-like semantics.

    >>> class MyClass:
    ...     a: int
    ...     b: int
    ...     _c: int
    ...     n: ClassVar[int] = 2
    ...
    ...     def print(self): ...
    ...
    >>> _get_attribute_names(MyClass)
    ('a', 'b')
    """
    return tuple(
        k
        for k, v in cls.__annotations__.items()
        if not callable(k)
        if not k.startswith("_")
        if not str(v).startswith("ClassVar")
    )


@dataclass_transform()
def _set_assumptions(
    **assumptions: Unpack[SymPyAssumptions],
) -> Callable[[type[T]], type[T]]:
    def class_wrapper(cls: T) -> T:
        for assumption, value in assumptions.items():
            setattr(cls, f"is_{assumption}", value)
        return cls

    return class_wrapper


def _eval_subs_method(self, old, new, **hints):
    # https://github.com/sympy/sympy/blob/1.12/sympy/core/basic.py#L1117-L1147
    hit = False
    substituted_attrs = list(self._all_args)
    for i, old_attr in enumerate(substituted_attrs):
        if not hasattr(old_attr, "_eval_subs"):
            continue
        if isclass(old_attr):
            continue
        new_attr = old_attr._subs(old, new, **hints)
        if not _aresame(new_attr, old_attr):
            hit = True
            substituted_attrs[i] = new_attr
    if hit:
        rv = self.func(*substituted_attrs)
        hack2 = hints.get("hack2", False)
        if hack2 and self.is_Mul and not rv.is_Mul:  # 2-arg hack
            coefficient = sp.S.One
            nonnumber = []
            for i in substituted_attrs:
                if i.is_Number:
                    coefficient *= i
                else:
                    nonnumber.append(i)
            nonnumber = self.func(*nonnumber)
            if coefficient is sp.S.One:
                return nonnumber
            return self.func(coefficient, nonnumber, evaluate=False)
        return rv
    return self


def _hashable_content_method(self) -> tuple:
    hashable_content = super(sp.Expr, self)._hashable_content()
    if not self._non_sympy_args:
        return hashable_content
    remaining_content = (_get_hashable_object(arg) for arg in self._non_sympy_args)
    return (*hashable_content, *remaining_content)


def _xreplace_method(self, rule) -> tuple[sp.Expr, bool]:
    # https://github.com/sympy/sympy/blob/1.12/sympy/core/basic.py#L1233-L1253
    if self in rule:
        return rule[self], True
    if rule:
        new_args = []
        hit = False
        for arg in self._all_args:
            if hasattr(arg, "_xreplace") and not isclass(arg):
                replace_result, is_replaced = arg._xreplace(rule)
            elif isinstance(rule, abc.Mapping):
                is_replaced = bool(arg in rule)
                replace_result = rule.get(arg, arg)
            else:
                replace_result = arg
                is_replaced = False
            new_args.append(replace_result)
            hit |= is_replaced
        if hit:
            return self.func(*new_args), True
    return self, False
