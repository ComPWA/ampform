from __future__ import annotations

import dataclasses
import functools
import inspect
import sys
import warnings
from collections import abc
from dataclasses import MISSING, Field
from dataclasses import astuple as _get_arguments
from dataclasses import dataclass as _create_dataclass
from dataclasses import field as _create_field
from dataclasses import fields as _get_fields
from inspect import isclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypedDict, TypeVar, overload

import sympy as sp
from sympy.core.basic import _aresame  # noqa: PLC2701
from sympy.utilities.exceptions import SymPyDeprecationWarning

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from sympy.printing.latex import LatexPrinter

    if sys.version_info >= (3, 11):
        from typing import ParamSpec, Unpack
    else:
        from typing_extensions import ParamSpec, Unpack

    H = TypeVar("H", bound=Hashable)
    P = ParamSpec("P")
    T = TypeVar("T")

ExprClass = TypeVar("ExprClass", bound=sp.Expr)


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
def argument(*, default: T = MISSING, sympify: bool = True) -> T: ...  # type: ignore[assignment]
@overload
def argument(
    *,
    default_factory: Callable[[], T] = MISSING,  # type: ignore[assignment]
    sympify: bool = True,
) -> T: ...
def argument(
    *,
    default=MISSING,
    default_factory=MISSING,
    sympify=True,
):
    """Add qualifiers to fields of `unevaluated` SymPy expression classes.

    Creates a :class:`dataclasses.Field` with additional metadata for
    :func:`unevaluated` by wrapping around :func:`dataclasses.field`.

    .. versionadded:: 0.14.8
    """
    return _create_field(
        default=default,
        default_factory=default_factory,
        metadata={"sympify": sympify},
    )


@overload
def unevaluated(cls: type[ExprClass]) -> type[ExprClass]: ...
@overload
def unevaluated(
    *,
    implement_doit: bool = True,
    **assumptions: Unpack[SymPyAssumptions],
) -> Callable[[type[ExprClass]], type[ExprClass]]: ...
@dataclass_transform(field_specifiers=(argument, _create_field))
def unevaluated(
    cls: type[ExprClass] | None = None, *, implement_doit=True, **assumptions
):
    r"""Decorator for defining 'unevaluated' SymPy expressions.

    Unevaluated expressions are handy for defining large expressions that consist of
    several sub-definitions. They are 'unfolded' to their definition once you call their
    :meth`~sympy.core.expr.Expr.doit` method. For example:

    >>> @unevaluated
    ... class MyExpr(sp.Expr):
    ...     x: sp.Symbol
    ...     y: sp.Symbol
    ...     _latex_repr_ = R"z\left({x}, {y}\right)"
    ...
    ...     def evaluate(self) -> sp.Expr:
    ...         x, y = self.args
    ...         return x**2 + y**2
    >>> a, b = sp.symbols("a b")
    >>> expr = MyExpr(a, b**2)
    >>> sp.latex(expr)
    'z\\left(a, b^{2}\\right)'
    >>> expr.doit()
    a**2 + b**4

    A LaTeX representation for the unevaluated state can be provided by providing an
    `f-string <https://docs.python.org/3/reference/lexical_analysis.html#f-strings>`_ or
    method called :code:`_latex_repr_`:

    >>> @unevaluated
    ... class Function(sp.Expr):
    ...     x: sp.Symbol
    ...     _latex_repr_ = R"f\left({x}\right)"  # not an f-string!
    ...
    ...     def evaluate(self) -> sp.Expr:
    ...         return sp.sqrt(self.x)
    >>> y = sp.Symbol("y", nonnegative=True)
    >>> expr = Function(x=y**2)
    >>> sp.latex(expr)
    'f\\left(y^{2}\\right)'
    >>> expr.doit()
    y

    Or, `as a method <https://docs.sympy.org/latest/modules/printing.html#example-of-custom-printing-method>`_:

    >>> from sympy.printing.latex import LatexPrinter
    >>> @unevaluated
    ... class Function(sp.Expr):
    ...     x: sp.Symbol
    ...
    ...     def evaluate(self) -> sp.Expr:
    ...         return self.x**2
    ...
    ...     def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
    ...         x = printer._print(self.x)  # important to convert to string first
    ...         x, *_ = map(printer._print, self.args)  # also possible via its args
    ...         return Rf"g\left({x}\right)"  # this is an f-string
    >>> expr = Function(y)
    >>> sp.latex(expr)
    'g\\left(y\\right)'

    Attributes to the class are fed to the `~object.__new__` constructor of the
    :class:`~sympy.core.expr.Expr` class and are therefore also called "arguments". Just
    like in the :class:`~sympy.core.expr.Expr` class, these arguments are automatically
    `sympified
    <https://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify>`_.
    Attributes/arguments that should not be sympified with :func:`argument`:

    >>> class Transformation:
    ...     def __call__(self, x: sp.Basic, y: sp.Basic) -> sp.Expr: ...
    >>> @unevaluated
    ... class MyExpr(sp.Expr):
    ...     x: Any
    ...     y: Any
    ...     functor: Callable = argument(sympify=False)
    ...
    ...     def evaluate(self) -> sp.Expr:
    ...         return self.functor(self.x, self.y)
    >>> expr = MyExpr(0, y=3.14, functor=Transformation)
    >>> isinstance(expr.x, sp.Integer)
    True
    >>> isinstance(expr.y, sp.Float)
    True
    >>> expr.functor is Transformation
    True

    .. versionadded:: 0.14.8
    .. versionchanged:: 0.14.7
        Renamed from :code:`@unevaluated_expression()` to :code:`@unevaluated()`.`
    """
    if assumptions is None:
        assumptions = {}
    if not assumptions.get("commutative"):
        assumptions["commutative"] = True

    def decorator(cls: type[ExprClass]) -> type[ExprClass]:
        cls = _implement_new_method(cls)
        if implement_doit:
            cls = _implement_doit(cls)
        typos = ["_latex_repr"]
        for typo in typos:
            if hasattr(cls, typo):
                msg = f"Class defines a {typo} attribute, but it should be _latex_repr_"
                warnings.warn(msg, category=UserWarning, stacklevel=1)
        if hasattr(cls, "_latex_repr_"):
            cls = _implement_latex_repr(cls)
        _set_assumptions(**assumptions)(cls)
        return cls

    if cls is None:
        return decorator
    return decorator(cls)


@dataclass_transform(field_specifiers=(argument, _create_field))
def _implement_new_method(cls: type[ExprClass]) -> type[ExprClass]:
    """Implement :meth:`~object.__new__` for dataclass-like SymPy expression classes.

    >>> @_implement_new_method
    ... class MyExpr(sp.Expr):
    ...     a: sp.Symbol
    ...     b: sp.Symbol
    >>> x, y = sp.symbols("x y")
    >>> expr = MyExpr(x**2, y**2)
    >>> expr.a
    x**2
    >>> expr.args
    (x**2, y**2)
    >>> sp.sqrt(expr)
    sqrt(MyExpr(x**2, y**2))
    """
    cls = _create_dataclass(
        init=False,  # __new__ method through sp.Expr
        repr=False,
        eq=False,
        order=False,
        unsafe_hash=False,
        frozen=False,
    )(cls)
    cls = _update_field_metadata(cls)
    non_sympy_fields = tuple(f for f in _get_fields(cls) if not _is_sympify(f))  # type: ignore[arg-type]
    cls.__slots__ = tuple(f.name for f in non_sympy_fields)  # type: ignore[arg-type]

    @functools.wraps(cls.__new__)
    @_insert_args_in_signature([f.name for f in _get_fields(cls)], idx=1)  # type:ignore[arg-type]
    def new_method(cls, *args, evaluate: bool = False, **kwargs) -> type[ExprClass]:
        fields_with_values, hints = _extract_field_values(cls, *args, **kwargs)
        fields_with_sympified_values = {
            field: _safe_sympify(field, value)
            for field, value in fields_with_values.items()
        }
        sympy_args = tuple(
            value
            for field, value in fields_with_sympified_values.items()
            if _is_sympify(field)
        )
        expr = sp.Expr.__new__(cls, *sympy_args, **hints)
        for field, value in fields_with_sympified_values.items():
            setattr(expr, field.name, value)
        if evaluate:
            return expr.evaluate()
        return expr

    cls.__new__ = new_method  # type: ignore[assignment]
    cls.__getnewargs__ = _get_arguments  # type: ignore[assignment,method-assign]
    cls._hashable_content = _hashable_content_method  # type: ignore[method-assign]
    if non_sympy_fields:
        cls._eval_subs = _eval_subs_method  # type: ignore[method-assign]
        cls._xreplace = _xreplace_method  # type: ignore[method-assign]
    return cls


def _update_field_metadata(cls: T) -> T:
    """Set the :code:`sympify` metadata for all fields of a dataclass-like class."""
    for field in _get_fields(cls):  # type: ignore[arg-type]
        new_metadata = dict(field.metadata)
        if "sympify" not in new_metadata:
            new_metadata["sympify"] = True
        field.metadata = MappingProxyType(new_metadata)
    return cls


@overload
def _get_hashable_object(obj: type) -> str: ...  # type: ignore[overload-overlap]
@overload
def _get_hashable_object(obj: H) -> H: ...
@overload
def _get_hashable_object(obj: Any) -> str: ...
def _get_hashable_object(obj):
    if obj is None:
        obj = type(None)
    if isclass(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    try:
        hash(obj)
    except TypeError:
        return str(obj)
    return obj


def _extract_field_values(
    cls: type, *args, **kwargs
) -> tuple[dict[Field, Any], dict[str, Any]]:
    """Extract the attribute values from the constructor arguments.

    Returns a `tuple` of:

    1. the values for the dataclass fields extracted from :code:`*args` and
       :code:`**kwargs`,
    2. a `dict` of remaining keyword arguments that can be used hints for the
       constructed :class:`sp.Expr<sympy.core.expr.Expr>` instance.

    An attempt is made to get any missing attributes from the type hints in the class
    definition.
    """
    fields = _get_fields(cls)
    if len(args) == len(fields):
        return dict(zip(fields, args)), kwargs
    if len(args) > len(fields):
        msg = (
            f"Expecting {len(fields)} positional arguments"
            f" ({', '.join(f.name for f in fields)}), but got {len(args)}"
        )
        raise ValueError(msg)
    fields_with_values = dict(zip(fields, args))
    remaining_attrs = fields[len(args) :]
    missing: list[str] = []
    for field in remaining_attrs:
        if field.name in kwargs:
            fields_with_values[field] = kwargs.pop(field.name)
        elif field.default is MISSING:
            missing.append(field.name)
        else:
            fields_with_values[field] = field.default
    if missing:
        msg = f"Missing constructor arguments: {', '.join(missing)}"
        raise ValueError(msg)
    return fields_with_values, kwargs


def _safe_sympify(field: Field, value: dict[Field, Any]) -> dict[Field, Any]:
    if _is_sympify(field):
        try:
            return sp.sympify(value)
        except (sp.SympifyError, TypeError, SymPyDeprecationWarning) as exc:
            msg = (
                f"Attribute {field.name} could not be sympified. Did you forget to mark"
                " it with argument(sympify=False)?"
            )
            raise TypeError(msg) from exc
    return value


class LatexMethod(Protocol):
    def __call__(self, printer: LatexPrinter, *args) -> str: ...


@dataclass_transform(field_specifiers=(argument, _create_field))
def _implement_latex_repr(cls: type[T]) -> type[T]:
    repr_name = "_latex_repr_"
    _latex_repr_: LatexMethod | str | None = getattr(cls, repr_name, None)
    if _latex_repr_ is None:
        msg = (
            f"You need to define a {repr_name} str or method in order to decorate an"
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


@dataclass_transform(field_specifiers=(argument, _create_field))
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


@dataclass_transform(field_specifiers=(argument, _create_field))
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
    old_args = _get_arguments(self)
    new_args = list(old_args)
    for i, old_arg in enumerate(old_args):
        if not hasattr(old_arg, "_eval_subs"):
            continue
        if isclass(old_arg):
            continue
        new_attr = old_arg._subs(old, new, **hints)  # noqa: SLF001
        if not _aresame(new_attr, old_arg):
            hit = True
            new_args[i] = new_attr
    if hit:
        rv = self.func(*new_args)
        hack2 = hints.get("hack2", False)
        if hack2 and self.is_Mul and not rv.is_Mul:  # 2-arg hack
            coefficient = sp.S.One
            nonnumber = []
            for i in new_args:
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
    if not dataclasses.is_dataclass(self):
        return hashable_content
    remaining_content = (
        _get_hashable_object(getattr(self, field.name))
        for field in _get_fields(self)
        if not _is_sympify(field)
    )
    return (*hashable_content, *remaining_content)


def _xreplace_method(self, rule) -> tuple[sp.Expr, bool]:
    # https://github.com/sympy/sympy/blob/1.12/sympy/core/basic.py#L1233-L1253
    if self in rule:
        return rule[self], True
    if rule:
        new_args = []
        hit = False
        for arg in _get_arguments(self):
            if hasattr(arg, "_xreplace") and not isclass(arg):
                replace_result, is_replaced = arg._xreplace(rule)  # noqa: SLF001
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


def get_sympy_fields(cls) -> tuple[Field, ...]:
    return tuple(f for f in _get_fields(cls) if _is_sympify(f))


def _is_sympify(field: Field) -> bool:
    return bool(field.metadata.get("sympify"))
