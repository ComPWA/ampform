# cspell:ignore rpartition
# pylint: disable=redefined-builtin
"""Create interactive plots for `sympy` expressions.

The procedure to create interactive plots with for :mod:`sympy` expressions
with :doc:`mpl-interactions <mpl_interactions:index>` has been extracted to
this module.

The module is only available here, under the documentation. If this feature
turns out to be popular, it can be published as an independent package.

The package also provides other helpful functions, like
:func:`substitute_indexed_symbols`, that are useful when visualizing
`sympy` expressions.
"""

import inspect
import logging
from collections import abc
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import sympy as sp
from ipywidgets.widgets import FloatSlider, IntSlider
from sympy.printing.latex import translate

try:
    from IPython.lib.pretty import PrettyPrinter
except ImportError:
    PrettyPrinter = Any

Slider = Union[FloatSlider, IntSlider]
RangeDefinition = Union[
    Tuple[float, float],
    Tuple[float, float, Union[float, int]],
]


class SliderKwargs(abc.Mapping):
    """Wrapper around a `dict` of sliders that can serve as keyword arguments.

    Sliders can be defined in :func:`~mpl_interactions.pyplot.interactive_plot`
    through :term:`kwargs <python:keyword argument>`. This wrapper class can be
    used for that.
    """

    def __init__(
        self,
        sliders: Mapping[str, Slider],
        arg_to_symbol: Mapping[str, str],
    ) -> None:
        self._verify_arguments(sliders, arg_to_symbol)
        self._sliders = dict(sliders)
        self._arg_to_symbol = {
            arg: symbol
            for arg, symbol in arg_to_symbol.items()
            if symbol in self._sliders
        }

    @staticmethod
    def _verify_arguments(
        sliders: Mapping[str, Slider], arg_to_symbol: Mapping[str, str]
    ) -> None:
        symbol_names = set(arg_to_symbol.values())
        for arg_name in arg_to_symbol:
            if not arg_name.isidentifier():
                raise ValueError(
                    f'Argument "{arg_name}" in arg_to_symbol is not a'
                    " valid identifier for a Python variable"
                )
        for slider_name in sliders:
            if not isinstance(slider_name, str):
                raise TypeError(
                    f'Slider name "{slider_name}" is not of type str'
                )
            if slider_name not in symbol_names:
                raise ValueError(
                    f'Slider with name "{slider_name}" is not covered by '
                    "arg_to_symbol"
                )
        for name, slider in sliders.items():
            if not isinstance(slider, Slider.__args__):  # type: ignore
                raise TypeError(
                    f'Slider "{name}" is not a valid ipywidgets slider'
                )

    def __getitem__(self, key: Union[str, sp.Symbol]) -> Slider:
        """Get slider by symbol, symbol name, or argument name."""
        if isinstance(key, sp.Symbol):
            key = key.name
        if key in self._arg_to_symbol:
            slider_name = self._arg_to_symbol[key]
        else:
            slider_name = key
        if slider_name not in self._sliders:
            raise KeyError(f'"{key}" is neither an argument nor a symbol name')
        return self._sliders[slider_name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the arguments of the `.LambdifiedExpression`.

        This is useful for unpacking an instance of `SliderKwargs` as
        :term:`kwargs <python:keyword argument>`.
        """
        return self._arg_to_symbol.__iter__()

    def __len__(self) -> int:
        return len(self._sliders)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sliders={self._sliders}, "
            f"arg_to_symbol={self._arg_to_symbol})"
        )

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = self.__class__.__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}("):
                p.breakable()
                p.text("sliders=")
                p.pretty(self._sliders)
                p.text(",")
                p.breakable()
                p.text("arg_to_symbol=")
                p.pretty(self._arg_to_symbol)
                p.text(",")
            p.breakable()
            p.text(")")

    def set_values(self, *args: Dict[str, float], **kwargs: float) -> None:
        """Set initial values for the sliders.

        Either use a `dict` as input, or use :term:`kwargs <python:keyword
        argument>` with slider names as the keywords (see
        `.SliderKwargs.__getitem__`). This façade method exists in particular
        for `.parameter_defaults`.
        """
        value_mapping = _merge_args_kwargs(*args, **kwargs)
        for keyword, value in value_mapping.items():
            try:
                self[keyword].value = value
            except KeyError:
                logging.warning(
                    f'There is no slider with name or symbol "{keyword}"'
                )
                continue

    def set_ranges(  # noqa: R701
        self, *args: Dict[str, RangeDefinition], **kwargs: RangeDefinition
    ) -> None:
        """Set min, max and (optionally) the nr of steps for each slider.

        .. tip::
            :code:`n_steps` becomes the step **size** if its value is
            `float`.
        """
        range_definitions = _merge_args_kwargs(*args, **kwargs)
        for slider_name, range_def in range_definitions.items():
            if not isinstance(range_def, tuple):
                raise TypeError(
                    f'Range definition for slider "{slider_name}" is not a tuple'
                )
            slider = self[slider_name]
            if len(range_def) == 2:
                min_, max_ = range_def  # type: ignore
                step_size = slider.step
            elif len(range_def) == 3:
                min_, max_, n_steps = range_def  # type: ignore
                if n_steps <= 0:
                    raise ValueError("Number of steps has to be positive")
                if isinstance(n_steps, float):
                    step_size = n_steps
                else:
                    step_size = (max_ - min_) / n_steps
            else:
                raise ValueError(
                    f'Range definition {range_def} for slider "{slider_name}"'
                    " is neither of shape (min, max) nor (min, max, n_steps)"
                )
            if min_ > slider.max:
                slider.max = max_
                slider.min = min_
            else:
                slider.min = min_
                slider.max = max_
            if isinstance(slider, FloatSlider):
                slider.step = step_size


ValueType = TypeVar("ValueType")


def _merge_args_kwargs(
    *args: Dict[str, ValueType], **kwargs: ValueType
) -> Dict[str, ValueType]:
    r"""Merge positional `dict` arguments and keyword arguments into one `dict`.

    >>> _merge_args_kwargs(x="X", y="Y")
    {'x': 'X', 'y': 'Y'}
    >>> _merge_args_kwargs({R"\theta": 0}, a=1, b=2)
    {'\\theta': 0, 'a': 1, 'b': 2}
    """
    output_dict = {}
    for arg in args:
        if not isinstance(arg, dict):
            raise TypeError("Positional arguments have to be of type dict")
        output_dict.update(arg)
    output_dict.update(kwargs)
    return output_dict


def prepare_sliders(
    expression: sp.Expr, plot_symbol: sp.Symbol
) -> Tuple[Callable, SliderKwargs]:
    # cspell:ignore lambdifygenerated
    """Lambdify a `sympy` expression and create sliders for its arguments.

    >>> import sympy as sp
    >>> from symplot import prepare_sliders
    >>> n = sp.Symbol("n", integer=True)
    >>> x = sp.Symbol("x")
    >>> expression, sliders = prepare_sliders(x ** n, plot_symbol=x)
    >>> expression
    <function _lambdifygenerated at ...>
    >>> sliders
    SliderKwargs(...)
    """
    slider_symbols = _extract_slider_symbols(expression, plot_symbol)
    lambdified_expression = sp.lambdify(
        (plot_symbol, *slider_symbols),
        expression,
        modules="numpy",
    )
    sliders_mapping = {
        symbol.name: create_slider(symbol) for symbol in slider_symbols
    }
    symbols_names = map(lambda s: s.name, (plot_symbol, *slider_symbols))
    arg_names = inspect.signature(lambdified_expression).parameters
    arg_to_symbol = dict(zip(arg_names, symbols_names))
    sliders = SliderKwargs(sliders_mapping, arg_to_symbol)
    return lambdified_expression, sliders


def create_slider(symbol: sp.Symbol) -> Slider:
    """Create an `int` or `float` slider, depending on Symbol assumptions.

    The description for the slider is rendered as LaTeX from the
    `~sympy.core.symbol.Symbol` name.

    >>> import sympy as sp
    >>> from symplot import create_slider
    >>> create_slider(sp.Symbol("y"))
    FloatSlider(value=0.0, description='$y$')
    >>> create_slider(sp.Symbol("n", integer=True))
    IntSlider(value=0, description='$n$')
    """
    description = f"${sp.latex(symbol)}$"
    if symbol.is_integer:
        return IntSlider(description=description)
    return FloatSlider(description=description)


def _extract_slider_symbols(
    expression: sp.Expr, plot_symbol: sp.Symbol
) -> Tuple[sp.Symbol, ...]:
    """Extract sorted, remaining free symbols of a `sympy` expression."""
    if plot_symbol not in expression.free_symbols:
        raise ValueError(
            f"Expression does not contain a free symbol named {plot_symbol}"
        )
    ordered_symbols = sorted(expression.free_symbols, key=lambda s: s.name)
    return tuple(s for s in ordered_symbols if s != plot_symbol)


def partial_doit(
    expression: sp.Expr,
    doit_classes: Union[Type[sp.Basic], Tuple[Type[sp.Basic]]],
) -> sp.Expr:
    """Perform :meth:`~sympy.core.basic.Basic.doit` up to a certain level.

    Arguments
    ---------
    expression: the `~sympy.core.expr.Expr` on which you want to perform a
        :meth:`~sympy.core.basic.Basic.doit`.
    doit_classes: types on which the :meth:`~sympy.core.basic.Basic.doit`
        should be performed.
    """
    new_expression = expression
    for node in sp.preorder_traversal(expression):
        if isinstance(node, doit_classes):
            new_expression = new_expression.xreplace(
                {node: node.doit(deep=False)}
            )
    return new_expression


def _indexed_to_symbol(idx: sp.Indexed) -> sp.Symbol:
    base_name, _, _ = str(idx).rpartition("[")
    subscript = ",".join(map(str, idx.indices))
    if len(idx.indices) > 1:
        base_name = translate(base_name)
        subscript = "_{" + subscript + "}"
    return sp.Symbol(f"{base_name}{subscript}")


def substitute_indexed_symbols(expression: sp.Expr) -> sp.Expr:
    """Substitute `~sympy.tensor.indexed.IndexedBase` with symbols.

    See :doc:`compwa-org:report/008` for more info.
    """
    return expression.subs(
        {
            s: _indexed_to_symbol(s)
            for s in expression.free_symbols
            if isinstance(s, sp.Indexed)
        }
    )
