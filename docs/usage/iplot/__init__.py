# pylint: disable=redefined-builtin
"""Tools and wrappers for :doc:`mpl_interactions:index`.

The procedure to create interactive plots with for :mod:`sympy` expressions
with :doc:`mpl-interactions <mpl_interactions:index>` has been extracted to
this module.

The module is only available here, under the documentation. If this feature
turns out to be popular, it can be published as an independent package.
"""

import logging
from collections import abc
from typing import Any, Dict, Iterator, Mapping, Tuple, Union

import sympy as sp
from ipywidgets.widgets import FloatSlider, IntSlider

try:
    from IPython.lib.pretty import PrettyPrinter
except ImportError:
    PrettyPrinter = Any

Slider = Union[FloatSlider, IntSlider]


class SliderKwargs(abc.Mapping):
    """Wrapper around a `dict` of sliders that can serve as keyword arguments.

    Sliders can be defined in `mpl_interactions.interactive_plot` through
    :code:`kwargs`. This wrapper class can be used for that.
    """

    def __init__(
        self,
        sliders: Mapping[str, Slider],
        arg_to_symbol: Mapping[str, str],
    ) -> None:
        self._verify_arguments(sliders, arg_to_symbol)
        self.__sliders = dict(sliders)
        self.__arg_to_symbol = {
            arg: symbol
            for arg, symbol in arg_to_symbol.items()
            if symbol in self.__sliders
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
        if key in self.__arg_to_symbol:
            slider_name = self.__arg_to_symbol[key]
        else:
            slider_name = key
        if slider_name not in self.__sliders:
            raise KeyError(f'"{key}" is neither an argument nor a symbol name')
        return self.__sliders[slider_name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the arguments of the `.LambdifiedExpression`.

        This"""
        return self.__arg_to_symbol.__iter__()

    def __len__(self) -> int:
        return len(self.__sliders)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sliders={self.__sliders}, "
            f"arg_to_symbol={self.__arg_to_symbol})"
        )

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = self.__class__.__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}("):
                p.breakable()
                p.text("sliders=")
                p.pretty(self.__sliders)
                p.text(",")
                p.breakable()
                p.text("arg_to_symbol=")
                p.pretty(self.__arg_to_symbol)
                p.text(",")
            p.breakable()
            p.text(")")

    def set_values(
        self, *args: Dict[Union[sp.Symbol, str], float], **kwargs: float
    ) -> None:
        """Set initial values for the sliders.

        Either use a `dict` as input, or use :code:`kwargs` with slider names
        as the keywords (see `.SliderKwargs.__getitem__`). This façade method
        exists in particular for `.parameter_defaults`.
        """
        if len(args) != 0:
            for arg in args:
                if not isinstance(arg, dict):
                    raise TypeError(
                        "Positional arguments have to be of type dict"
                    )
                kwargs.update(arg)
        for keyword, value in kwargs.items():
            try:
                self[keyword].value = value
            except KeyError:
                logging.warning(
                    f'There is no slider with name or symbol "{keyword}"'
                )
                continue

    def set_ranges(
        self,
        range_definitions: Mapping[
            str,
            Union[
                Tuple[float, float],
                Tuple[float, float, int],
            ],
        ],
    ) -> None:
        """Set min, max and (optionally) the number of steps for each slider."""
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
