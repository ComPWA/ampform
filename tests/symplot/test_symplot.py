# pylint: disable=eval-used, no-self-use, protected-access, redefined-outer-name
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Pattern, Type, no_type_check

import pytest
import sympy as sp
from IPython.lib.pretty import pretty
from ipywidgets.widgets.widget_float import FloatSlider
from ipywidgets.widgets.widget_int import IntSlider

from symplot import Slider, SliderKwargs


class TestSliderKwargs:
    @pytest.fixture()
    def slider_kwargs(self) -> SliderKwargs:
        return SliderKwargs(
            sliders={
                "n": IntSlider(value=2, min=0, max=5, description="$n$"),
                "alpha": FloatSlider(
                    value=0.3, min=0.0, max=2.5, description=R"$\alpha$"
                ),
                R"\theta_1+2": FloatSlider(
                    value=1.5, min=0.0, max=3.14, description=R"$\theta_{1+2}$"
                ),
            },
            arg_to_symbol={
                "n": "n",
                "alpha": "alpha",
                "Dummy": R"\theta_1+2",
            },
        )

    def test_getitem(self, slider_kwargs: SliderKwargs) -> None:
        n_slider = slider_kwargs["n"]
        assert isinstance(n_slider, IntSlider)
        assert n_slider.description == "$n$"
        assert slider_kwargs[sp.Symbol("n")] == n_slider

        alpha_slider = slider_kwargs.get("alpha")
        assert alpha_slider is not None
        assert isinstance(alpha_slider, FloatSlider)
        assert alpha_slider.value == 0.3
        assert alpha_slider.description == R"$\alpha$"

        theta_slider = slider_kwargs[sp.Symbol(R"\theta_1+2", real=True)]
        assert theta_slider is not None
        assert isinstance(theta_slider, FloatSlider)
        assert theta_slider.max == 3.14
        assert theta_slider.description == R"$\theta_{1+2}$"
        assert slider_kwargs["Dummy"] == theta_slider

        error_message_pattern = r"is neither an argument nor a symbol name"
        with pytest.raises(KeyError, match=error_message_pattern):
            assert slider_kwargs["non-existent"]
        with pytest.raises(KeyError, match=error_message_pattern):
            assert slider_kwargs[sp.Symbol("Gamma")]

    def test_iter(self, slider_kwargs: SliderKwargs) -> None:
        assert set(slider_kwargs) == {"Dummy", "alpha", "n"}

    def test_kwargs(self, slider_kwargs: SliderKwargs) -> None:
        def some_function(**kwargs: Slider) -> None:
            assert set(kwargs) == {"Dummy", "alpha", "n"}
            assert isinstance(kwargs["alpha"], FloatSlider)
            assert isinstance(kwargs["n"], IntSlider)
            assert kwargs["Dummy"].max == 3.14

        some_function(**slider_kwargs)

    def test_len(self, slider_kwargs: SliderKwargs) -> None:
        assert len(slider_kwargs) == 3

    @pytest.mark.parametrize("repr_function", [repr, pretty])
    def test_repr(
        self, repr_function: Callable[[Any], str], slider_kwargs: SliderKwargs
    ) -> None:
        from_repr: SliderKwargs = eval(repr_function(slider_kwargs))
        assert set(from_repr) == set(slider_kwargs)
        for slider_name in slider_kwargs:
            slider = slider_kwargs[slider_name]
            slider_from_repr = from_repr[slider_name]
            assert slider.description == slider_from_repr.description
            assert slider.min == slider_from_repr.min
            assert slider.max == slider_from_repr.max
            assert slider.value == slider_from_repr.value

    @pytest.mark.parametrize(
        ("slider_name", "min_", "max_", "n_steps", "step_size"),
        [
            ("n", 10, 20, None, 1),
            ("n", 10, 20, 20, 1),
            ("alpha", -1.5, 0.7, None, 0.1),
            ("alpha", 8, 10, 4, 0.5),
        ],
    )
    def test_set_ranges(  # pylint: disable=too-many-arguments
        self,
        slider_name: str,
        min_: float,
        max_: float,
        n_steps: Optional[int],
        step_size: float,
        slider_kwargs: SliderKwargs,
    ) -> None:
        if n_steps is None:
            range_def = (min_, max_)
        else:
            range_def = (min_, max_, n_steps)  # type: ignore

        sliders = deepcopy(slider_kwargs)
        sliders.set_ranges({slider_name: range_def})
        slider = sliders[slider_name]
        assert slider.min == min_
        assert slider.max == max_
        assert slider.step == step_size

    @no_type_check
    def test_set_ranges_exceptions(self, slider_kwargs: SliderKwargs) -> None:
        with pytest.raises(TypeError, match=r"not a tuple"):
            slider_kwargs.set_ranges({"n": {"min": 0, "max": 1}})
        with pytest.raises(
            ValueError, match=r"shape \(min, max\) nor \(min, max, n_steps\)"
        ):
            slider_kwargs.set_ranges({"n": (1,)})
        with pytest.raises(
            ValueError, match=r"shape \(min, max\) nor \(min, max, n_steps\)"
        ):
            slider_kwargs.set_ranges({"alpha": (0.0, 0.1, 100, 200)})
        with pytest.raises(
            ValueError, match=r"Number of steps has to be positive"
        ):
            slider_kwargs.set_ranges({"n": (0, 10, -1)})

    def test_set_values(
        self, slider_kwargs: SliderKwargs, caplog: pytest.LogCaptureFixture
    ) -> None:
        sliders = deepcopy(slider_kwargs)
        assert sliders["n"].value == 2
        assert sliders["alpha"].value == 0.3
        assert sliders["Dummy"].value == 1.5

        # Using identifiers as kwargs
        with caplog.at_level(logging.WARNING):
            sliders.set_values(non_existent=0)
        sliders.set_values(n=1, Dummy=2)
        assert sliders["n"].value == 1
        assert sliders["Dummy"].value == 2

        # Using a symbol mapping
        n, alpha, theta = sp.symbols(R"n, alpha, \theta_1+2")
        sliders.set_values({n: 5, alpha: 2.1, theta: 1.57})
        assert sliders["n"].value == 5
        assert sliders["alpha"].value == 2.1
        assert sliders["Dummy"].value == 1.57

    @no_type_check
    def test_set_values_exceptions(self, slider_kwargs: SliderKwargs) -> None:
        with pytest.raises(
            TypeError, match="Positional arguments have to be of type dict"
        ):
            slider_kwargs.set_values(0)

    @pytest.mark.parametrize(
        ("sliders", "arg_to_symbol", "exception", "match"),
        [
            (
                {"b": FloatSlider()},
                {R"\alpha": "b"},
                ValueError,
                r'"\\alpha" in arg_to_symbol is not a valid identifier',
            ),
            (
                {sp.Symbol("b"): FloatSlider()},
                {"b": sp.Symbol("b")},
                TypeError,
                r'Slider name "b" is not of type str',
            ),
            (
                {R"\alpha": FloatSlider()},
                {"Dummy": "not a symbol"},
                ValueError,
                r'Slider with name "\\alpha" is not covered by arg_to_symbol',
            ),
            (
                {"one": None},
                {"one": "one"},
                TypeError,
                r"is not a valid ipywidgets slider",
            ),
        ],
    )
    def test_verify_arguments(
        self,
        sliders: Dict[str, FloatSlider],
        arg_to_symbol: Dict[str, str],
        exception: Type[ValueError],
        match: Pattern[str],
    ) -> None:
        with pytest.raises(exception, match=match):
            SliderKwargs._verify_arguments(sliders, arg_to_symbol)
