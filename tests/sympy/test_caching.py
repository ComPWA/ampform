from __future__ import annotations

import logging
import os

import pytest
import sympy as sp
from _pytest.logging import LogCaptureFixture

from ampform.dynamics import EnergyDependentWidth
from ampform.helicity import HelicityModel
from ampform.sympy import _warn_about_unsafe_hash, get_readable_hash


@pytest.mark.parametrize(
    ("assumptions", "expected_hash"),
    [
        # pylint: disable=use-dict-literal
        (dict(), "pythonhashseed-0+7459658071388516764"),
        (dict(real=True), "pythonhashseed-0+3665410414623666716"),
        (dict(rational=True), "pythonhashseed-0-7926839224244779605"),
    ],
)
def test_get_readable_hash(assumptions, expected_hash, caplog: LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed is None or not python_hash_seed.isdigit():
        assert h[:7] == "bbc9833"
        # pylint: disable=too-many-function-args
        if _warn_about_unsafe_hash.cache_info().hits == 0:
            assert "PYTHONHASHSEED has not been set." in caplog.text
            caplog.clear()
    elif python_hash_seed == "0":
        assert h == expected_hash
    else:
        pytest.skip("PYTHONHASHSEED has been set, but is not 0")
    assert caplog.text == ""


def test_get_readable_hash_energy_dependent_width():
    angular_momentum = sp.Symbol("L", integer=True)
    s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d", nonnegative=True)
    expr = EnergyDependentWidth(
        s=s,
        mass0=m0,
        gamma0=w0,
        m_a=m_a,
        m_b=m_b,
        angular_momentum=angular_momentum,
        meson_radius=d,
    )
    h = get_readable_hash(expr)
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed is None:
        pytest.skip("PYTHONHASHSEED has been set, but is not 0")
    assert h == "pythonhashseed-0+5847558977249966029"


def test_get_readable_hash_large(amplitude_model: tuple[str, HelicityModel]):
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed != "0":
        pytest.skip("PYTHONHASHSEED is not 0")
    formalism, model = amplitude_model
    expected_hash = {
        "canonical-helicity": "pythonhashseed-0-7143983882032045549",
        "helicity": "pythonhashseed-0+3357246175053927117",
    }[formalism]
    assert get_readable_hash(model.expression) == expected_hash
