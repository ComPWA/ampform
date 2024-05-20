from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform.dynamics import EnergyDependentWidth
from ampform.sympy._cache import _warn_about_unsafe_hash, get_readable_hash

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from ampform.helicity import HelicityModel


@pytest.mark.parametrize(
    ("assumptions", "expected_hashes"),
    [
        (
            dict(),
            {
                "3.7": 7060330373292767180,
                "3.8": 7459658071388516764,
                "3.9": 7459658071388516764,
                "3.10": 7459658071388516764,
                "3.11": 8778804591879682108,
                "3.12": 8778804591879682108,
            },
        ),
        (
            dict(real=True),
            {
                "3.7": 118635607833730864,
                "3.8": 3665410414623666716,
                "3.9": 3665410414623666716,
                "3.10": 3665410414623666716,
                "3.11": -7967572625470457155,
                "3.12": -7967572625470457155,
            },
        ),
        (
            dict(rational=True),
            {
                "3.7": -1011754479721050016,
                "3.8": -7926839224244779605,
                "3.9": -7926839224244779605,
                "3.10": -7926839224244779605,
                "3.11": -8321323707982755013,
                "3.12": -8321323707982755013,
            },
        ),
    ],
)
def test_get_readable_hash(assumptions, expected_hashes, caplog: LogCaptureFixture):
    python_version = ".".join(map(str, sys.version_info[:2]))
    expected_hash = expected_hashes[python_version]
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h_str = get_readable_hash(expr)
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed is None:
        assert h_str[:7] == "bbc9833"
        if _warn_about_unsafe_hash.cache_info().hits == 0:
            assert "PYTHONHASHSEED has not been set." in caplog.text
            caplog.clear()
    elif python_hash_seed == "0":
        h = int(h_str.replace("pythonhashseed-0", ""))
        assert h == expected_hash
    else:
        pytest.skip(f"PYTHONHASHSEED has been set, but is {python_hash_seed}, not 0")
    assert not caplog.text


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
        pytest.skip("PYTHONHASHSEED has not been set")
    if python_hash_seed != "0":
        pytest.skip(f"PYTHONHASHSEED is not set to 0, but to {python_hash_seed}")
    if sys.version_info < (3, 8):
        assert h == "pythonhashseed-0-6795262906917625791"
    elif sys.version_info >= (3, 11):
        assert h == "pythonhashseed-0+4377931190501974271"
    else:
        assert h == "pythonhashseed-0+8267198661922532208"


def test_get_readable_hash_large(amplitude_model: tuple[str, HelicityModel]):
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    if python_hash_seed != "0":
        pytest.skip("PYTHONHASHSEED is not 0")
    formalism, model = amplitude_model
    if sys.version_info < (3, 8):
        # https://github.com/ComPWA/ampform/actions/runs/3277058875/jobs/5393849802
        # https://github.com/ComPWA/ampform/actions/runs/3277143883/jobs/5394043014
        expected_hash = {
            "canonical-helicity": "pythonhashseed-0-4409019767276782833",
            "helicity": "pythonhashseed-0+8495836064961054249",
        }[formalism]
    elif sys.version_info >= (3, 11):
        expected_hash = {
            "canonical-helicity": "pythonhashseed-0-8140852268928771574",
            "helicity": "pythonhashseed-0-991855900379383849",
        }[formalism]
    else:
        expected_hash = {
            "canonical-helicity": "pythonhashseed-0+3166036244969111461",
            "helicity": "pythonhashseed-0+4247688887304834148",
        }[formalism]
    assert get_readable_hash(model.expression) == expected_hash
