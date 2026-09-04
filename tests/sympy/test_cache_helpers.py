from __future__ import annotations

import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING, ClassVar

import pytest
import qrules
import sympy as sp
from frozendict import frozendict

import ampform
from ampform._qrules import get_qrules_version
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import RelativisticBreitWignerBuilder
from ampform.sympy import _cache
from ampform.sympy._cache import cache_to_disk, get_readable_hash

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from _typeshed import SupportsWrite
    from qrules.transition import SpinFormalism


def test_cache_to_disk_writes_atomically(tmp_path: Path, monkeypatch: MonkeyPatch):
    monkeypatch.setattr(_cache, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("NO_CACHE", raising=False)
    first_write_started = Event()
    continue_first_write = Event()
    dump_calls = 0

    def dump_in_two_steps(value: Any, stream: SupportsWrite[bytes]) -> None:
        nonlocal dump_calls
        dump_calls += 1
        data = pickle.dumps(value)
        if dump_calls == 1:
            stream.write(data[:1])
            first_write_started.set()
            continue_first_write.wait(timeout=5)
            stream.write(data[1:])
        else:
            stream.write(data)

    @cache_to_disk(dump_function=dump_in_two_steps)
    def cached_function():
        return "result"

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_result = executor.submit(cached_function)
        try:
            assert first_write_started.wait(timeout=5)
            assert cached_function() == "result"
        finally:
            continue_first_write.set()
        assert first_result.result(timeout=5) == "result"

    assert not list(tmp_path.rglob("*.tmp"))
    cache_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(cache_files) == 1
    assert pickle.loads(cache_files[0].read_bytes()) == "result"


@pytest.mark.parametrize("corrupt_data", [b"", b"not a pickle"])
def test_cache_to_disk_repairs_corrupt_file(
    corrupt_data: bytes, tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(_cache, "_get_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("NO_CACHE", raising=False)
    call_count = 0

    @cache_to_disk
    def cached_function():
        nonlocal call_count
        call_count += 1
        return "result"

    assert cached_function() == "result"
    cache_files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(cache_files) == 1
    cache_files[0].write_bytes(corrupt_data)

    assert cached_function() == "result"
    assert cached_function() == "result"
    assert call_count == 2


@pytest.mark.parametrize(
    ("expected_hash", "assumptions"),
    [
        ("a7559ca", dict()),
        ("278bcee", dict(real=True)),
        ("bc417f2", dict(rational=True)),
    ],
    ids=["symbol", "symbol-real", "symbol-rational"],
)
def test_get_readable_hash(
    assumptions: dict, expected_hash: str, caplog: LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    x, y = sp.symbols("x y", **assumptions)
    expr = x**2 + y
    h = get_readable_hash(expr)[:7]
    assert h == expected_hash
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
    h = get_readable_hash(expr)[:7]
    assert h == "3d076c6"


class TestLargeHash:
    initial_state: ClassVar = [("J/psi(1S)", [-1, 1])]
    final_state: ClassVar = ["gamma", "pi0", "pi0"]
    allowed_intermediate_particles: ClassVar = ["f(0)(980)", "f(0)(1500)"]
    allowed_interaction_types: ClassVar = "strong"

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            (
                "762cc00" if sys.version_info >= (3, 11) else "1f5ac33",
                "canonical-helicity",
            ),
            (
                "17fefe5" if sys.version_info >= (3, 11) else "7b5fad1",
                "helicity",
            ),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    def test_reaction(self, expected_hash: str, formalism: SpinFormalism):
        if get_qrules_version() < (0, 10):
            pytest.skip("Hashes of are not stable in qrules<0.10")
        reaction = qrules.generate_transitions(
            initial_state=self.initial_state,
            final_state=self.final_state,
            allowed_intermediate_particles=self.allowed_intermediate_particles,
            allowed_interaction_types=self.allowed_interaction_types,
            formalism=formalism,
        )
        h = get_readable_hash(reaction)[:7]
        assert h == expected_hash

    @pytest.mark.parametrize(
        ("expected_hashes", "formalism"),
        [
            ({"2b77221", "8397450", "a80fbd1", "dc1ee0e"}, "canonical-helicity"),
            ({"7be27a6", "8c8c070", "aced899", "cbd5ff0", "ceecb32"}, "helicity"),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    @pytest.mark.slow
    def test_amplitude_model(self, expected_hashes: set[str], formalism: SpinFormalism):
        reaction = qrules.generate_transitions(
            initial_state=[("J/psi(1S)", [-1, 1])],
            final_state=["p~", "K0", "Sigma+"],
            allowed_intermediate_particles=[
                "N(1650)+",  # largest branching fraction
                "N(1675)+",  # high LS couplings
                "Sigma(1385)",  # largest branching fraction
                "Sigma(1775)",  # high LS couplings
            ],
            allowed_interaction_types="strong",
            formalism=formalism,
        )
        model_builder = ampform.get_builder(reaction)
        has_ls_couplings = formalism == "canonical-helicity"
        dynamics_builder = RelativisticBreitWignerBuilder(
            form_factor=has_ls_couplings,
            energy_dependent_width=has_ls_couplings,
        )
        for name in reaction.get_intermediate_particles().names:
            model_builder.dynamics.assign(name, dynamics_builder)
        model = model_builder.formulate()

        intensity = model.intensity.doit()
        assert any(isinstance(s, sp.Indexed) for s in intensity.free_symbols)

        intensity_hash = get_readable_hash(intensity)[:7]
        assert intensity_hash in {"c83b853", "d113a38"}

        amplitudes = frozendict({k: v.doit() for k, v in model.amplitudes.items()})
        unfolded_intensity = intensity.xreplace(amplitudes)
        unfolded_intensity_hash = get_readable_hash(unfolded_intensity)[:7]
        assert unfolded_intensity_hash in expected_hashes
        # Hash is not fully stable yet! See https://github.com/ComPWA/ampform-dpd/discussions/163
