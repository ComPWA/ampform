from __future__ import annotations

import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import TYPE_CHECKING

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


def describe_cache_to_disk():
    def it_writes_atomically(tmp_path: Path, monkeypatch: MonkeyPatch):
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
    def it_repairs_corrupt_file(
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


def describe_get_readable_hash():
    @pytest.mark.parametrize(
        ("expected_hash", "assumptions"),
        [
            ("238baa6", dict()),
            ("72a46dd", dict(real=True)),
            ("f34cac7", dict(rational=True)),
        ],
        ids=["symbol", "symbol-real", "symbol-rational"],
    )
    def it_hashes_symbol_assumptions(
        assumptions: dict, expected_hash: str, caplog: LogCaptureFixture
    ):
        caplog.set_level(logging.WARNING)
        x, y = sp.symbols("x y", **assumptions)
        expr = x**2 + y
        h = get_readable_hash(expr)[:7]
        assert h == expected_hash
        assert not caplog.text

    def it_hashes_energy_dependent_widths_consistently():
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
        assert h == "1b63a45"

    @pytest.mark.parametrize(
        ("expected_hash", "obj"),
        [
            ("a36cb47", b"raw bytes"),
            ("cb5e378", "a string"),
            ("dc0dafb", (1, "a", (2, 3))),
            ("af94f10", frozendict({"b": 2, "a": 1})),
            ("d7210d2", frozendict({"x": frozenset({1, 2, 3}), "y": (4, 5)})),
        ],
        ids=["bytes", "str", "tuple", "frozendict", "nested"],
    )
    def it_hashes_containers(expected_hash: str, obj: Any):
        """Pin the hashes of the container types that :func:`.to_bytes` is used on."""
        assert get_readable_hash(obj)[:7] == expected_hash


def describe_large_hash():
    initial_state = [("J/psi(1S)", [-1, 1])]
    final_state = ["gamma", "pi0", "pi0"]
    allowed_intermediate_particles = ["f(0)(980)", "f(0)(1500)"]
    allowed_interaction_types = "strong"

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            (
                "627ee45" if sys.version_info >= (3, 11) else "206587e",
                "canonical-helicity",
            ),
            (
                "422ec6b" if sys.version_info >= (3, 11) else "4c37f61",
                "helicity",
            ),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    def it_hashes_reactions_consistently(expected_hash: str, formalism: SpinFormalism):
        if get_qrules_version() < (0, 10):
            pytest.skip("Hashes of are not stable in qrules<0.10")
        reaction = qrules.generate_transitions(
            initial_state=initial_state,
            final_state=final_state,
            allowed_intermediate_particles=allowed_intermediate_particles,
            allowed_interaction_types=allowed_interaction_types,
            formalism=formalism,
        )
        h = get_readable_hash(reaction)[:7]
        assert h == expected_hash

    @pytest.mark.parametrize(
        ("expected_hash", "formalism"),
        [
            ("6165ac0", "canonical-helicity"),
            ("8f6174c", "helicity"),
        ],
        ids=["canonical-helicity", "helicity"],
    )
    @pytest.mark.slow
    def it_hashes_amplitude_models_consistently(
        expected_hash: str, formalism: SpinFormalism
    ):
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
        assert intensity_hash == "1cd3567"

        amplitudes = frozendict({k: v.doit() for k, v in model.amplitudes.items()})
        unfolded_intensity = intensity.xreplace(amplitudes)
        unfolded_intensity_hash = get_readable_hash(unfolded_intensity)[:7]
        assert unfolded_intensity_hash == expected_hash
