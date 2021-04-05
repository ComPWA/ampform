# cspell:ignore isclass
"""Check duck typing.

Ideally, the rule input classes use a `~typing.Protocol`. This is not possible,
however, because of https://github.com/python/mypy/issues/6850. Duck typing is
therefore checked through functions defined in this test.
"""

import inspect
from typing import Set, Type

import attr

from expertsystem.reaction import conservation_rules
from expertsystem.reaction.particle import Particle
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumbers,
)

RULE_INPUT_CLASSES = set(
    getattr(conservation_rules, name)
    for name in dir(conservation_rules)
    if name.endswith("Input")
)


def test_protocol_compliance():
    edge_input_classes = __get_duck_types(EdgeQuantumNumbers)
    assert edge_input_classes == {
        conservation_rules.CParityEdgeInput,
        conservation_rules.GParityEdgeInput,
        conservation_rules.HelicityParityEdgeInput,
        conservation_rules.IdenticalParticleSymmetryOutEdgeInput,
        conservation_rules.IsoSpinEdgeInput,
        conservation_rules.MassEdgeInput,
        conservation_rules.SpinEdgeInput,
        conservation_rules.GellMannNishijimaInput,
    }
    node_input_classes = __get_duck_types(NodeQuantumNumbers)
    assert node_input_classes == {
        conservation_rules.CParityNodeInput,
        conservation_rules.GParityNodeInput,
        conservation_rules.SpinNodeInput,
        conservation_rules.SpinMagnitudeNodeInput,
    }
    assert edge_input_classes | node_input_classes == RULE_INPUT_CLASSES

    interaction_input_classes = __get_duck_types(InteractionProperties)
    assert interaction_input_classes == node_input_classes

    particle_input_classes = __get_duck_types(Particle)
    assert particle_input_classes == {
        conservation_rules.MassEdgeInput,
    }


def __get_duck_types(instance: Type) -> Set[Type]:
    """Get a `set` of rule input classes that this instance can duck type."""
    return {c for c in RULE_INPUT_CLASSES if __is_duck_type(c, instance)}


def test_is_duck_type():
    assert __is_duck_type(NodeQuantumNumbers, InteractionProperties)
    assert __is_duck_type(
        conservation_rules.CParityNodeInput, NodeQuantumNumbers
    )
    assert __is_duck_type(conservation_rules.MassEdgeInput, Particle)


def __is_duck_type(duck_type: Type, class_type: Type) -> bool:
    """See https://github.com/python/mypy/issues/6850."""
    return __get_members(duck_type) <= __get_members(class_type)


def test_get_members():
    assert __get_members(conservation_rules.GParityEdgeInput) == {
        "g_parity",
        "isospin_magnitude",
        "pid",
        "spin_magnitude",
    }
    assert __get_members(NodeQuantumNumbers) == {
        "l_magnitude",
        "l_projection",
        "parity_prefactor",
        "s_magnitude",
        "s_projection",
    }
    assert __get_members(conservation_rules.SpinNodeInput) <= __get_members(
        NodeQuantumNumbers
    )
    assert __get_members(InteractionProperties) == {
        "l_magnitude",
        "l_projection",
        "parity_prefactor",
        "s_magnitude",
        "s_projection",
    }


def __get_members(class_type: Type) -> Set[str]:
    use_attrs = class_type not in {EdgeQuantumNumbers, NodeQuantumNumbers}
    if use_attrs and attr.has(class_type):
        return {f.name for f in attr.fields(class_type)}
    return {
        a.name
        for a in inspect.classify_class_attrs(class_type)
        if not a.name.startswith("__")
    }
