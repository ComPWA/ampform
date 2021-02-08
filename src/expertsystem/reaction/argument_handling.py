"""Handles argument handling for rules.

Responsibilities are the check of requirements for rules and the creation of
the arguments from general graph property maps. The information is extracted
from the type annotations of the rules.
"""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr

from expertsystem.particle import Parity

from .conservation_rules import (
    ConservationRule,
    EdgeQNConservationRule,
    GraphElementRule,
)
from .quantum_numbers import EdgeQuantumNumber, NodeQuantumNumber

Scalar = Union[int, float]

# InteractionRule = Union[EdgeQNConservationRule, ConservationRule]
Rule = Union[GraphElementRule, EdgeQNConservationRule, ConservationRule]

_ElementType = TypeVar("_ElementType", EdgeQuantumNumber, NodeQuantumNumber)

GraphElementPropertyMap = Dict[Type[_ElementType], Scalar]
GraphEdgePropertyMap = GraphElementPropertyMap[EdgeQuantumNumber]
GraphNodePropertyMap = GraphElementPropertyMap[NodeQuantumNumber]


def _is_optional(field_type: Optional[type]) -> bool:
    if (
        hasattr(field_type, "__origin__")
        and field_type.__origin__ is Union  # type: ignore
        and type(None) in field_type.__args__  # type: ignore
    ):
        return True
    return False


def _is_sequence_type(input_type: type) -> bool:
    # pylint: disable=unidiomatic-typecheck
    return hasattr(input_type, "__origin__") and (
        input_type.__origin__ is list  # type: ignore
        or input_type.__origin__ is tuple  # type: ignore
        or input_type.__origin__ is List  # type: ignore
        or input_type.__origin__ is Tuple  # type: ignore
    )


def _is_edge_quantum_number(qn_type: Any) -> bool:
    return qn_type in EdgeQuantumNumber.__args__  # type: ignore


def _is_node_quantum_number(qn_type: Any) -> bool:
    return qn_type in NodeQuantumNumber.__args__  # type: ignore


class _CompositeArgumentCheck:
    def __init__(
        self,
        class_field_types: Union[
            List[EdgeQuantumNumber], List[NodeQuantumNumber]
        ],
    ) -> None:
        self.__class_field_types = class_field_types

    def __call__(
        self,
        props: GraphElementPropertyMap,
    ) -> bool:
        return all(
            class_field_type in props
            for class_field_type in self.__class_field_types
        )


def _direct_qn_check(
    qn_type: Union[Type[EdgeQuantumNumber], Type[NodeQuantumNumber]]
) -> Callable[[GraphElementPropertyMap], bool]:
    def wrapper(props: GraphElementPropertyMap) -> bool:
        return qn_type in props

    return wrapper


def _sequence_input_check(func: Callable) -> Callable[[Sequence], bool]:
    def wrapper(edge_props_list: Sequence[Any]) -> bool:
        if not isinstance(edge_props_list, (list, tuple)):
            raise TypeError("Rule evaluated with invalid argument type...")

        return all(func(x) for x in edge_props_list)

    return wrapper


def _check_all_arguments(checks: List[Callable]) -> Callable[..., bool]:
    def wrapper(*args: Any) -> bool:
        return all(check(arg) for check, arg in zip(checks, args))

    return wrapper


class _ValueExtractor(Generic[_ElementType]):
    def __init__(self, obj_type: Optional[Type[_ElementType]]) -> None:
        self.__obj_type: Type[_ElementType] = obj_type  # type: ignore
        self.__function = self.__extract  # type: ignore

        if _is_optional(obj_type):
            self.__obj_type = obj_type.__args__[0]  # type: ignore
            self.__function = self.__optional_extract  # type: ignore

    def __call__(
        self, props: GraphElementPropertyMap[_ElementType]
    ) -> Optional[_ElementType]:
        return self.__function(props)

    def __optional_extract(
        self, props: GraphElementPropertyMap[_ElementType]
    ) -> Optional[_ElementType]:
        if self.__obj_type in props:
            return self.__extract(props)

        return None

    def __extract(
        self, props: GraphElementPropertyMap[_ElementType]
    ) -> _ElementType:
        value = props[self.__obj_type]
        if value is None:
            return None
        if (
            "__supertype__" in self.__obj_type.__dict__
            and self.__obj_type.__supertype__ == Parity  # type: ignore
        ):
            return self.__obj_type.__supertype__(value)  # type: ignore
        return self.__obj_type(value)  # type: ignore


class _CompositeArgumentCreator:
    def __init__(self, class_type: type) -> None:
        self.__class_type = class_type
        self.__extractors = dict(
            {
                class_field.name: _ValueExtractor[EdgeQuantumNumber](
                    class_field.type
                )
                if _is_edge_quantum_number(class_field.type)
                else _ValueExtractor[NodeQuantumNumber](class_field.type)
                for class_field in attr.fields(class_type)
            }
        )

    def __call__(
        self,
        props: GraphElementPropertyMap,
    ) -> Any:
        return self.__class_type(
            **{
                arg_name: extractor(props)  # type: ignore
                for arg_name, extractor in self.__extractors.items()
            }
        )


def _sequence_arg_builder(func: Callable) -> Callable[[Sequence], List[Any]]:
    def wrapper(edge_props_list: Sequence[Any]) -> List[Any]:
        if not isinstance(edge_props_list, (list, tuple)):
            raise TypeError("Rule evaluated with invalid argument type...")

        return [func(x) for x in edge_props_list if x]

    return wrapper


def _build_all_arguments(checks: List[Callable]) -> Callable:
    def wrapper(*args: Any) -> List[Any]:
        return [check(arg) for check, arg in zip(checks, args) if arg]

    return wrapper


class RuleArgumentHandler:
    def __init__(self) -> None:
        self.__rule_to_requirements_check: Dict[Rule, Callable] = {}
        self.__rule_to_argument_builder: Dict[Rule, Callable] = {}

    def __verify(self, rule_annotations: list) -> None:
        pass

    @staticmethod
    def __create_requirements_check(
        argument_types: List[type],
    ) -> Callable:
        individual_argument_checkers = []
        for input_type in argument_types:
            is_list = False
            qn_type = input_type
            if _is_sequence_type(input_type):
                qn_type = input_type.__args__[0]  # type: ignore
                is_list = True

            if attr.has(qn_type):
                class_field_types = [
                    class_field.type
                    for class_field in attr.fields(qn_type)
                    if not _is_optional(class_field.type)
                ]
                qn_check_function: Callable[
                    ..., bool
                ] = _CompositeArgumentCheck(
                    class_field_types  # type: ignore
                )
            else:
                qn_check_function = _direct_qn_check(qn_type)

            if is_list:
                qn_check_function = _sequence_input_check(qn_check_function)

            individual_argument_checkers.append(qn_check_function)

        return _check_all_arguments(individual_argument_checkers)

    @staticmethod
    def __create_argument_builder(
        argument_types: List[type],
    ) -> Callable:
        individual_argument_builders = []
        for input_type in argument_types:
            is_list = False
            qn_type = input_type
            if _is_sequence_type(input_type):
                qn_type = input_type.__args__[0]  # type: ignore
                is_list = True

            if attr.has(qn_type):
                arg_builder: Callable[..., Any] = _CompositeArgumentCreator(
                    qn_type
                )
            else:
                if _is_edge_quantum_number(qn_type):
                    arg_builder = _ValueExtractor[EdgeQuantumNumber](qn_type)
                elif _is_node_quantum_number(qn_type):
                    arg_builder = _ValueExtractor[NodeQuantumNumber](qn_type)
                else:
                    raise TypeError(
                        f"Quantum number type {qn_type} is not supported."
                        " Has to be of type Edge/NodeQuantumNumber."
                    )

            if is_list:
                arg_builder = _sequence_arg_builder(arg_builder)

            individual_argument_builders.append(arg_builder)

        return _build_all_arguments(individual_argument_builders)

    def register_rule(self, rule: Rule) -> Tuple[Callable, Callable]:
        if (
            rule not in self.__rule_to_requirements_check
            or rule not in self.__rule_to_argument_builder
        ):
            rule_annotations = list()
            rule_func_signature = inspect.signature(rule)
            if not rule_func_signature.return_annotation:
                raise TypeError(
                    f"missing return type annotation for rule {str(rule)}"
                )
            for par in rule_func_signature.parameters.values():
                if not par.annotation:
                    raise TypeError(
                        f"missing type annotations for argument {par.name}"
                        f" of rule {str(rule)}"
                    )
                rule_annotations.append(par.annotation)

            # check type annotations are legal
            try:
                self.__verify(rule_annotations)
            except TypeError as exception:
                raise TypeError(
                    f"rule {str(rule)}: {str(exception)}"
                ) from exception

            # then create requirements check function and add to dict
            self.__rule_to_requirements_check[
                rule
            ] = self.__create_requirements_check(rule_annotations)

            # then create arguments builder function and add to dict
            self.__rule_to_argument_builder[
                rule
            ] = self.__create_argument_builder(rule_annotations)

        return (
            self.__rule_to_requirements_check[rule],
            self.__rule_to_argument_builder[rule],
        )


def get_required_qns(
    rule: Rule,
) -> Tuple[Set[Type[EdgeQuantumNumber]], Set[Type[NodeQuantumNumber]]]:
    rule_annotations = list()
    for par in inspect.signature(rule).parameters.values():
        if not par.annotation:
            raise TypeError(f"missing type annotations for rule {str(rule)}")
        rule_annotations.append(par.annotation)

    required_edge_qns: Set[Type[EdgeQuantumNumber]] = set()
    required_node_qns: Set[Type[NodeQuantumNumber]] = set()

    arg_counter = 0
    for input_type in rule_annotations:
        class_type = input_type
        if _is_sequence_type(input_type):
            class_type = input_type.__args__[0]

        if attr.has(class_type):
            for class_field in attr.fields(class_type):
                field_type = (
                    class_field.type.__args__[0]  # type: ignore
                    if _is_optional(class_field.type)
                    else class_field.type
                )
                if _is_edge_quantum_number(field_type):
                    required_edge_qns.add(field_type)
                else:
                    required_node_qns.add(field_type)
        else:
            if _is_edge_quantum_number(class_type):
                required_edge_qns.add(class_type)
            else:
                required_node_qns.add(class_type)
        arg_counter += 1

    return (required_edge_qns, required_node_qns)
