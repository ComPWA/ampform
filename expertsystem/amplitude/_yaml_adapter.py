"""Temporary helper functions to convert from XML to a YAML structure."""

from typing import (
    Any,
    Dict,
    Generator,
    List,
    Union,
)

from expertsystem import io


def to_dynamics(recipe: Dict[str, Any]) -> Dict[str, Any]:
    def determine_type(definition: Dict[str, Any]) -> str:
        decay_type = definition["Type"]
        decay_type_xml_to_yaml = {
            "relativisticBreitWigner": "RelativisticBreitWigner",
            "nonResonant": "NonDynamic",
        }
        for xml_type, yaml_type in decay_type_xml_to_yaml.items():
            decay_type = decay_type.replace(xml_type, yaml_type)
        return decay_type

    def determine_form_factor(
        definition: Dict[str, Any],
        particle_xml: Dict[str, Any],
        decay_type: str,
    ) -> Dict[str, Any]:
        form_factor = definition.get("FormFactor", None)
        if form_factor is None:
            if decay_type == "NonDynamic":
                form_factor = {
                    "Type": "BlattWeisskopf",
                    "MesonRadius": 1.0,
                }
        else:
            parameters = _safe_wrap_in_list(
                particle_xml["DecayInfo"]["Parameter"]
            )
            meson_radius_candidates = [
                par for par in parameters if par["Type"] == "MesonRadius"
            ]
            if meson_radius_candidates:
                meson_radius_xml = meson_radius_candidates[0]
                meson_radius_value = float(meson_radius_xml["Value"])
                if len(meson_radius_xml) == 1:
                    form_factor["MesonRadius"] = meson_radius_value
                else:
                    meson_radius_yml = {"Value": meson_radius_value}
                    optional_keys = [
                        ("Min", float),
                        ("Max", float),
                        ("Fix", bool),
                    ]
                    for key, converter in optional_keys:
                        if key in meson_radius_xml:
                            meson_radius_yml[key] = converter(
                                meson_radius_xml[key]
                            )
                    form_factor["MesonRadius"] = meson_radius_yml
        return form_factor

    particle_list_xml = recipe["ParticleList"]["Particle"]
    dynamics_yml: Dict[str, Any] = dict()
    for particle_xml in particle_list_xml:
        name = particle_xml["Name"]
        decay_info_xml = particle_xml.get("DecayInfo", None)
        if decay_info_xml and "Type" in decay_info_xml:
            decay_type = determine_type(decay_info_xml)
            form_factor = determine_form_factor(
                decay_info_xml, particle_xml, decay_type
            )
            decay_info_yml = {"Type": decay_type, "FormFactor": form_factor}
            dynamics_yml[name] = decay_info_yml
    return dynamics_yml


def to_intensity(recipe: Dict[str, Any]) -> Dict[str, Any]:
    intensity_xml = recipe["Intensity"]
    intensity_yml = _extract_intensity_component(intensity_xml)
    return intensity_yml


def to_kinematics_dict(recipe: Dict[str, Any]) -> Dict[str, Any]:
    kinematics_xml = recipe["HelicityKinematics"]
    initialstate_xml = kinematics_xml["InitialState"]["Particle"]
    initial_state_yml = list()
    for state_def_xml in initialstate_xml:
        state_def_yml = {
            "Particle": state_def_xml["Name"],
            "ID": int(state_def_xml["Id"]),
        }
        initial_state_yml.append(state_def_yml)
    kinematics_yaml = {
        "Type": "Helicity",
        "InitialState": _to_state_list(kinematics_xml, "InitialState"),
        "FinalState": _to_state_list(kinematics_xml, "FinalState"),
    }
    return kinematics_yaml


def to_parameter_list(recipe: Dict[str, Any]) -> List[Dict[str, Any]]:
    intensity_dict = recipe["Intensity"]
    xml_parameter_defs = _extract_parameter_definitions_from_intensity(
        intensity_dict
    )
    parameter_dict: Dict[str, Dict[str, Any]] = dict()
    for parameter_xml in xml_parameter_defs:
        name = parameter_xml["Name"]
        if name not in parameter_dict:
            parameter_yml = dict()
            parameter_yml["Type"] = parameter_xml["Type"]
            parameter_yml["Value"] = parameter_xml["Value"]
            if "Fix" in parameter_xml:
                is_fix = bool(parameter_xml["Fix"])
                if is_fix:
                    parameter_yml["Fix"] = is_fix
            parameter_dict[name] = parameter_yml
    parameter_list = [
        {"Name": name, **value} for name, value in parameter_dict.items()
    ]
    return parameter_list


def to_particle_dict(recipe: Dict[str, Any]) -> Dict[str, Any]:
    particle_list_xml = recipe["ParticleList"]["Particle"]
    particle_collection = io.xml.dict_to_particle_collection(particle_list_xml)
    particle_list_yaml = io.yaml.object_to_dict(particle_collection)
    particle_list_yaml = particle_list_yaml["ParticleList"]
    return particle_list_yaml


def _extract_parameter_definitions_from_intensity(
    definition: dict,
) -> List[Dict[str, Any]]:
    if not isinstance(definition, dict):
        return list()
    search_results = gen_dict_extract("Parameter", definition)
    parameter_defs = list()
    for item in search_results:
        if isinstance(item, list):
            parameter_defs.extend(item)
        else:
            parameter_defs.append(item)
    return parameter_defs


def gen_dict_extract(
    search_term: str, dictionary: Dict[str, Any]
) -> Generator:
    """See https://stackoverflow.com/a/29652561."""
    for key, value in dictionary.items():
        if key == search_term:
            yield value
        if isinstance(value, dict):
            for result in gen_dict_extract(search_term, value):
                yield result
        elif isinstance(value, list):
            for item in value:
                for result in gen_dict_extract(search_term, item):
                    yield result


def _to_scalar(
    definition: Dict[str, str], key: str = "Value"
) -> Union[float, int]:
    value = _downgrade_float(float(definition[key]))
    return value


def _downgrade_float(value: float) -> Union[float, int]:
    if value.is_integer():
        return int(value)
    return value


def _to_isospin(definition: Dict[str, Any]) -> Union[float, Dict[str, float]]:
    """Isospin is 'stable', so always needs a projection."""
    value = _to_scalar(definition, "Value")
    if value == 0:
        return value
    return {
        "Value": value,
        "Projection": _to_scalar(definition, "Projection"),
    }


def _to_state_list(
    definition: Dict[str, Any], key: str
) -> List[Dict[str, Union[str, int]]]:
    state_list_xml = definition[key]["Particle"]
    state_list_yml = list()
    for state_def_xml in state_list_xml:
        state_def_yml = {
            "Particle": state_def_xml["Name"],
            "ID": int(state_def_xml["Id"]),
        }
        state_list_yml.append(state_def_yml)
    return state_list_yml


def _extract_intensity_component(definition: Dict[str, Any]) -> Dict[str, Any]:
    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    output_dict = dict()
    class_name = definition["Class"]
    if class_name == "StrengthIntensity":
        output_dict["Class"] = class_name
        output_dict["Component"] = definition["Component"]
        parameters_xml = _safe_wrap_in_list(definition["Parameter"])
        parameter_names = [par["Name"] for par in parameters_xml]
        output_dict["Strength"] = parameter_names[0]
        output_dict["Intensity"] = _extract_intensity_component(
            definition["Intensity"]
        )
    elif class_name == "NormalizedIntensity":
        output_dict["Class"] = class_name
        output_dict["Intensity"] = _extract_intensity_component(
            definition["Intensity"]
        )
    elif class_name == "IncoherentIntensity":
        output_dict["Class"] = class_name
        intensities_yml = list()
        for intensity in definition["Intensity"]:
            intensities_yml.append(_extract_intensity_component(intensity))
        output_dict["Intensities"] = intensities_yml
    elif class_name == "CoherentIntensity":
        output_dict["Class"] = class_name
        output_dict["Component"] = definition["Component"]
        amplitudes_xml = _safe_wrap_in_list(definition["Amplitude"])
        amplitudes_yml = [
            _extract_intensity_component(amplitude)
            for amplitude in amplitudes_xml
        ]
        output_dict["Amplitudes"] = amplitudes_yml
    elif class_name == "CoefficientAmplitude":
        output_dict["Class"] = class_name
        output_dict["Component"] = definition["Component"]
        if "PreFactor" in definition:
            output_dict["PreFactor"] = definition["PreFactor"]
        parameters_xml = _safe_wrap_in_list(definition["Parameter"])
        parameter_names = [par["Name"] for par in parameters_xml]
        for name in parameter_names:
            type_name = name.split("_")[0]
            output_dict[type_name] = name
        amplitudes_xml = definition["Amplitude"]
        amplitudes_yml = _extract_intensity_component(amplitudes_xml)  # type: ignore
        output_dict["Amplitude"] = amplitudes_yml
    elif class_name == "SequentialAmplitude":
        output_dict["Class"] = class_name
        amplitudes_xml = _safe_wrap_in_list(definition["Amplitude"])
        amplitudes_yml = [
            _extract_intensity_component(amplitude)
            for amplitude in amplitudes_xml
        ]
        output_dict["Amplitudes"] = amplitudes_yml
    elif class_name == "HelicityDecay":
        output_dict["Class"] = class_name
        decay_particle = definition["DecayParticle"]
        decay_particle["Helicity"] = float(decay_particle["Helicity"])
        output_dict["DecayParticle"] = decay_particle
        decay_products = definition["DecayProducts"]["Particle"]
        decay_products = _safe_wrap_in_list(decay_products)
        for decay_product in decay_products:
            final_states = decay_product["FinalState"].split(" ")
            final_states = [int(state_id) for state_id in final_states]
            decay_product["FinalState"] = final_states
            decay_product["Helicity"] = float(decay_product["Helicity"])
        output_dict["DecayProducts"] = decay_products
        if "RecoilSystem" in definition:
            recoil_system = definition["RecoilSystem"]
            recoil_final_states = recoil_system["RecoilFinalState"].split(" ")
            recoil_system["RecoilFinalState"] = [
                int(state_id) for state_id in recoil_final_states
            ]
            if "ParentRecoilFinalState" in recoil_system:
                parent_recoil_final_states = recoil_system[
                    "ParentRecoilFinalState"
                ].split(" ")
                recoil_system["ParentRecoilFinalState"] = [
                    int(state_id) for state_id in parent_recoil_final_states
                ]

            output_dict["RecoilSystem"] = recoil_system
        if "CanonicalSum" in definition:
            cano_sum_old = definition["CanonicalSum"]
            cano_sum_new = dict()
            clebsch_gordan_list = _safe_wrap_in_list(
                cano_sum_old["ClebschGordan"]
            )
            for clebsch_gordan_old in clebsch_gordan_list:
                type_name = clebsch_gordan_old["Type"]
                clebsch_gordan_new = {
                    "J": clebsch_gordan_old["J"],
                    "M": clebsch_gordan_old["M"],
                }
                attributes = {
                    key[1:]: value
                    for key, value in clebsch_gordan_old.items()
                    if key.startswith("@")
                }
                clebsch_gordan_new.update(attributes)
                embed_clebsch_gordan = {"ClebschGordan": clebsch_gordan_new}
                cano_sum_new[type_name] = embed_clebsch_gordan
            output_dict["Canonical"] = cano_sum_new
    return output_dict


def _safe_wrap_in_list(input_object: Any) -> List[Any]:
    if isinstance(input_object, list):
        return input_object
    return [input_object]
