#!/usr/bin/env python3

import logging
from math import cos

logging.basicConfig(level=logging.INFO)


def generate_model_xml():
    from pycompwa.expertsystem.ui.system_control import (
        StateTransitionManager,
        InteractionTypes,
    )

    from pycompwa.expertsystem.amplitude.helicitydecay import (
        HelicityDecayAmplitudeGeneratorXML,
    )

    from pycompwa.expertsystem.state.particle import (
        get_xml_label,
        XMLLabelConstants,
    )

    # initialize the graph edges (initial and final state)
    initial_state = [("D1(2420)0", [-1, 1])]
    final_state = [("D0", [0]), ("pi-", [0]), ("pi+", [0])]

    tbd_manager = StateTransitionManager(initial_state, final_state, ["D*"])
    tbd_manager.number_of_threads = 1
    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    graph_interaction_settings_groups = tbd_manager.prepare_graphs()

    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings_groups
    )

    print("found " + str(len(solutions)) + " solutions!")

    print("intermediate states:")
    decinfo_label = get_xml_label(XMLLabelConstants.DecayInfo)
    for g in solutions:
        print(g.edge_props[1]["@Name"])
        for edge_props in g.edge_props.values():
            if decinfo_label in edge_props:
                del edge_props[decinfo_label]
                edge_props[decinfo_label] = {
                    get_xml_label(XMLLabelConstants.Type): "nonResonant"
                }

    xml_generator = HelicityDecayAmplitudeGeneratorXML()
    xml_generator.generate(solutions)
    xml_generator.write_to_file("model.xml")


def test_angular_distributions(make_plots=False):
    from pycompwa.plotting import chisquare_test
    import os

    thisdirectory = os.path.dirname(os.path.realpath(__file__))
    import sys

    sys.path.append(thisdirectory + "/..")
    from distributioncomparison import (
        test_angular_distributions,
        ComparisonTuple,
    )

    # In this example model the magnitude of A_00 = 0.5 and of A_10=A_-10=1
    # x = cos(theta) distribution from D1 decay should be 1.25 + 0.75*x^2
    # x = cos(theta') distribution from D* decay should be 1 - 0.75*x^2
    # phi distribution of the D* decay should be 1 - 1/2.25*cos(2*phi)

    tuples = [
        ComparisonTuple(
            "theta_34_2",
            lambda x: 1.25 + 0.75 * x * x,
            chisquare_test,
            **{"number_of_bins": 80},
        ),
        ComparisonTuple(
            "theta_3_4_vs_2",
            lambda x: 1 - 0.75 * x * x,
            chisquare_test,
            **{"number_of_bins": 80},
        ),
        ComparisonTuple(
            "phi_3_4_vs_2",
            lambda x: 1 - 1 / 2.25 * cos(2 * x),
            chisquare_test,
            **{"number_of_bins": 80},
        ),
    ]
    test_angular_distributions(
        thisdirectory + "/model.xml", tuples, 20000, make_plots=make_plots
    )


if __name__ == "__main__":
    test_angular_distributions(make_plots=True)
