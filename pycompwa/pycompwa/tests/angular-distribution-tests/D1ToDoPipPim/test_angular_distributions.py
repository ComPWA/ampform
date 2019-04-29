#!/usr/bin/env python3

import pycompwa.ui as ui
import logging
from math import cos

logging.basicConfig(level=logging.INFO)


def generate_model_xml():
    from pycompwa.expertsystem.ui.system_control import (
        StateTransitionManager, InteractionTypes)

    from pycompwa.expertsystem.amplitude.helicitydecay import (
        HelicityDecayAmplitudeGeneratorXML)

    from pycompwa.expertsystem.state.particle import (
        get_xml_label, XMLLabelConstants)
    # initialize the graph edges (initial and final state)
    initial_state = [("D1(2420)0", [-1, 1])]
    final_state = [("D0", [0]), ("pi-", [0]), ("pi+", [0])]

    tbd_manager = StateTransitionManager(initial_state, final_state,
                                         ['D*'])
    tbd_manager.number_of_threads = 1
    tbd_manager.set_allowed_interaction_types(
        [InteractionTypes.Strong])
    graph_interaction_settings_groups = tbd_manager.prepare_graphs()

    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings_groups)

    print("found " + str(len(solutions)) + " solutions!")

    print("intermediate states:")
    decinfo_label = get_xml_label(XMLLabelConstants.DecayInfo)
    for g in solutions:
        print(g.edge_props[1]['@Name'])
        for edge_props in g.edge_props.values():
            if decinfo_label in edge_props:
                del edge_props[decinfo_label]
                edge_props[decinfo_label] = {
                    get_xml_label(XMLLabelConstants.Type): "nonResonant"}

    xml_generator = HelicityDecayAmplitudeGeneratorXML()
    xml_generator.generate(solutions)
    xml_generator.write_to_file('model.xml')


def test_angular_distributions(make_plots=False):
    # generate_model_xml()
    import os
    thisdirectory = os.path.dirname(os.path.realpath(__file__))
    generate_data_samples(thisdirectory+"/model.xml", "plot.root")
    # In this example model the magnitude of A_00 = 0.5 and of A_10=A_-10=1
    # x = cos(theta) distribution from D1 decay should be 1.25 + 0.75*x^2
    # x = cos(theta') distribution from D* decay should be 1 - 0.75*x^2
    # dphi = phi - phi' distribution should be 1 - 1/2.25*cos(2*dphi)
    tuples = [(['theta_34_2'], {'number_of_bins': 120},
               lambda x: 1.25+0.75*x*x),
              (['theta_3_4_vs_2'], {'number_of_bins': 120},
               lambda x: 1-0.75*x*x),
              (['phi_34_2'],
               {'number_of_bins': 120, 'second_column_names': ['phi_3_4_vs_2'],
                'binary_operator': lambda x, y: x-y},
               lambda x: 1-1/2.25*cos(2*x))]
    compare_data_samples_and_theory("plot.root", tuples, make_plots)


def generate_data_samples(model_filename, output_filename):
    intensTrue, kinTrue = ui.create_intensity_and_kinematics(model_filename)

    # Generate phase space sample
    gen = ui.RootGenerator(
        kinTrue.get_particle_state_transition_kinematics_info(), 123456)
    phspSample = ui.generate_phsp(50000, gen)
    phspSample.convert_events_to_datapoints(kinTrue)

    # Generate Data
    sample = ui.generate(10000, kinTrue, gen, intensTrue)

    # Plotting
    kinTrue.create_all_subsystems()
    # recreate datapoints
    phspSample.convert_events_to_datapoints(kinTrue)
    sample.convert_events_to_datapoints(kinTrue)
    ui.create_rootplotdata(output_filename, kinTrue, sample,
                           phspSample, intensTrue)


def compare_data_samples_and_theory(input_rootfile,
                                    distribution_test_tuples,
                                    make_plots):
    from pycompwa.plotting import (
        make_binned_distributions, chisquare_test, plot_distributions_1d,
        function_to_histogram, scale_to_other_histogram,
        convert_helicity_column_name_to_title
    )
    from pycompwa.plotting.rootplotdatareader import open_compwa_plot_data

    plot_data = open_compwa_plot_data(input_rootfile)

    #data_variables = list(plot_data.data.dtype.names)
    #print("found data variables:", data_variables)

    for var_names, kwargs, func in distribution_test_tuples:
        binned_dists = make_binned_distributions(
            plot_data, var_names, **kwargs)
        for var_name, dists in binned_dists.items():
            data_hist = dists['data']
            if make_plots:
                function_hist = function_to_histogram(func, data_hist[0])
                function_hist = scale_to_other_histogram(
                    function_hist, data_hist)

                hist_bundle = {'data': data_hist,
                               'theory': function_hist + ({'fmt': '-'},)
                               }
                xtitle = convert_helicity_column_name_to_title(
                    var_name, plot_data)
                plot_distributions_1d(hist_bundle, var_name, xtitle=xtitle)

            chisquare_value, chisquare_error = chisquare_test(data_hist[:-1],
                                                              func)
            assert(abs(1.0 - chisquare_value) < 2 * chisquare_error)


if __name__ == '__main__':
    test_angular_distributions(make_plots=True)
