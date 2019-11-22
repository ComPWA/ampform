import pycompwa.ui as ui


class ComparisonTuple:
    def __init__(self, variable_names, required_distribution, test_function,
                 **kwargs):
        self.variable_names = variable_names
        self.required_distribution = required_distribution
        self.test_function = test_function
        self.kwargs = kwargs


def test_angular_distributions(model_file, distribution_test_tuples,
                               number_of_events=20000, make_plots=False):
    plot_data = generate_data_samples(model_file, number_of_events)
    compare_data_samples_and_theory(plot_data, distribution_test_tuples,
                                    make_plots)


def generate_data_samples(model_filename, number_of_events):
    ParticleList = ui.read_particles(model_filename)

    kin = ui.create_helicity_kinematics(model_filename, ParticleList)

    gen = ui.EvtGenGenerator(
        kin.get_particle_state_transition_kinematics_info())

    rand_gen = ui.StdUniformRealGenerator(123)

    # Generate phase space sample
    phsp_sample = ui.generate_phsp(number_of_events, gen, rand_gen)

    intens = ui.create_intensity(
        model_filename, ParticleList, kin, phsp_sample)

    # Generate Data
    sample = ui.generate(number_of_events, kin, gen, intens, rand_gen)

    # Plotting
    kin.create_all_subsystems()
    # create dataset
    dataset = ui.convert_events_to_dataset(sample, kin)

    # use the direct data point access
    from pycompwa.plotting import (PlotData, create_nprecord)
    var_names, dataarray = ui.create_data_array(dataset)
    data_record = create_nprecord(var_names, dataarray)

    return PlotData(data_record=data_record)


def compare_data_samples_and_theory(plot_data,
                                    distribution_test_tuples,
                                    make_plots):
    from pycompwa.plotting import (
        make_binned_distributions, plot_distributions_1d,
        function_to_histogram, scale_to_other_histogram,
        create_axis_title, plot_histogram_difference_2d
    )

    for x in distribution_test_tuples:
        binned_dists = make_binned_distributions(
            plot_data, x.variable_names, **x.kwargs)
        for dists in binned_dists:
            data_hist = dists['data']
            if make_plots and len(data_hist.dimensions) == 1:
                function_hist = function_to_histogram(x.required_distribution,
                                                      data_hist)
                function_hist = scale_to_other_histogram(
                    function_hist, data_hist)
                function_hist.mpl_kwargs = {'fmt': '-'}

                hist_bundle = {'data': data_hist,
                               'theory': function_hist
                               }
                xtitle = create_axis_title(data_hist.dimensions[0], plot_data)
                plot_distributions_1d(hist_bundle, xtitle=xtitle)

            if make_plots and len(data_hist.dimensions) == 2:
                function_hist = function_to_histogram(x.required_distribution,
                                                      data_hist)
                function_hist = scale_to_other_histogram(
                    function_hist, data_hist)
                function_hist.mpl_kwargs = {'fmt': '-'}

                hist_bundle = {'data': data_hist,
                               'theory': function_hist
                               }
                xtitle = create_axis_title(data_hist.dimensions[0], plot_data)
                ytitle = create_axis_title(data_hist.dimensions[1], plot_data)

                plot_histogram_difference_2d(
                    hist_bundle, xtitle=xtitle, ytitle=ytitle)

            x.test_function(data_hist, x.required_distribution)
            #assert(abs(expected - value) < error)
