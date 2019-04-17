from pycompwa.expertsystem.ui.system_control import (
    StateTransitionManager, InteractionTypes)

# import logging
# logging.basicConfig(level=logging.INFO)

import pytest


@pytest.mark.parametrize("test_input,expected", [
    ((['p', 'pbar'], ['pi+', 'pi0']), ['ChargeConservation']),
    ((['eta'], ['gamma', 'gamma']), []),
    ((['sigma0'], ['lambda', 'pi0']), ['MassConservation']),
    ((['sigma-'], ['n', 'pi-']), []),
    ((['e+', 'e-'], ['mu+', 'mu-']), []),
    ((['mu-'], ['e-', 'vebar']), ['MuonLNConservation', 'SpinConservation']),
    # this is just an additional lepton number test
    ((['mu-'], ['e-', 've']), ['ElectronLNConservation',
                               'MuonLNConservation',
                               'SpinConservation']),
    ((['Delta(1232)+'], ['p', 'pi0']), []),
    ((['vebar', 'p'], ['n', 'e+']), []),
    ((['e-', 'p'], ['ve', 'pi0']), ['BaryonNumberConservation',
                                    'SpinConservation']),
    ((['p', 'p'], ['sigma+', 'n', 'K_S0', 'pi+', 'pi0']), []),
    ((['p'], ['e+', 'gamma']), ['ElectronLNConservation',
                                'BaryonNumberConservation']),
    ((['p', 'p'], ['p', 'p', 'p', 'pbar']), []),
    ((['n', 'nbar'], ['pi+', 'pi-', 'pi0']), []),
    ((['pi+', 'n'], ['pi-', 'p']), ['ChargeConservation']),
    ((['K-'], ['pi-', 'pi0']), []),
    ((['sigma+', 'n'], ['sigma-', 'p']), ['ChargeConservation']),
    ((['sigma0'], ['lambda', 'gamma']), []),
    ((['Xi-'], ['lambda', 'pi-']), []),
    ((['Xi0'], ['p', 'pi-']), []),
    ((['pi-', 'p'], ['lambda', 'K_S0']), []),
    ((['pi0'], ['gamma', 'gamma']), []),
    ((['pi0'], ['gamma', 'gamma', 'gamma']), []),
    ((['sigma-'], ['n', 'e-', 'vebar']), []),
    ((['rho(770)0'], ['pi0', 'pi0']), ['IdenticalParticleSymmetrization']),
    ((['rho(770)0'], ['gamma', 'gamma']), []),
    ((['J/psi'], ['pi0', 'eta']), []),
    ((['J/psi'], ['rho(770)0', 'rho(770)0']), []),
    ((['K_S0'], ['pi+', 'pi-', 'pi0']), []),
])
def test_general_reaction(test_input, expected):
    # define all of the different decay scenarios
    print("processing case:" + str(test_input))

    tbd_manager = StateTransitionManager(test_input[0], test_input[1],
                                         formalism_type='canonical',
                                         topology_building='nbody',
                                         propagation_mode='full')

    graph_interaction_settings = tbd_manager.prepare_graphs()
    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings)

    if len(solutions) > 0:
        print("is valid")
        assert len(expected) == 0
    else:
        print("not allowed! violates: " + str(violated_rules))
        assert set(violated_rules) == set(expected)


@pytest.mark.parametrize("test_input,expected", [
    ((['f0(980)'], ['pi+', 'pi-']), []),
    ((['J/psi'], ['pi0', 'f0(980)']), ['CParityConservation',
                                       'SpinConservation']),
    ((['pi0'], ['gamma', 'gamma']), []),
    ((['pi0'], ['gamma', 'gamma', 'gamma']), [
     'CParityConservation', 'SpinConservation']),
    ((['pi0'], ['e+', 'e-', 'gamma']), []),
    ((['pi0'], ['e+', 'e-']), [])
])
def test_em_reactions(test_input, expected):
    # general checks
    print("processing case:" + str(test_input))

    tbd_manager = StateTransitionManager(test_input[0], test_input[1],
                                         formalism_type='canonical',
                                         topology_building='nbody',
                                         propagation_mode='full')

    tbd_manager.set_allowed_interaction_types([InteractionTypes.EM])

    graph_interaction_settings = tbd_manager.prepare_graphs()
    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings)

    assert set(violated_rules) == set(expected)


@pytest.mark.parametrize("test_input,expected", [
    ((['f0(980)'], ['pi+', 'pi-']), []),
    ((['J/psi'], ['pi0', 'f0(980)']), ['IsoSpinConservation',
                                       'CParityConservation',
                                       'SpinConservation'])
])
def test_strong_reactions(test_input, expected):
    # general checks
    print("processing case:" + str(test_input))

    tbd_manager = StateTransitionManager(test_input[0], test_input[1],
                                         formalism_type='canonical',
                                         topology_building='nbody',
                                         propagation_mode='full')

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])

    graph_interaction_settings = tbd_manager.prepare_graphs()
    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings)

    assert set(violated_rules) == set(expected)
