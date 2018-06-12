import logging

from expertsystem.ui.system_control import (
    StateTransitionManager, InteractionTypes)


# logging.basicConfig(level=logging.INFO)


def test_general_reaction():
    # define all of the different decay scenarios
    cases = [
        (['p', 'pbar'], ['pi+', 'pi0'], ['ChargeConservation']),
        (['eta'], ['gamma', 'gamma'], []),
        (['sigma0'], ['lambda', 'pi0'], ['MassConservation']),
        (['sigma-'], ['n', 'pi-'], []),
        (['e+', 'e-'], ['mu+', 'mu-'], []),
        (['mu-'], ['e-', 'vebar'], ['MuonLNConservation']),
        # this is just an additional lepton number test
        (['mu-'], ['e-', 've'], ['ElectronLNConservation',
                                 'MuonLNConservation']),
        (['Delta(1232)+'], ['p', 'pi0'], []),
        (['vebar', 'p'], ['n', 'e+'], []),
        (['e-', 'p'], ['ve', 'pi0'], ['BaryonNumberConservation']),
        (['p', 'p'], ['sigma+', 'n', 'K_S0', 'pi+', 'pi0'], []),
        (['p'], ['e+', 'gamma'], ['ElectronLNConservation',
                                  'BaryonNumberConservation']),
        (['p', 'p'], ['p', 'p', 'p', 'pbar'], []),
        (['n', 'nbar'], ['pi+', 'pi-', 'pi0'], []),
        (['pi+', 'n'], ['pi-', 'p'], ['ChargeConservation']),
        (['K-'], ['pi-', 'pi0'], []),
        (['sigma+', 'n'], ['sigma-', 'p'], ['ChargeConservation']),
        (['sigma0'], ['lambda', 'gamma'], []),
        (['Xi-'], ['lambda', 'pi-'], []),
        (['Xi0'], ['p', 'pi-'], []),
        (['pi-', 'p'], ['lambda', 'K_S0'], []),
        (['pi0'], ['gamma', 'gamma'], []),
        (['sigma-'], ['n', 'e-', 'vebar'], []),
        (['rho(770)0'], ['pi0', 'pi0'], ['IdenticalParticleSymmetrization']),
        (['rho(770)0'], ['gamma', 'gamma'], []),
        (['J/psi'], ['pi0', 'eta'], []),
        (['J/psi'], ['rho(770)0', 'rho(770)0'], []),
        (['K_S0'], ['pi+', 'pi-', 'pi0'], []),
    ]

    for case in cases:
        print("processing case:" + str(case))

        tbd_manager = StateTransitionManager(case[0], case[1], [], {},
                                             'canonical', 'nbody')

        graph_interaction_settings = tbd_manager.prepare_graphs()
        (solutions, violated_rules) = tbd_manager.find_solutions(
            graph_interaction_settings)

        if len(solutions) > 0:
            print("is valid")
            assert len(case[2]) == 0
        else:
            print("not allowed! violates: " + str(violated_rules))
            assert set(violated_rules) == set(case[2])


def test_strong_reactions():
    em_cases = [
        (['f0(980)'], ['pi+', 'pi-'], []),
        (['J/psi'], ['pi0', 'f0(980)'], ['IsoSpinConservation',
                                         'CParityConservation'])
    ]

    # general checks
    for case in em_cases:
        print("processing case:" + str(case))

        tbd_manager = StateTransitionManager(case[0], case[1], [], {},
                                             'canonical', 'nbody')

        tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])

        graph_interaction_settings = tbd_manager.prepare_graphs()
        (solutions, violated_rules) = tbd_manager.find_solutions(
            graph_interaction_settings)

        if len(solutions) > 0:
            print("is valid")
            assert len(case[2]) == 0
        else:
            print("not allowed! violates: " + str(violated_rules))
            assert set(violated_rules) == set(case[2])
