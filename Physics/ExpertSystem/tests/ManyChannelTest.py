import logging

from expertsystem.ui.system_control import (StateTransitionManager)

#logging.basicConfig(level=logging.INFO)

# define all of the different decay scenarios
cases = [
    (['p', 'pbar'], ['pi+', 'pi0']),
    (['eta'], ['gamma', 'gamma']),
    (['sigma0'], ['lambda', 'pi0']),
    (['sigma-'], ['n', 'pi-']),
    (['e+', 'e-'], ['mu+', 'mu-']),
    (['mu-'], ['e-', 'vebar']),
    (['mu-'], ['e-', 've']),  # this is just an additional lepton number test
    (['Delta(1232)+'], ['p', 'pi0']),
    (['vebar', 'p'], ['n', 'e+']),
    (['e-', 'p'], ['ve', 'pi0']),
    (['p', 'p'], ['sigma+', 'n', 'K_S0', 'pi+', 'pi0']),
    (['p'], ['e+', 'gamma']),
    (['p', 'p'], ['p', 'p', 'p', 'pbar']),
    (['n', 'nbar'], ['pi+', 'pi-', 'pi0']),
    (['pi+', 'n'], ['pi-', 'p']),
    (['K-'], ['pi-', 'pi0']),
    (['sigma+', 'n'], ['sigma-', 'p']),
    (['sigma0'], ['lambda', 'gamma']),
    (['Xi-'], ['lambda', 'pi-']),
    (['Xi0'], ['p', 'pi-']),
    (['pi-', 'p'], ['lambda', 'K_S0']),
    (['pi0'], ['gamma', 'gamma']),
    (['sigma-'], ['n', 'e-', 'vebar']),
    (['rho(770)0'], ['pi0', 'pi0']),
    (['rho(770)0'], ['gamma', 'gamma']),
    (['J/psi'], ['pi0', 'eta']),
    (['J/psi'], ['rho(770)0', 'rho(770)0']),
    (['K_S0'], ['pi+', 'pi-', 'pi0'])
]

for case in cases:
    print("processing case:" + str(case))

    tbd_manager = StateTransitionManager(case[0], case[1], [],
                                         'canonical', 'nbody')

    graph_interaction_settings = tbd_manager.prepare_graphs()
    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings)

    if len(solutions) > 0:
        print("is valid")
    else:
        print("not allowed! violates: " + str(violated_rules))
