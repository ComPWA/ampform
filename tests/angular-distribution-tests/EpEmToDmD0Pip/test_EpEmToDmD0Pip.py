#!/usr/bin/env python3

import logging
from math import cos

logging.basicConfig(level=logging.INFO)


def test_angular_distributions(make_plots=False):
    from pycompwa.plotting import (
        chisquare_test
    )
    import os
    thisdirectory = os.path.dirname(os.path.realpath(__file__))
    import sys
    sys.path.append(thisdirectory + "/..")
    from distributioncomparison import (
        test_angular_distributions, ComparisonTuple)

    # x = cos(theta) distribution from EpEm decay should be 1 + x^2
    # x = cos(theta') distribution from D2* decay should be 1 - (2*x^2 - 1)^2
    # phi distribution of the EpEm decay should be constant
    # phi' distribution of the D2* decay should be 2 + cos(2phi)

    tuples = [
        ComparisonTuple('theta_34_2', lambda x: 1+x*x,
                        chisquare_test, **{'number_of_bins': 80}),
        ComparisonTuple('theta_3_4_vs_2', lambda x: 1-(2*x*x-1)**2,
                        chisquare_test, **{'number_of_bins': 80}),
        ComparisonTuple('phi_34_2', lambda x: 1,
                        chisquare_test, **{'number_of_bins': 80}),
        ComparisonTuple(['theta_3_4_vs_2', 'phi_3_4_vs_2'],
                        lambda x, y: (1-x**2)*(x**2)*(2+cos(2*y)),
                        chisquare_test, **{'number_of_bins': 25}),
        ComparisonTuple('phi_3_4_vs_2', lambda x: 2+cos(2*x),
                        chisquare_test, **{'number_of_bins': 80}),

    ]
    test_angular_distributions(
        thisdirectory+"/model.xml", tuples, 20000, make_plots=make_plots)


if __name__ == '__main__':
    test_angular_distributions(make_plots=True)
