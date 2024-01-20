import pytest

from ampform.kinematics.phasespace import is_within_phasespace


@pytest.mark.parametrize(
    ("s1", "s2", "expected"),
    [
        (0.0, 3.0, 0),
        (1.0, 1.0, 1),
        (2.0, 2.0, 1),
    ],
)
def test_is_within_phasespace(s1, s2, expected):
    # See widget https://compwa.github.io/report/017
    m0 = 2.1
    m1 = 0.2
    m2 = 0.4
    m3 = 0.4
    computed = is_within_phasespace(s1, s2, m0, m1, m2, m3, outside_value=0)
    assert computed.doit() == expected
