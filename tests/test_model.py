import numpy as np

from thermal_opd.model import solve_modal_source, ModalResolution
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.steady import steady_temperature


def test_normal_incidence_does_not_project_roundoff_into_nonaxisymmetric_modes():
    p = PhysicalParameters()
    coefficients, roots = solve_modal_source(p, 0., ModalResolution(6, 8, 200, 32, 16))
    assert set(coefficients) == {0}
    actual = steady_temperature(p.radius/3, np.array([0., 0.73, 2.17]), p.thickness/3,
                                coefficients, roots, p.conductivity, p.thickness, p.h_eff, p.h_eff)
    np.testing.assert_array_equal(actual, np.full(3, actual[0]))
