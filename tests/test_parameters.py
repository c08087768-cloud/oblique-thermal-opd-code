import numpy as np
from scipy.constants import Stefan_Boltzmann

from thermal_opd.parameters import PhysicalParameters


def test_effective_heat_transfer_matches_small_rise_heat_loss():
    p = PhysicalParameters()
    delta = 1e-3
    heat_loss = (
        p.h_conv * delta
        + p.emissivity * Stefan_Boltzmann
        * ((p.ambient_temperature + delta)**4 - p.ambient_temperature**4)
    )

    assert np.isclose(p.h_eff, heat_loss / delta, rtol=1e-5)
