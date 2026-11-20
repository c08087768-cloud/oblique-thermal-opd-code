import numpy as np
import pytest
from scipy.special import jv
from thermal_opd.eigenmodes import bessel_j


@pytest.mark.parametrize('order',[0,2,4,6,8,12,16])
def test_integer_bessel_matches_independent_scipy_at_axis_transition_and_high_roots(order):
    # An unstable recurrence near the axis or a wrong order corrupts the ray
    # field. Absolute accuracy is used at Bessel zeros, where relative errors
    # are undefined.
    threshold = max(8.,2.*order)
    x = np.unique(np.r_[0.,1e-12,np.geomspace(1e-8,1e3,300),
                        np.linspace(threshold*.99,threshold*1.01,51),-2.])
    np.testing.assert_allclose(bessel_j(order,x),jv(order,x),rtol=1e-11,atol=3e-14)
    assert np.shape(bessel_j(order,0.))==()
