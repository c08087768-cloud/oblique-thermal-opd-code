"""Source/known-truth checks frozen before the new manuscript calculations."""
import numpy as np
import pytest
from thermal_opd.materials import corning_7980_reference
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.metrics import azimuthal_omission_metrics


@pytest.mark.parametrize('lam,t,expected',[(.8,295,1.45373),(1.,300,1.45086),(1.5,200,1.44433)])
def test_source_table4_indices(lam,t,expected):
    assert abs(corning_7980_reference(lam*1e-6,t)[0]-expected) < 5e-6


def test_absolute_derivative_at_reference_against_one_sided_difference():
    # The independent finite-difference arithmetic remains inside the fit domain.
    t, h = 300., .02
    values=[corning_7980_reference(1064e-9,t-i*h)[0] for i in range(5)]
    slope=(25*values[0]-48*values[1]+36*values[2]-16*values[3]+3*values[4])/(12*h)
    n,dn=corning_7980_reference(1064e-9,t)
    assert slope == pytest.approx(dn,abs=2e-13)
    assert n == pytest.approx(1.450076167940166,abs=2e-15)
    p=PhysicalParameters()
    assert (p.substrate_refractive_index,p.dn_dT)==(n,dn)
    assert p.wavelength == 1064e-9
    assert p.ambient_temperature == 300.
    assert p.dn_dT == pytest.approx(8.415426824978581e-6, rel=0., abs=2e-13)
    with pytest.raises(ValueError):corning_7980_reference(1064e-9,301)


def test_planted_omission_piston_scaling_and_null():
    # Full=[0,2,4], m0=[0,1,2] gives delta=[-1,0,1] after piston.
    full=np.array([0.,2.,4.]);zero=np.array([0.,1.,2.]);mask=np.ones(3,dtype=bool)
    expected=np.sqrt(2/3)/4
    for f,z in [(full,zero),(full+10,zero-7),(3*full,3*zero)]:
        result=azimuthal_omission_metrics(f,z,mask)
        assert result['rms_over_full_pv']==pytest.approx(expected,abs=1e-15)
        assert result['linf_over_full_pv']==pytest.approx(.25,abs=1e-15)
    assert azimuthal_omission_metrics(full,full,mask)['rms_over_full_pv']==0
    # Deliberately corrupt a selected value/shape; both must be caught.
    with pytest.raises(ValueError):azimuthal_omission_metrics(full,[0,np.nan,2],mask)
    with pytest.raises(ValueError):azimuthal_omission_metrics(full,zero,mask[:2])
    with pytest.raises(ValueError):azimuthal_omission_metrics(np.zeros(3),np.zeros(3),mask)
    corrupt=zero.copy();corrupt[1]+=1
    assert not np.isclose(azimuthal_omission_metrics(full,corrupt,mask)['rms_over_full_pv'],expected)


def test_cross_method_optical_ratios_are_invariant_under_positive_common_scale():
    # This identity justifies inheriting the theta=0 optical comparisons when
    # the only operative optical-input change is a positive common gamma.
    from thermal_opd.convergence import field_errors
    temperature=np.array([[1.,2.,3.],[2.,4.,6.]])
    sa=np.array([0.,2.,4.])
    fe=np.array([.3,2.1,3.9])
    def measured(scale):
        return field_errors(temperature,temperature,np.ones(3),scale*fe,scale*sa)
    baseline=measured(1.)
    for scale in (.8766,3.,1e-7):
        values=measured(scale)
        for key in ('opd_relative_pv','opd_piston_removed_rms_over_reference_pv',
                    'opd_piston_removed_linf_over_reference_pv'):
            assert values[key]==pytest.approx(baseline[key],abs=2e-15)
