"""Public model lifecycle and equivalence to the established numerical API."""
from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.model import ThermalModel, ModalResolution
from thermal_opd.transient import PowerHistory, TransientSolution
from thermal_opd.steady import steady_temperature, steady_temperature_gradient
from thermal_opd.opd import steady_refracted_opd


def test_model_preparation_is_explicit_and_shared_without_transient_work(monkeypatch):
    import thermal_opd.model as module
    import thermal_opd.transient as transient
    original = module.solve_modal_source
    calls = []
    def prepare(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(module,'solve_modal_source',prepare)
    def no_axial(*args,**kwargs):
        pytest.fail('Steady preparation must not prepare transient axial modes')
    monkeypatch.setattr(transient,'AxialModes',no_axial)
    model = ThermalModel(PhysicalParameters(),angle_deg=60.,resolution=ModalResolution(2,3,80,16,16))
    assert not calls
    with pytest.raises(FrozenInstanceError):
        model.angle_deg = 30.
    prepared = model.prepare()
    assert len(calls)==1
    steady = prepared.solve_steady()
    steady.temperature(.001,.2,.002)
    assert len(calls)==1
    with pytest.raises(TypeError):
        prepared.coefficients[0] = np.ones(3)
    with pytest.raises(ValueError):
        prepared.coefficients[0][0] = 0.


@pytest.mark.parametrize('angle',[0.,60.,-30.])
def test_solutions_reuse_prepared_data_and_preserve_temperature_gradient_and_optics(angle):
    p = PhysicalParameters()
    s = ModalResolution(2,3,80,16,24)
    prepared = ThermalModel(p,angle_deg=angle,resolution=s).prepare()
    steady = prepared.solve_steady()
    transient = prepared.solve_transient(num_axial_modes=24,history=PowerHistory('linear',5.))
    assert isinstance(transient,TransientSolution)
    assert steady.prepared is transient.prepared is prepared
    args = (prepared.coefficients,prepared.roots,p.conductivity,p.thickness,p.h_eff,p.h_eff)
    r,phi,z = np.array([.001,.003]),np.array([.2,.7]),np.array([.001,.003])
    np.testing.assert_array_equal(steady.temperature(r,phi,z),steady_temperature(r,phi,z,*args))
    np.testing.assert_array_equal(steady.temperature_gradient(r,phi,z),steady_temperature_gradient(r,phi,z,*args))
    eps=1e-7
    dr,dz=transient.temperature_gradient(r,phi,z,.625)
    np.testing.assert_allclose(dr,(transient.temperature(r+eps,phi,z,.625)-transient.temperature(r-eps,phi,z,.625))/(2*eps),rtol=2e-7,atol=1e-7)
    np.testing.assert_allclose(dz,(transient.temperature(r,phi,z+eps,.625)-transient.temperature(r,phi,z-eps,.625))/(2*eps),rtol=2e-7,atol=1e-7)
    kw = dict(radius=p.radius,refraction_angle=prepared.refraction_angle,num_ray_points=s.ray_points,point_chunk_size=512)
    x,y = np.array([0.,.002]),np.array([0.,-.003])
    np.testing.assert_array_equal(steady.opd(x,y),steady_refracted_opd(x,y,*args,p.dn_dT,**kw))
    np.testing.assert_allclose(transient.opd_series(x,y,[0.,.625,2.5]),
                               np.stack([transient.opd(x,y,t) for t in [0.,.625,2.5]]),rtol=1e-13,atol=1e-20)
    np.testing.assert_array_equal(transient.prepare_opd(.625)(x,y),transient.opd(x,y,.625))
    with pytest.raises(ValueError,match='rear|cylinder'):
        steady.opd(p.radius*2,0.)


@pytest.mark.parametrize('angle',[90.,-90.,np.nan,np.inf])
def test_model_rejects_invalid_angles_without_precomputing(angle):
    with pytest.raises(ValueError):
        ThermalModel(PhysicalParameters(),angle_deg=angle,resolution=ModalResolution(2,3,80,16,16))


def test_prepared_model_copies_external_arrays():
    from thermal_opd.model import PreparedModel
    p = PhysicalParameters()
    c,roots = {0:np.array([1000.])},{0:np.array([80.])}
    prepared = PreparedModel(p,0.,ModalResolution(0,1,80,16,16),c,roots)
    c[0][0] = 0.
    roots[0][0] = 0.
    assert prepared.coefficients[0][0]==1000.
    assert prepared.roots[0][0]==80.


@pytest.mark.parametrize('angle', [0., 30., 60.])
def test_modal_preparation_rejects_unimplemented_insulated_zero_mode(angle):
    p = replace(PhysicalParameters(), h_conv=0., emissivity=0.)
    model = ThermalModel(p, angle, ModalResolution(2, 3, 80, 16, 16))
    with pytest.raises(ValueError, match='zero mode'):
        model.prepare()
