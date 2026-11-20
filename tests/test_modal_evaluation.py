import numpy as np


def test_shared_modal_evaluation_broadcasts_and_preserves_derivatives():
    from thermal_opd import modal_evaluation
    c,roots = {0:np.array([1.]),2:np.array([2.])},{0:np.array([80.]),2:np.array([300.])}
    r,phi,z = np.linspace(0,.01,5)[:,None,None],np.arange(8)[None,:,None]*np.pi/4,np.array([0,.002,.005])[None,None,:]
    value = lambda m,n,z:np.exp(-roots[m][n]*z)
    response = lambda m,n,z:(value(m,n,z),-roots[m][n]*value(m,n,z))
    actual = modal_evaluation.modal_temperature(r,phi,z,c,roots,value)
    dr,dz = modal_evaluation.modal_temperature_gradient(r,phi,z,c,roots,response)
    assert actual.shape==dr.shape==dz.shape==(5,8,3)
    eps = 1e-7
    expected = (modal_evaluation.modal_temperature(r,phi,z+eps,c,roots,value)-
                modal_evaluation.modal_temperature(r,phi,z-eps,c,roots,value))/(2*eps)
    np.testing.assert_allclose(dz,expected,rtol=2e-8,atol=1e-8)


def test_temperature_sum_groups_radial_terms_by_azimuthal_order(monkeypatch):
    from thermal_opd import modal_evaluation as modal
    # Known exact sum: (.5+.5) + (1e16-1e16) = 1. Flattening all
    # radial/azimuthal terms loses the initial unit before cancellation.
    monkeypatch.setattr(modal,'jv',lambda m,lam_r:np.ones_like(lam_r,dtype=float))
    coefficients={0:np.zeros(2),2:np.zeros(2)}
    roots={m:np.ones(2) for m in coefficients}
    values={(0,0):.5,(0,1):.5,(2,0):1e16,(2,1):-1e16}
    response=lambda m,n,z:values[m,n]+np.zeros_like(z)
    r,phi,z=np.ones((3,1)),np.zeros((1,4)),np.array(0.)
    legacy=modal.modal_temperature(r,phi,z,coefficients,roots,response)
    np.testing.assert_array_equal(legacy,np.zeros((3,4)))  # Historical path unchanged.
    actual=modal.modal_temperature(r,phi,z,coefficients,roots,response,group_by_order=True)
    np.testing.assert_array_equal(actual,np.ones((3,4)))
    orders=np.array([modal.modal_temperature(r,phi,z,{m:coefficients[m]},{m:roots[m]},response)
                     for m in coefficients])
    np.testing.assert_array_equal(actual,orders.sum(axis=0))


def test_temperature_empty_orders_preserve_broadcast_shape():
    from thermal_opd.modal_evaluation import modal_temperature
    actual=modal_temperature(np.zeros((2,1)),np.zeros((1,3)),0.,{}, {},lambda *a:0.)
    np.testing.assert_array_equal(actual,np.zeros((2,3)))
