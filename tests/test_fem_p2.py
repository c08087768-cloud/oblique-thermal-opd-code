"""Production P2 controls: polynomial fields, independent slab/FV, mirror guards."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from thermal_opd import fem
from thermal_opd.transient import PowerHistory


ARGS=(1.,.5,1.4,2.,4.,0.,lambda x,y:6.+0*x)
KW=dict(disk_refinements=1,axial_layers=4,quadrature_order=6)


def test_p2_steady_reproduces_independent_robin_slab_and_real_dof_count():
    result=fem.solve_steady_fem(*ARGS,**KW,solver='direct',rtol=1e-12,field_degree=2)
    resistance=1/4.+.5/1.4
    q=6/(1+2*resistance)
    expected=q*(1/4.+(.5-result.z_m)/1.4)
    np.testing.assert_allclose(result.temperature_K,expected[:,None]+np.zeros_like(result.temperature_K),rtol=3e-12)
    assert result.field_degree==2 and result.z_m.size==9
    assert result.temperature_K.shape[1]>result.disk_mesh.nvertices
    assert result.diagnostics['relative_energy_imbalance']<1e-11


def test_p2_interpolation_and_refracted_ray_integrate_known_quartic_exactly():
    from numpy.polynomial import Polynomial as P
    from skfem import Basis,ElementTriP2
    result=fem.solve_steady_fem(*ARGS,**KW,solver='direct',rtol=1e-12,field_degree=2)
    xy=Basis(result.disk_mesh,ElementTriP2()).doflocs;z=result.z_m[:,None]
    result.temperature_K=(1+xy[0]**2+2*xy[1]**2)[None,:]*(1+z+3*z*z)
    x=np.array([.1,-.15]);y=np.array([.15,.2]);zz=np.array([.035,.41])
    np.testing.assert_allclose(result.temperature_at(x,y,zz),(1+x*x+2*y*y)*(1+zz+3*zz*zz),rtol=3e-14)
    angle=.25;expected=[]
    for xx,yy in zip(x,y):
        poly=(P([1+xx*xx])+2*P([yy,np.tan(angle)])**2)*P([1,1,3])
        anti=poly.integ();expected.append(2*(anti(.5)-anti(0))/np.cos(angle))
    np.testing.assert_allclose(result.refracted_opd(x,y,angle,2.,point_chunk_size=1),expected,rtol=4e-14)


@pytest.mark.parametrize('integrator',['backward_euler','bdf2'])
def test_p2_mirror_reconstruction_matches_full_space_and_all_step_diagnostics(integrator):
    kw=dict(**KW,density=3.,specific_heat=4.,times=[0.,.001,.0025,.02],time_step=.001,
        history=PowerHistory(),solver='cg',rtol=1e-12,field_degree=2,
        linear_preconditioner='axial_lines',time_integrator=integrator)
    a=fem.solve_transient_fem(*ARGS,**kw)
    b=fem.solve_transient_fem(*ARGS,**kw,mirror_reduction=True)
    for aa,bb in zip(a.frames,b.frames):
        assert aa.field_degree==bb.field_degree==2
        np.testing.assert_allclose(aa.temperature_K,bb.temperature_K,rtol=2e-9,atol=2e-13)
    for aa,bb in zip(a.steps,b.steps):
        assert bb['relative_linear_residual']<=1e-12
        for key in ('stored_energy_J','heat_loss_W'):
            np.testing.assert_allclose(aa[key],bb[key],rtol=2e-9,atol=1e-12)
    assert b.diagnostics['mirror_reduction']['full_space_physical_diagnostics'] is True


def test_p2_against_independent_finite_volume_time_reference():
    from thermal_opd.transient_validation import transient_uniform_flux_1d_reference
    times=np.array([0.,.05,.15,.4]);depths=np.array([0.,.25,.5])
    z,T=transient_uniform_flux_1d_reference(.5,1.4,12.,6.,2.,4.,times,num_intervals=1024)
    reference=np.stack([np.interp(depths,z,t) for t in T])
    result=fem.solve_transient_fem(*ARGS,**{**KW,'axial_layers':16},density=3.,specific_heat=4.,
        times=times,time_step=.0005,history=PowerHistory(),solver='cg',rtol=1e-12,
        field_degree=2,mirror_reduction=True,linear_preconditioner='axial_lines',time_integrator='bdf2')
    actual=np.array([f.temperature_at(.1,.2,depths) for f in result.frames])
    assert np.max(abs(actual-reference))/np.max(reference)<1e-4
    np.testing.assert_array_equal(result.frames[0].temperature_K,0.)


def test_asymmetric_load_cannot_be_silently_mirror_projected():
    with pytest.raises(ValueError,match='mirror|symmetr'):
        fem.solve_transient_fem(*ARGS[:-1],lambda x,y:6.+x,**KW,density=3.,specific_heat=4.,
            times=[0.,.001],time_step=.001,history=PowerHistory(),solver='cg',rtol=1e-12,
            field_degree=2,mirror_reduction=True)


def test_p2_line_preconditioner_matches_dense_vertical_blocks():
    system=fem.assemble_thermal_fem(*ARGS,**KW,field_degree=2)
    matrix=30*system.mass+system.matrix;n=system.Mxy.shape[0]
    rhs=np.random.default_rng(5).normal(size=matrix.shape[0])
    actual=fem._axial_line_preconditioner(matrix,n,field_degree=2)@rhs
    expected=np.zeros_like(rhs)
    for j in range(n):
        ids=np.arange(j,matrix.shape[0],n)
        expected[ids]=np.linalg.solve(matrix[ids][:,ids].toarray(),rhs[ids])
    np.testing.assert_allclose(actual,expected,rtol=4e-13,atol=4e-13)


@pytest.mark.parametrize('order',[2,3])
def test_p2_rejects_underintegrated_consistent_mass_and_robin(order):
    with pytest.raises(ValueError,match='quadrature'):
        fem.assemble_thermal_fem(*ARGS,**{**KW,'quadrature_order':order},field_degree=2)


@pytest.mark.parametrize('method',['backward_euler','bdf2'])
def test_observation_writer_does_not_change_any_node_or_time_step(method):
    kw=dict(**KW,density=3.,specific_heat=4.,times=[0.,.001,.0025],time_step=.0005,
        history=PowerHistory(),solver='cg',rtol=1e-12,field_degree=2,
        linear_preconditioner='axial_lines',time_integrator=method,mirror_reduction=True)
    expected=fem.solve_transient_fem(*ARGS,**kw);observations=[]
    actual=fem.solve_transient_fem(*ARGS,**kw,observation_callback=lambda f:observations.append(f.temperature_K.copy()))
    assert len(observations)==len(actual.frames)==3
    for a,b,saved in zip(actual.frames,expected.frames,observations):
        np.testing.assert_array_equal(a.temperature_K,b.temperature_K)
        np.testing.assert_array_equal(a.temperature_K,saved)
    assert actual.steps==expected.steps
def test_mirror_cg_corrects_recursive_residual_drift_without_relaxing_tolerance():
    from scipy.sparse.linalg import cg
    from thermal_opd import fem
    system=fem.assemble_thermal_fem(1.,.5,1.4,2.,2.,1.,lambda x,y:1.+x*x+y*y,
        disk_refinements=1,axial_layers=4,field_degree=2,quadrature_order=6)
    q,_=fem._mirror_basis(system)
    calls=[];observed=[]
    def drifted(matrix,rhs,**kwargs):
        x,info=cg(matrix,rhs,**kwargs);calls.append(kwargs)
        if len(calls)==1:x=x*(1.+1e-7)
        return x,info
    solution,info=fem._mirror_cg(system,q,drifted)(system.matrix,system.rhs,rtol=1e-12,atol=0.,maxiter=10000,
        callback=lambda value:observed.append(value.copy()))
    assert info==0 and len(calls)>1
    assert np.linalg.norm(system.matrix@solution-system.rhs)<=1e-12*np.linalg.norm(system.rhs)
    np.testing.assert_allclose(observed[-1],solution,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('corrupt',[0.,np.nan])
def test_mirror_cg_never_certifies_a_bad_inner_solver(corrupt):
    system=fem.assemble_thermal_fem(*ARGS,**KW,field_degree=2)
    q,_=fem._mirror_basis(system)
    def invalid(matrix,rhs,**kwargs):return np.full(rhs.shape,corrupt),0
    with pytest.raises(RuntimeError,match='residual|non-finite'):
        fem._mirror_cg(system,q,invalid)(system.matrix,system.rhs,rtol=1e-12,atol=0.,maxiter=100)


@pytest.mark.skipif(np.finfo(np.longdouble).eps>=np.finfo(float).eps,reason='Platform has no wider floating accumulator')
def test_same_float64_nodes_get_accurate_residual_without_relaxing_tolerance(monkeypatch):
    from scipy.sparse import csr_matrix,eye
    from types import SimpleNamespace
    # This isolated acceptance control supplies a known exact solution; it does
    # not claim that this nonsymmetric planted matrix is a thermal FEM matrix.
    A=csr_matrix([[1e16,1.,-1e16],[0.,1.,0.],[0.,0.,1.]])
    exact=np.ones(3);b=np.ones(3)
    assert np.linalg.norm(A@exact-b)>1e-12*np.linalg.norm(b)
    system=SimpleNamespace(z_m=np.array([0.]),field_degree=2,diagnostics={})
    monkeypatch.setattr(fem,'_axial_line_preconditioner',lambda *a,**k:None)
    calls=[]
    def known_solution(matrix,rhs,**kwargs):
        calls.append(1);return (exact.copy() if len(calls)==1 else np.zeros(3)),0
    result,code=fem._mirror_cg(system,eye(3,format='csr'),known_solution)(A,b,rtol=1e-12,atol=0.)
    np.testing.assert_array_equal(result,exact);assert result.dtype==np.float64 and code==0
    events=system.diagnostics['linear_residual_arithmetic']['events']
    assert len(calls)==1 and events[0]['passed'] and events[0]['residual_norm_extended']==0.
    assert events[0]['linear_solve_index']==1 and events[0]['residual_norm_float64']>events[0]['original_tolerance']
    assert events[0]['mantissa_bits']>np.finfo(float).nmant+1
    calls.clear();b[0]+=.01
    with pytest.raises(RuntimeError,match='original full-space'):
        fem._mirror_cg(system,eye(3,format='csr'),known_solution)(A,b,rtol=1e-12,atol=0.)


def test_extended_residual_check_never_claims_wider_precision_when_unavailable(monkeypatch):
    from scipy.sparse import eye
    monkeypatch.setattr(fem,'_wider_residual_dtype',lambda:None)
    assert fem._extended_residual_check(eye(2,format='csr'),np.ones(2),np.ones(2),1e-12) is None


@pytest.mark.skipif(np.finfo(np.longdouble).eps>=np.finfo(float).eps,reason='Platform has no wider floating accumulator')
def test_residual_correction_uses_the_accurate_defect_not_float64_cancellation(monkeypatch):
    from scipy.sparse import csr_matrix,eye
    from types import SimpleNamespace
    # A known-answer arithmetic control, not a thermal matrix or CG oracle.
    A=csr_matrix([[1e16,1.,-1e16],[0.,1.,0.],[0.,0.,1.]])
    # 1/8 remains exact even in the 64-bit-mantissa Linux accumulator at 1e16.
    b=np.ones(3);initial=np.array([1.,1.125,1.]);calls=[]
    system=SimpleNamespace(z_m=np.array([0.]),field_degree=2,diagnostics={})
    monkeypatch.setattr(fem,'_axial_line_preconditioner',lambda *a,**k:None)
    def controlled_inner(matrix,rhs,**kw):
        calls.append(rhs.copy())
        return (initial.copy() if len(calls)==1 else np.linalg.solve(matrix.toarray(),rhs)),0
    result,code=fem._mirror_cg(system,eye(3,format='csr'),controlled_inner)(A,b,rtol=1e-12,atol=0.)
    assert code==0 and len(calls)==2
    # Exact cancellation of the two 1e16 terms leaves x[1]; no wide oracle needed.
    expected=np.array([1.-initial[1],1.-initial[1],0.])
    np.testing.assert_array_equal(calls[1],expected)
    np.testing.assert_array_equal(result,np.ones(3))
    assert result.dtype==np.float64
    events=system.diagnostics['linear_residual_arithmetic']['events']
    assert events[0]['correction_defect_arithmetic']=='extended_accumulation_then_float64'


def test_mirror_raw_pass_never_enters_extended_path_and_bad_solver_is_bounded(monkeypatch):
    from scipy.sparse import eye
    from types import SimpleNamespace
    A=eye(3,format='csr');b=np.ones(3)
    system=SimpleNamespace(z_m=np.array([0.]),field_degree=2,diagnostics={})
    monkeypatch.setattr(fem,'_axial_line_preconditioner',lambda *a,**k:None)
    def forbidden(*a,**kw):raise AssertionError('Raw acceptance must not call extended arithmetic')
    monkeypatch.setattr(fem,'_extended_residual_check',forbidden)
    result,code=fem._mirror_cg(system,A,lambda *a,**kw:(b.copy(),0))(A,b,rtol=1e-12)
    np.testing.assert_array_equal(result,b);assert code==0 and not system.diagnostics
    monkeypatch.setattr(fem,'_extended_residual_check',lambda *a,**kw:None)
    calls=[]
    def wrong(*a,**kw):calls.append(1);return np.zeros(3),0
    with pytest.raises(RuntimeError,match='original full-space'):
        fem._mirror_cg(system,A,wrong)(A,b,rtol=1e-12)
    assert len(calls)==4  # One initial solve plus at most three corrections.
