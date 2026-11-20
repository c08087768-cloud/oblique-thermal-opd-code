"""Mesh/time resolution controls must preserve the existing FEM equations."""
import numpy as np
import pytest
from thermal_opd.fem import assemble_thermal_fem, solve_steady_fem, solve_transient_fem
from thermal_opd.transient import PowerHistory


ARGS = (1., .5, 1.4, 2., 2., 0., lambda x, y: 6.+0*x)
KW = dict(disk_refinements=1, axial_layers=16, quadrature_order=4)


def test_explicit_uniform_mesh_preserves_stiffness_mass_load_and_transient():
    # A changed default assembly or inconsistent explicit z grid changes these arrays.
    z = np.linspace(0., .5, 17)
    implicit = assemble_thermal_fem(*ARGS, **KW)
    explicit = assemble_thermal_fem(*ARGS, **KW, axial_nodes_m=z)
    np.testing.assert_array_equal(explicit.z_m, implicit.z_m)
    np.testing.assert_array_equal(explicit.matrix.toarray(), implicit.matrix.toarray())
    np.testing.assert_array_equal(explicit.mass.toarray(), implicit.mass.toarray())
    np.testing.assert_array_equal(explicit.rhs, implicit.rhs)
    options = dict(density=3., specific_heat=4., times=[0., .1, .3],
                   time_step=.02, history=PowerHistory(), solver='direct', rtol=1e-11)
    original = solve_transient_fem(*ARGS, **KW, **options)
    controlled = solve_transient_fem(*ARGS, **KW, **options, axial_nodes_m=z,
                                   time_step_by_interval_s=[.02, .02])
    for a, b in zip(original.frames, controlled.frames):
        np.testing.assert_array_equal(a.temperature_K, b.temperature_K)
    assert original.steps == controlled.steps


def test_graded_mesh_recovers_independent_linear_steady_solution():
    # Constant planar flux, insulated side: exact T=C+D*z, independent of mesh spacing.
    z = .5*np.expm1(4*np.linspace(0., 1., 17))/np.expm1(4.)
    result = solve_steady_fem(*ARGS, **KW, axial_nodes_m=z, solver='direct', rtol=1e-11)
    c = 6./(2.+1.4*2./(1.4+2.*.5))
    d = -2.*c/(1.4+2.*.5)
    np.testing.assert_allclose(result.temperature_K, np.broadcast_to((c+d*z)[:, None], result.temperature_K.shape), rtol=1e-12)
    np.testing.assert_allclose(result.temperature_at(.1, .2, [.01,.17,.42]), c+d*np.array([.01,.17,.42]), rtol=1e-12)
    assert result.diagnostics['relative_energy_imbalance'] < 1e-12


def test_interval_steps_obey_caps_and_keep_independent_nodal_energy():
    # Assigning a cap to the wrong interval or assuming uniform z breaks these checks.
    z = .5*np.expm1(4*np.linspace(0., 1., 17))/np.expm1(4.)
    result = solve_transient_fem(*ARGS, **KW, axial_nodes_m=z, density=3., specific_heat=4.,
        times=[0.,.1,.3], time_step=.05, time_step_by_interval_s=[.02,.05],
        history=PowerHistory(), solver='direct', rtol=1e-11)
    np.testing.assert_array_equal(result.frames[0].temperature_K, 0.)
    np.testing.assert_allclose([s['dt_s'] for s in result.steps], [.02]*5+[.05]*4, rtol=1e-14)
    final = result.frames[-1]
    xy, tri = final.disk_mesh.p, final.disk_mesh.t
    edges1, edges2 = xy[:, tri[1]]-xy[:, tri[0]], xy[:, tri[2]]-xy[:, tri[0]]
    areas = np.abs(edges1[0]*edges2[1]-edges1[1]*edges2[0])/2
    planar_integrals = np.sum(final.temperature_K[:,tri].mean(axis=1)*areas, axis=1)
    energy = 3.*4.*np.trapezoid(planar_integrals, x=z)
    assert result.steps[-1]['stored_energy_J'] == pytest.approx(energy, rel=1e-12)
    for step in result.steps:
        assert step['storage_rate_W']+step['heat_loss_W'] == pytest.approx(step['input_W'], abs=1e-10)


@pytest.mark.parametrize('z', [np.linspace(0.,.5,16), np.linspace(.01,.5,17),
    np.linspace(0.,.49,17), np.r_[0.,np.ones(16)*.5], np.r_[0.,np.nan,np.linspace(.1,.5,15)]])
def test_rejects_invalid_explicit_axial_nodes(z):
    with pytest.raises(ValueError):
        assemble_thermal_fem(*ARGS, **KW, axial_nodes_m=z)


@pytest.mark.parametrize('caps,times', [([.02], [0.,.1,.3]),
    ([.02,0.], [0.,.1,.3]), ([.02,np.nan], [0.,.1,.3]),
    ([.02,.1], [0.,.1,.3]), ([.02], [.1,.3])])
def test_rejects_invalid_interval_caps(caps, times):
    with pytest.raises(ValueError):
        solve_transient_fem(*ARGS, **KW, density=3., specific_heat=4., times=times,
            time_step=.05, time_step_by_interval_s=caps, history=PowerHistory(),
            solver='direct', rtol=1e-11)


def test_axial_line_preconditioner_matches_independent_vertical_blocks():
    from thermal_opd.fem import _axial_line_preconditioner
    system = assemble_thermal_fem(*ARGS, **KW,
        axial_nodes_m=.5*np.expm1(4*np.linspace(0.,1.,17))/np.expm1(4.))
    matrix = system.matrix+12.*system.mass/.003
    rhs = np.sin(np.arange(matrix.shape[0])+1.)
    actual = _axial_line_preconditioner(matrix,system.disk_mesh.nvertices)@rhs
    expected = np.empty_like(rhs)
    for xy_node in range(system.disk_mesh.nvertices):
        indices = xy_node+np.arange(17)*system.disk_mesh.nvertices
        expected[indices] = np.linalg.solve(matrix[indices][:,indices].toarray(),rhs[indices])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_axial_line_cg_reproduces_direct_implicit_steps_on_graded_mesh():
    z=.5*np.expm1(4*np.linspace(0.,1.,17))/np.expm1(4.)
    options=dict(axial_nodes_m=z,density=3.,specific_heat=4.,times=[0.,.003,.1,.3],
        time_step=.05,time_step_by_interval_s=[.0001,.02,.05],history=PowerHistory(),rtol=1e-11)
    direct=solve_transient_fem(*ARGS,**KW,**options,solver='direct')
    iterative=solve_transient_fem(*ARGS,**KW,**options,solver='cg',linear_preconditioner='axial_lines')
    for a,b in zip(direct.frames,iterative.frames):
        np.testing.assert_allclose(a.temperature_K,b.temperature_K,rtol=1e-9,atol=1e-11)
    assert max(s['relative_linear_residual'] for s in iterative.steps)<1e-10
    assert max(s['balance_absolute_error_W'] for s in iterative.steps)<1e-8


def test_rejects_unknown_linear_preconditioner():
    with pytest.raises(ValueError):
        solve_transient_fem(*ARGS,**KW,density=3.,specific_heat=4.,times=[0.,.1],
            time_step=.05,history=PowerHistory(),solver='cg',rtol=1e-11,linear_preconditioner='invalid')


def test_axial_lines_avoid_general_sparse_lu_workspace(monkeypatch):
    # A fine XY mesh has millions of disjoint short vertical lines. General
    # sparse LU needlessly allocates ordering/workspace for the whole system.
    from thermal_opd import fem
    from scipy.sparse import diags
    def disallow_general_lu(*args, **kwargs):
        raise AssertionError('Vertical tridiagonal blocks must not use general sparse LU')
    monkeypatch.setattr(fem,'factorized',disallow_general_lu)
    xy_nodes,layers=19,4
    diagonal=4.+np.arange(xy_nodes*layers)/100.
    off=-np.ones(xy_nodes*(layers-1))
    matrix=diags((off,diagonal,off),(-xy_nodes,0,xy_nodes)).tocsr()
    rhs=np.cos(np.arange(diagonal.size))
    actual=fem._axial_line_preconditioner(matrix,xy_nodes)@rhs
    np.testing.assert_allclose(matrix@actual,rhs,rtol=1e-12,atol=1e-12)
