"""Shared numerical consistency and saved-evidence checks for both models.

No solver dispatch or model-specific pass criteria belong in this module.
"""
from pathlib import Path
import numpy as np
from thermal_opd.run_io import read_json,read_numeric_data,data_directory,validation_check,validation_report

def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_field_errors(actual, stored, *, skip_pv=False):
    for key, value in actual.items():
        if skip_pv and key == 'opd_relative_pv':
            continue
        require(np.allclose(value, stored[key], rtol=1e-8, atol=1e-12), f'Saved field metric differs: {key}')


def mesh_boundary_edges(triangles):
    edges = np.concatenate((triangles[[0, 1]].T, triangles[[1, 2]].T, triangles[[2, 0]].T))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return edges[counts==1]


def quadratic_fem_integral_weights(xy, tri, z, xy_dof_coordinates):
    """Analytically integrate saved P2 fields on a linear triangular prism mesh.

    Returns disk-area (m²), polygon-boundary (m), and sorted axial (m)
    weights.  These integrate the actual quadratic nodal shape functions;
    they are not lumped P1 weights.  Basis is used only to identify DOFs,
    never to assemble a form or rerun a thermal solver.  Corrupt coordinates
    or axial nodes that are not vertex/midpoint pairs are rejected.
    """
    from skfem import MeshTri, Basis, ElementTriP2
    xy, tri = np.asarray(xy), np.asarray(tri)
    z, coordinates = np.asarray(z), np.asarray(xy_dof_coordinates)
    require(xy.ndim==2 and xy.shape[0]==2 and np.all(np.isfinite(xy)), 'Invalid FEM vertex coordinates')
    require(tri.ndim==2 and tri.shape[0]==3 and tri.shape[1]>0
            and np.issubdtype(tri.dtype,np.integer) and tri.min()>=0 and tri.max()<xy.shape[1], 'Invalid FEM topology')
    require(z.ndim==1 and z.size>=3 and z.size%2==1 and np.all(np.isfinite(z))
            and np.all(np.diff(z)>0), 'Invalid quadratic FEM axial coordinates')
    require(np.allclose(z[1::2], (z[:-2:2]+z[2::2])/2, rtol=0,
                        atol=64*np.finfo(float).eps*max(np.max(np.abs(z)),np.finfo(float).tiny)),
            'Quadratic FEM axial midpoint coordinates differ')
    mesh=MeshTri(xy,tri,sort_t=False)
    basis=Basis(mesh,ElementTriP2())
    tolerance=64*np.finfo(float).eps*max(np.max(np.abs(xy)),np.finfo(float).tiny)
    require(coordinates.shape==basis.doflocs.shape and np.all(np.isfinite(coordinates))
            and np.allclose(coordinates,basis.doflocs,rtol=0,atol=tolerance), 'Quadratic FEM DOF coordinates differ')
    v1,v2=xy[:,tri[1]]-xy[:,tri[0]],xy[:,tri[2]]-xy[:,tri[0]]
    areas=np.abs(v1[0]*v2[1]-v1[1]*v2[0])/2
    require(np.all(areas>0), 'Degenerate FEM triangle')
    area_weights=np.zeros(basis.N)
    # Integral of each P2 vertex shape is zero; each edge shape is A/3.
    np.add.at(area_weights,basis.element_dofs[3:].ravel(),np.tile(areas/3,3))
    side_weights=np.zeros(basis.N)
    facets=mesh.boundary_facets();edges=mesh.facets[:,facets]
    lengths=np.linalg.norm(xy[:,edges[1]]-xy[:,edges[0]],axis=0)
    np.add.at(side_weights,edges.ravel(),np.tile(lengths/6,2))
    np.add.at(side_weights,basis.facet_dofs[0,facets],2*lengths/3)
    axial_weights=np.zeros(z.size);widths=z[2::2]-z[:-2:2]
    axial_weights[:-2:2]+=widths/6
    axial_weights[1::2]+=2*widths/3
    axial_weights[2::2]+=widths/6
    return area_weights,side_weights,axial_weights


def points_inside_polygon(x, y, p, edges):
    inside = np.ones(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=bool)
    scale = np.max(np.abs(p))
    for edge in edges:
        a, b = p[:, edge].T
        normal = np.array([b[1]-a[1], a[0]-b[0]])
        normal *= 1 if np.dot(normal, a+b) >= 0 else -1
        inside &= normal[0]*(x-a[0])+normal[1]*(y-a[1]) <= 64*np.finfo(float).eps*scale**2
    return inside


def finite_difference_laplacian(temperature,p,r,phi,z,*,space_step):
    """Independent cylindrical spatial stencil shared by steady/transient checks."""
    r,phi,z=np.broadcast_arrays(np.asarray(r,dtype=float),np.asarray(phi,dtype=float),np.asarray(z,dtype=float))
    h=space_step
    if not np.isfinite(h) or h<=0:
        raise ValueError('Require positive finite spatial step')
    if np.any(r<=h) or np.any(r+h>=p.radius) or np.any(z<=h) or np.any(z+h>=p.thickness):
        raise ValueError('PDE stencil must stay inside the cylinder away from r=0')
    center=temperature(r,phi,z)
    plus,minus=temperature(r+h,phi,z),temperature(r-h,phi,z)
    radial=(plus-2*center+minus)/h**2+(plus-minus)/(2*h*r)
    dphi=h/r
    angular=(temperature(r,phi+dphi,z)-2*center+temperature(r,phi-dphi,z))/(dphi**2*r**2)
    axial=(temperature(r,phi,z+h)-2*center+temperature(r,phi,z-h))/h**2
    return radial+angular+axial


def saved_validation_checks(directory,summary):
    """Original report slots/order; only the model validator marks executed checks."""
    checks={}
    def add(name,scope,evidence=None,metrics=None,threshold=None):
        checks[name]=validation_check(name,'not_run',scope=scope,evidence=evidence,metrics=metrics,threshold=threshold)
    add('joint_modal_truncation','Only the recorded angles, times, history and finite-reference tolerances',summary.get('selection_directory',str(directory)))
    add('projection_quadrature_convergence','Joint modal/projection refinement does not isolate projection quadrature')
    add('pde_residual','Independent PDE experiment/evidence required')
    add('initial_condition','Saved zero-time semi-analytical field; FEM nodes additionally checked in FEM studies')
    add('initial_neighborhood','Exact t=0 assignment does not establish t→0+ convergence')
    add('long_time_steady_limit','Recorded long-time comparison with same retained steady modes','summary.json',threshold=summary.get('precision_target',1e-3))
    add('thermal_boundaries','Sampled semi-analytical front/rear/side residuals against retained heat flux','boundary_summary.json')
    add('steady_energy_balance','Independent steady surface integrals; only the retained source','summary.json',metrics=summary.get('energy_balance'),threshold='1e-9 W + 1e-7*abs(modal_input_W)')
    add('fem_energy_balance','Reloaded nodal storage/loss integrals and recorded input')
    add('fem_weak_equation_residual','Discrete weak-form linear residual; not pointwise normal flux')
    add('fem_independent_reference','Independent FEM reference experiment/evidence required')
    add('fem_space_convergence','Only recorded mesh-refinement target; smoke is not a final certificate')
    add('fem_time_convergence','Separate transient time-refinement acceptance required')
    add('cross_method_comparison','Fields/metrics may be available; agreement is not an independent verification certificate')
    return checks


def validate_saved_evidence_run(directory,summary=None):
    """Recheck saved scalar evidence, including old mixed/missing-model-type runs."""
    path=Path(directory)
    summary=read_json(path/'summary.json') if summary is None else summary
    require(summary.get('status')=='COMPLETE','Run is not complete')
    require(summary.get('kind')=='scientific_verification',f"Unsupported run kind: {summary.get('kind')}")
    report=read_json(path/'verification.json')
    with read_numeric_data(data_directory(path)/'verification_fields.npz') as f:
        if not all(np.all(np.isfinite(f[k])) for k in f.files):
            raise ValueError('Nonfinite scientific verification evidence')
        for row in report['checks']:
            validation_check(row['name'],row['status'])
            if row['status']=='not_run':
                continue
            key=row['evidence']
            actual=float(np.max(np.abs(f[key])))
            threshold=row['threshold']
            expected='passed' if actual<=threshold else 'failed'
            if not np.isfinite(threshold) or threshold<0 or row['status']!=expected or not np.isclose(actual,row['metrics']['max_error'],rtol=1e-12,atol=1e-15):
                raise ValueError('Scientific verification metric/status differs from saved evidence')
    recomputed=validation_report(report['checks'],config=report['config'],run_directory=path)
    if report['status']!=recomputed['status'] or report['counts']!=recomputed['counts'] or summary['verification_status']!=report['status']:
        raise ValueError('Scientific verification aggregate status differs')
    return saved_validation_result(summary,report)


def saved_validation_result(summary,report,boundary_check=None):
    """Serialize the established result envelope without changing legacy field names."""
    return {'status':'PASS','kind':summary.get('kind'),'run_directory':'.',
            'mesh_convergence_established':summary.get('mesh_convergence_established'),
            'transient_truncation_established':summary.get('transient_truncation_established'),
            'fem_convergence_established':summary.get('fem_convergence_established'),
            'transient_boundary_check':boundary_check,'verification':report,
            'note':'PASS verifies saved run consistency and stated numerical gates; smoke PASS is not final convergence.'}
