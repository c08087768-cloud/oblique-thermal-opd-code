"""Independent tensor-product P1/P2 thermal FEM on a polygonal cylinder.

Six-node P1 or eighteen-node P2 prisms retain independent nodal unknowns.
Constant k and face-specific Robin h permit exact tensor-product assembly.
The Gaussian is integrated directly on the front, without modal projection.
"""

from dataclasses import dataclass, field
from operator import index
from time import perf_counter

import numpy as np
from matplotlib.tri import Triangulation
from scipy.sparse import diags, kron, csr_matrix, eye
from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg, spsolve, factorized, LinearOperator
from scipy.linalg.lapack import dpttrf, dpttrs, dpbtrf, dpbtrs
from skfem import MeshTri, MeshLine, Basis, FacetBasis, ElementTriP1, ElementLineP1, ElementTriP2, ElementLineP2, LinearForm, BilinearForm, asm
from skfem.models.poisson import laplace


@BilinearForm
def value_mass(u, v, w):
    # scikit-fem 8's DiscreteField subclass predates NumPy 2's array_wrap API.
    return np.asarray(u)*np.asarray(v)


@dataclass
class FemSolution:
    disk_mesh: MeshTri
    z_m: np.ndarray
    temperature_K: np.ndarray  # (z node, xy node), rise above ambient
    diagnostics: dict
    field_degree: int = 1
    _finder: object = field(init=False, repr=False)
    _origins: np.ndarray = field(init=False, repr=False)
    _inverse: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.field_degree not in (1,2):raise ValueError('Require P1 or P2 FEM field degree')
        p, t = self.disk_mesh.p, self.disk_mesh.t
        self._finder = Triangulation(p[0], p[1], t.T).get_trifinder()
        triangles = p[:, t].transpose(2, 1, 0)
        self._origins = triangles[:, 0]
        self._inverse = np.linalg.inv(np.stack((triangles[:, 1]-triangles[:, 0],
                                                triangles[:, 2]-triangles[:, 0]), axis=2))
        if self.field_degree==2:
            self._quadratic_dofs=Basis(self.disk_mesh,ElementTriP2()).element_dofs

    def transmitted_mask(self, x0, y0, refraction_angle):
        if not np.isfinite(refraction_angle) or abs(refraction_angle) >= np.pi/2:
            raise ValueError('Require a finite forward refraction angle')
        x, y = np.broadcast_arrays(x0, y0)
        shift = self.z_m[-1]*np.tan(refraction_angle)
        # Convex polygon: both endpoints inside guarantees the whole ray inside.
        return (self._finder(x, y) >= 0) & (self._finder(x, y+shift) >= 0)

    def temperature_at(self, x, y, z):
        """Interpolate the actual prism shape functions; reject extrapolation."""
        x, y, z = np.broadcast_arrays(x, y, z)
        if not all(np.all(np.isfinite(a)) for a in (x, y, z)):
            raise ValueError('Coordinates must be finite')
        cells = np.asarray(self._finder(x.ravel(), y.ravel()), dtype=int)
        if np.any(cells < 0) or np.any(z < 0) or np.any(z > self.z_m[-1]):
            raise ValueError('Point is outside the FEM polygon or axial interval')
        planar = np.column_stack((x.ravel(), y.ravel()))-self._origins[cells]
        bary = np.einsum('nij,nj->ni', self._inverse[cells], planar)
        bary = np.column_stack((1-bary.sum(axis=1), bary))
        if self.field_degree==2:
            a,b,c=bary.T
            shapes=np.column_stack((a*(2*a-1),b*(2*b-1),c*(2*c-1),4*a*b,4*b*c,4*a*c))
            bounds=self.z_m[::2]
            layer=np.clip(np.searchsorted(bounds,z.ravel(),side='right')-1,0,bounds.size-2)
            s=(z.ravel()-bounds[layer])/(bounds[layer+1]-bounds[layer])
            line=np.column_stack(((1-s)*(1-2*s),4*s*(1-s),s*(2*s-1)))
            dofs=self._quadratic_dofs[:,cells].T;result=np.zeros(x.size)
            for j in range(3):
                result+=line[:,j]*np.sum(shapes*self.temperature_K[(2*layer+j)[:,None],dofs],axis=1)
            return result.reshape(x.shape)
        nodes = self.disk_mesh.t[:, cells].T
        layer = np.clip(np.searchsorted(self.z_m, z.ravel(), side='right')-1, 0, self.z_m.size-2)
        fraction = (z.ravel()-self.z_m[layer])/(self.z_m[layer+1]-self.z_m[layer])
        front = np.sum(bary*self.temperature_K[layer[:, None], nodes], axis=1)
        rear = np.sum(bary*self.temperature_K[(layer+1)[:, None], nodes], axis=1)
        return ((1-fraction)*front+fraction*rear).reshape(x.shape)

    def refracted_opd(self, x0, y0, refraction_angle, dn_dT, *, point_chunk_size):
        """Integrate prism temperature exactly along each fixed straight ray.

        Split at all crossed xy edges and z layers. Within each segment the
        P1 fields are quadratic and P2 fields quartic along a segment; two
        or three Gauss points integrate them exactly up to floating point error.
        """
        chunk = index(point_chunk_size)
        if chunk < 1:
            raise ValueError('point_chunk_size must be positive')
        x, y = np.broadcast_arrays(x0, y0)
        if not np.all(self.transmitted_mask(x, y, refraction_angle)):
            raise ValueError('A ray leaves the FEM mesh before the rear face')
        slope = np.tan(refraction_angle)
        edges = self.disk_mesh.p[:, self.disk_mesh.facets]
        dx = edges[0, 1]-edges[0, 0]
        nonvertical = np.abs(dx) > 8*np.finfo(float).eps*np.max(np.abs(self.disk_mesh.p))
        xa, ya, delta_x, delta_y = (edges[0, 0, nonvertical], edges[1, 0, nonvertical],
                                    dx[nonvertical], (edges[1, 1]-edges[1, 0])[nonvertical])
        result = np.empty(x.size)
        if self.field_degree==2:q,w=np.polynomial.legendre.leggauss(3)
        for begin in range(0, x.size, chunk):
            all_z, all_x, all_y, all_weights, all_ids = [], [], [], [], []
            for j in range(begin, min(begin+chunk, x.size)):
                xx, yy = x.ravel()[j], y.ravel()[j]
                cuts = self.z_m if self.field_degree==1 else self.z_m[::2]
                if slope != 0:
                    fraction = (xx-xa)/delta_x
                    active = (fraction >= 0) & (fraction <= 1)
                    crossings = (ya[active]+fraction[active]*delta_y[active]-yy)/slope
                    crossings = crossings[(crossings > 0) & (crossings < self.z_m[-1])]
                    cuts = np.unique(np.concatenate((cuts, crossings)))
                middle = (cuts[1:]+cuts[:-1])/2
                half = (cuts[1:]-cuts[:-1])/2
                if self.field_degree==2:
                    zq=(middle[:,None]+half[:,None]*q).ravel()
                else:
                    zq = (middle[:, None] + half[:, None]*np.array([-1., 1.])/np.sqrt(3)).ravel()
                all_z.append(zq)
                all_x.append(np.full(zq.size, xx))
                all_y.append(yy+slope*zq)
                all_weights.append((half[:,None]*w).ravel() if self.field_degree==2 else np.repeat(half, 2))
                all_ids.append(np.full(zq.size, j-begin, dtype=int))
            if not all_z:
                continue
            values = self.temperature_at(np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z))
            totals = np.bincount(np.concatenate(all_ids), weights=values*np.concatenate(all_weights),
                                 minlength=min(chunk, x.size-begin))
            result[begin:begin+totals.size] = dn_dT*totals/np.cos(refraction_angle)
        return result.reshape(x.shape)


@dataclass
class FemSystem:
    disk_mesh: MeshTri
    z_m: np.ndarray
    matrix: object
    mass: object  # geometric consistent mass; multiply by rho*cp for heat capacity
    rhs: np.ndarray
    Mxy: object
    Mz: object
    Sxy: object
    diagnostics: dict
    field_degree: int = 1
    xy_dof_coordinates: object = None
    Kxy: object = None


def assemble_thermal_fem(radius, thickness, conductivity, h_front, h_back, h_side, front_flux,
                         *, disk_refinements, axial_layers, quadrature_order, axial_nodes_m=None,field_degree=1):
    """Shared steady/transient prism space, conduction, Robin and front load."""
    if not all(np.isfinite(v) and v > 0 for v in (radius, thickness, conductivity)):
        raise ValueError('Geometry and conductivity must be finite and positive')
    if not all(np.isfinite(v) and v >= 0 for v in (h_front, h_back, h_side)) or max(h_front, h_back, h_side)==0:
        raise ValueError('Robin coefficients must be nonnegative, with some heat loss')
    level, layers, order = index(disk_refinements), index(axial_layers), index(quadrature_order)
    degree=index(field_degree)
    if degree not in (1,2):raise ValueError('Require P1 or P2 FEM field degree')
    if degree==2 and order<4:raise ValueError('P2 consistent mass and Robin terms require quadrature_order >= 4')
    if min(level, layers) < 1 or order < 2:
        raise ValueError('Require refinements/layers >= 1 and quadrature >= 2')
    started = perf_counter()
    mesh = MeshTri.init_circle(nrefs=level).scaled(radius)
    z = np.linspace(0, thickness, layers+1) if axial_nodes_m is None else np.array(axial_nodes_m,dtype=float,copy=True)
    if (z.shape!=(layers+1,) or not np.all(np.isfinite(z)) or z[0]!=0.
            or z[-1]!=thickness or np.any(np.diff(z)<=0)):
        raise ValueError('Axial nodes must have layers+1 increasing finite entries, from 0 to thickness')
    xy_element=ElementTriP1() if degree==1 else ElementTriP2()
    xy_basis = Basis(mesh, xy_element, intorder=order)
    side_basis = FacetBasis(mesh, xy_element, intorder=order)
    z_basis = Basis(MeshLine(z[None, :]), ElementLineP1() if degree==1 else ElementLineP2(), intorder=order)
    Mxy, Kxy, Sxy = asm(value_mass, xy_basis), asm(laplace, xy_basis), asm(value_mass, side_basis)
    Mz, Kz = asm(value_mass, z_basis), asm(laplace, z_basis)
    if degree==2:
        permutation=np.argsort(z_basis.doflocs[0]);z=z_basis.doflocs[0,permutation]
        Mz=Mz[permutation][:,permutation];Kz=Kz[permutation][:,permutation]
    first, last = np.zeros(z.size), np.zeros(z.size)
    first[0], last[-1] = 1., 1.
    matrix = (conductivity*kron(Kz, Mxy) + kron(Mz, conductivity*Kxy+h_side*Sxy)
              + kron(diags(h_front*first+h_back*last), Mxy)).tocsr()
    @LinearForm
    def source(v, w):
        q = front_flux(w.x[0], w.x[1])
        if not np.all(np.isfinite(q)):
            raise ValueError('Front heat flux must be finite')
        return q*np.asarray(v)
    planar_rhs = asm(source, xy_basis)
    rhs = np.zeros(matrix.shape[0])
    rhs[:xy_basis.N] = planar_rhs
    assembly_seconds = perf_counter()-started
    edges = mesh.p[:, mesh.facets[:, mesh.boundary_facets()]]
    area, perimeter = float(np.sum(xy_basis.dx)), float(np.sum(side_basis.dx))
    inradius = float(np.min(np.linalg.norm((edges[:, 0]+edges[:, 1])/2, axis=0)))
    metadata = {'degrees_of_freedom': int(matrix.shape[0]), 'disk_triangles': int(mesh.nelements),
                'prism_elements': int(mesh.nelements*layers), 'disk_refinements': level,
                'axial_layers': layers, 'quadrature_order': order, 'assembly_seconds': assembly_seconds,
                'polygon_area_m2': area, 'polygon_perimeter_m': perimeter, 'polygon_inradius_m': inradius,
                'relative_area_deficit': float(1-area/(np.pi*radius**2)),
                'relative_perimeter_deficit': float(1-perimeter/(2*np.pi*radius))}
    if degree==2:
        metadata.update(field_degree_xy=2,field_degree_z=2,geometry_degree=1,
            axial_dofs=int(z.size),xy_dofs=int(xy_basis.N))
    return FemSystem(mesh,z,matrix,kron(Mz,Mxy).tocsr(),rhs,Mxy,Mz,Sxy,metadata,degree,xy_basis.doflocs,Kxy)


def solve_steady_fem(radius, thickness, conductivity, h_front, h_back, h_side, front_flux,
                     *, disk_refinements, axial_layers, quadrature_order, solver, rtol, axial_nodes_m=None,
                     field_degree=1,mirror_reduction=False):
    """Solve k*grad + Robin = prescribed front surface flux (SI)."""
    if solver not in ('direct','cg') or not 0 < rtol < 1:
        raise ValueError('Require direct/cg solver and 0 < rtol < 1')
    if mirror_reduction and solver!='cg':raise ValueError('Mirror reduction requires CG')
    system = assemble_thermal_fem(radius,thickness,conductivity,h_front,h_back,h_side,front_flux,
                                  disk_refinements=disk_refinements,axial_layers=axial_layers,
                                  quadrature_order=quadrature_order,axial_nodes_m=axial_nodes_m,field_degree=field_degree)
    mesh, z, matrix, rhs = system.disk_mesh, system.z_m, system.matrix, system.rhs
    Mxy, Mz, Sxy = system.Mxy, system.Mz, system.Sxy
    solving = perf_counter()
    iterations = 0
    if solver == 'direct':
        vector = spsolve(matrix, rhs)
    else:
        def callback(_):
            nonlocal iterations
            iterations += 1
        solve_cg=cg
        if mirror_reduction:
            q,mirror_diagnostics=_mirror_basis(system)
            solve_cg=_mirror_cg(system,q,cg)
        vector, info = solve_cg(matrix, rhs, M=diags(1/matrix.diagonal()), rtol=rtol, atol=0,
                          maxiter=10000, callback=callback)
        if info != 0:
            raise RuntimeError(f'FEM CG failed to converge: info={info}')
    solve_seconds = perf_counter()-solving
    if not np.all(np.isfinite(vector)):
        raise RuntimeError('FEM solution contains nonfinite values')
    temperature = vector.reshape(z.size, Mxy.shape[0])
    front_loss = h_front*np.sum(Mxy@temperature[0])
    rear_loss = h_back*np.sum(Mxy@temperature[-1])
    side_loss = h_side*np.asarray(Mz.sum(axis=0)).ravel() @ temperature @ np.asarray(Sxy.sum(axis=0)).ravel()
    power = float(rhs.sum())
    residual = np.linalg.norm(matrix@vector-rhs)
    # Signed sources can have zero net input; preserve absolute diagnostics too.
    scale = max(abs(power), np.finfo(float).tiny)
    diagnostics = {**system.diagnostics, 'solver': solver, 'solver_rtol': rtol, 'iterations': iterations,
                   'solve_seconds': solve_seconds,
                   'input_W': power, 'front_loss_W': float(front_loss), 'rear_loss_W': float(rear_loss),
                   'side_loss_W': float(side_loss), 'balance_absolute_error_W': float(abs(front_loss+rear_loss+side_loss-power)),
                   'relative_energy_imbalance': float(abs(front_loss+rear_loss+side_loss-power)/scale),
                   'linear_residual_norm': float(residual),
                   'relative_linear_residual': float(residual/max(np.linalg.norm(rhs), np.finfo(float).tiny))}
    if mirror_reduction:
        diagnostics['mirror_reduction']=mirror_diagnostics
    return FemSolution(mesh, z, temperature, diagnostics,field_degree=system.field_degree)


@dataclass
class TransientFemSolution:
    times_s: np.ndarray
    frames: list
    steps: list
    diagnostics: dict


def _mirror_basis(system):
    """Orthonormal FEM mirror orbits; reject non-invariant operators or loads."""
    xy=system.xy_dof_coordinates;n=xy.shape[1];ids=np.arange(n)
    tree=cKDTree(xy.T);scale=np.max(np.abs(xy));permutations=[]
    for signs in ((-1,1),(1,-1)):
        distance,p=tree.query(xy.T*np.array(signs))
        if np.max(distance)>128*np.finfo(float).eps*scale or not np.array_equal(np.sort(p),ids) or not np.array_equal(p[p],ids):
            raise ValueError('FEM mesh has no exact mirror symmetry')
        permutations.append(p)
    px,py=permutations
    if not np.array_equal(px[py],py[px]):raise ValueError('FEM mirror permutations do not commute')
    errors={}
    for name,A in (('mass',system.Mxy),('stiffness',system.Kxy),('robin',system.Sxy)):
        denominator=np.linalg.norm(A.data)
        error=max(np.linalg.norm((A[p,:][:,p]-A).data) for p in permutations)/max(denominator,np.finfo(float).tiny)
        if error>5e-13:raise ValueError('FEM operator has broken mirror symmetry: '+name)
        errors[name]=float(error)
    load=system.rhs.reshape(-1,n)
    if any(np.linalg.norm(load[:,p]-load)>5e-13*max(np.linalg.norm(load),np.finfo(float).tiny) for p in permutations):
        raise ValueError('FEM load has broken mirror symmetry')
    representative=np.minimum.reduce((ids,px,py,px[py]))
    _,groups=np.unique(representative,return_inverse=True);sizes=np.bincount(groups)
    planar=csr_matrix((1/np.sqrt(sizes[groups]),(ids,groups)),shape=(n,len(sizes)))
    q=kron(eye(len(system.z_m),format='csr'),planar,format='csr')
    return q,dict(method='orthonormal_double_mirror_fem_subspace',
        full_degrees_of_freedom=q.shape[0],reduced_degrees_of_freedom=q.shape[1],
        xy_reflection_operator_relative_errors=errors,full_space_physical_diagnostics=True)


def _wider_residual_dtype():
    """Return an actually wider accumulator, never infer precision from its name."""
    return np.longdouble if np.finfo(np.longdouble).eps<np.finfo(float).eps else None


def _extended_residual_check(A,x,b,tolerance,*,return_defect=False):
    """Reevaluate the same float64 equation/nodes at higher accumulation precision.

    This does not change the linear equation, x, its saved dtype or tolerance.
    Some Windows builds have no wider NumPy dtype; they cannot claim this check.
    The optional float64 defect uses this same accurate matvec for refinement.
    """
    dtype=_wider_residual_dtype()
    if dtype is None:return None
    if not all(np.all(np.isfinite(v)) for v in (A.data,x,b)) or not np.isfinite(tolerance):
        raise RuntimeError('Non-finite input to extended FEM residual check')
    residual=A@np.asarray(x,dtype=dtype)-np.asarray(b,dtype=dtype)
    norm=np.linalg.norm(residual)
    if not np.all(np.isfinite(residual)) or not np.isfinite(norm):
        raise RuntimeError('Non-finite extended FEM residual')
    # Compare before converting to JSON float. A@x and the norm both use dtype.
    review=dict(passed=bool(norm<=dtype(tolerance)),residual_norm_extended=float(norm),
        original_tolerance=float(tolerance),accumulator_dtype=np.dtype(dtype).name,
        accumulator_epsilon=float(np.finfo(dtype).eps),mantissa_bits=int(np.finfo(dtype).nmant)+1)
    return (review,np.asarray(-residual,dtype=float)) if return_defect else review


def _mirror_cg(system,q,original_cg):
    """Reduce the solve, checking the original full equation after reconstruction.

    ``maxiter`` bounds each CG segment: the initial solve and at most three
    residual corrections. Time-step equations and the requested tolerance stay
    unchanged; a nonzero inner status or failed final check remains a failure.
    """
    nxy=q.shape[1]//len(system.z_m);cache={};solve_index=0
    def compress(value):
        reduced=np.asarray(q.T@value)
        if np.linalg.norm(q@reduced-value)>5e-12*max(np.linalg.norm(value),np.finfo(float).tiny):
            raise ValueError('CG vector has broken mirror symmetry')
        return reduced
    def solve(A,b,**kwargs):
        nonlocal solve_index
        solve_index+=1
        if cache.get('full') is not A:
            reduced=(q.T@(A@q)).tocsr()
            cache.clear();cache.update(full=A,reduced=reduced,
                preconditioner=_axial_line_preconditioner(reduced,nxy,field_degree=system.field_degree))
        options=dict(kwargs);options['M']=cache['preconditioner']
        if options.get('x0') is not None:options['x0']=compress(options['x0'])
        if options.get('callback') is not None:
            callback=options['callback'];options['callback']=lambda x:callback(q@x)
        reduced,info=original_cg(cache['reduced'],compress(b),**options)
        full=np.asarray(q@reduced)
        tolerance=max(options.get('atol',0.),options.get('rtol',1e-5)*np.linalg.norm(b))
        correction_defect=None
        def accepted(residual):
            nonlocal correction_defect
            correction_defect=residual
            norm=np.linalg.norm(residual)
            if not np.isfinite(norm):raise RuntimeError('Mirror CG returned a non-finite residual')
            if norm<=tolerance:return True
            result=_extended_residual_check(A,full,b,tolerance,return_defect=True)
            if result is None:return False
            review,correction_defect=result
            if not review['passed']:
                review['correction_defect_arithmetic']='extended_accumulation_then_float64'
            review.update(linear_solve_index=solve_index,residual_norm_float64=float(norm),
                rhs_norm=float(np.linalg.norm(b)),rtol=float(options.get('rtol',1e-5)),
                atol=float(options.get('atol',0.)))
            audit=system.diagnostics.setdefault('linear_residual_arithmetic',dict(
                formula='norm(A @ x - b)',saved_temperature_dtype='float64',
                raw_step_residuals_preserved=True,events=[]))
            audit['events'].append(review)
            return review['passed']
        if info==0:
            # Recursive CG residuals can drift from b-Ax on stiff late-time
            # systems. Correct the actual full equation, without relaxing its
            # original tolerance or changing the FEM/time discretization.
            correction_options={k:v for k,v in options.items() if k!='x0'}
            correction_options.update(rtol=.01,atol=0.)
            if not np.all(np.isfinite(full)):
                raise RuntimeError('Mirror CG returned a non-finite solution')
            converged=False
            for _ in range(3):
                residual=b-A@full
                if not np.all(np.isfinite(residual)):
                    raise RuntimeError('Mirror CG returned a non-finite residual')
                if accepted(residual):
                    converged=True;break
                if kwargs.get('callback') is not None:
                    base=full.copy();callback=kwargs['callback']
                    correction_options['callback']=lambda delta,base=base:callback(base+q@delta)
                # Once float64 cancellation matters, use the same accurately
                # accumulated defect that failed acceptance, cast only at the
                # correction solve boundary. A, b, float64 x and limits stay fixed.
                delta,code=original_cg(cache['reduced'],np.asarray(q.T@correction_defect),**correction_options)
                if code:raise RuntimeError('Mirror CG residual correction did not converge')
                reduced=reduced+delta;full=np.asarray(q@reduced)
            if not np.all(np.isfinite(full)) or (not converged and not accepted(b-A@full)):
                raise RuntimeError('Mirror CG did not meet the original full-space residual tolerance')
        return full,info
    return solve


def _axial_line_preconditioner(matrix, xy_nodes,field_degree=1):
    """Invert each vertical nodal block; the original CG equation is unchanged."""
    # Permute to contiguous vertical lines with zero couplings between lines.
    # SPD tridiagonal LDL factors need only two vectors, avoiding sparse-LU
    # ordering/workspace for a full system of millions of independent blocks.
    z_nodes = matrix.shape[0]//xy_nodes
    if field_degree==2:
        bands=np.zeros((3,xy_nodes,z_nodes))
        for offset in range(3):
            bands[offset,:,:z_nodes-offset]=matrix.diagonal(offset*xy_nodes).reshape(z_nodes-offset,xy_nodes).T
        packed=np.asfortranarray(bands.reshape(3,-1))
        factors,info=dpbtrf(packed,lower=1,overwrite_ab=1)
        if info:raise RuntimeError(f'P2 axial preconditioner factorization failed: info={info}')
        def apply_p2(vector):
            rhs=np.asfortranarray(np.asarray(vector).reshape(z_nodes,xy_nodes).T.reshape(-1,1))
            solved,info=dpbtrs(factors,rhs,lower=1,overwrite_b=1)
            if info:raise RuntimeError(f'P2 axial preconditioner solve failed: info={info}')
            return solved.reshape(xy_nodes,z_nodes).T.ravel()
        return LinearOperator(matrix.shape,matvec=apply_p2,rmatvec=apply_p2,dtype=float)
    diagonal = np.ascontiguousarray(matrix.diagonal().reshape(z_nodes,xy_nodes).T).ravel()
    off = np.zeros((xy_nodes,z_nodes))
    off[:,:-1] = matrix.diagonal(xy_nodes).reshape(z_nodes-1,xy_nodes).T
    diagonal,off,info = dpttrf(diagonal,off.ravel()[:-1],overwrite_d=1,overwrite_e=1)
    if info!=0:
        raise RuntimeError(f'Axial preconditioner factorization failed: info={info}')
    def apply(vector):
        rhs = np.array(np.asarray(vector).reshape(z_nodes,xy_nodes).T,copy=True,order='C').reshape(-1,1)
        solved,info = dpttrs(diagonal,off,rhs,overwrite_b=1)
        if info!=0:
            raise RuntimeError(f'Axial preconditioner solve failed: info={info}')
        return solved.reshape(xy_nodes,z_nodes).T.ravel()
    return LinearOperator(matrix.shape,matvec=apply,rmatvec=apply,dtype=float)


def solve_transient_fem(radius, thickness, conductivity, h_front, h_back, h_side, front_flux,
                        *, density, specific_heat, times, time_step, history,
                        disk_refinements, axial_layers, quadrature_order, solver, rtol,
                        axial_nodes_m=None, time_step_by_interval_s=None,linear_preconditioner='jacobi',
                        time_integrator='backward_euler',field_degree=1,mirror_reduction=False,observation_callback=None):
    """Zero initial rise, consistent capacity, backward Euler at exact observations.

    Each interval ends at the next requested observation (including fractional
    final steps).  Endpoint g(t_next) drives the implicit step.  Sparse factors
    are reused by dt.  Frames reuse FemSolution's interpolation/ray integration.
    Optional interval caps correspond to successive intervals in ``times``;
    this requires an explicit zero-time observation. Each cap is <= time_step.
    Explicit axial nodes change only mesh spacing; default calls remain uniform.
    Optional axial-line blocks accelerate CG on graded meshes without changing
    the equation, implicit method, stopping tolerance or direct solver.
    BDF2 is opt-in; its method-specific storage/balance evidence is additional
    to the unchanged endpoint loss and secant energy-rate fields.
    """
    times = np.asarray(times,dtype=float)
    if times.ndim!=1 or not times.size or not np.all(np.isfinite(times)) or times[0]<0 or np.any(np.diff(times)<=0):
        raise ValueError('Observation times must be finite, nonnegative and strictly increasing')
    if not all(np.isfinite(v) and v>0 for v in (density,specific_heat,time_step)):
        raise ValueError('Capacity and time step must be finite and positive')
    if solver not in ('direct','cg') or not 0 < rtol < 1:
        raise ValueError('Require direct/cg solver and 0 < rtol < 1')
    if linear_preconditioner not in ('jacobi','axial_lines'):
        raise ValueError('Require jacobi or axial_lines linear preconditioner')
    if time_integrator not in ('backward_euler','bdf2'):
        raise ValueError('Require backward_euler or bdf2 time integrator')
    if mirror_reduction and solver!='cg':raise ValueError('Mirror reduction requires CG')
    caps = None if time_step_by_interval_s is None else np.asarray(time_step_by_interval_s,dtype=float)
    if caps is not None and (times[0]!=0. or caps.shape!=(times.size-1,)
            or not np.all(np.isfinite(caps)) or np.any(caps<=0) or np.any(caps>time_step)):
        raise ValueError('Interval caps require t=0 and one finite positive cap <= time_step per interval')
    system = assemble_thermal_fem(radius,thickness,conductivity,h_front,h_back,h_side,front_flux,
                                  disk_refinements=disk_refinements,axial_layers=axial_layers,
                                  quadrature_order=quadrature_order,axial_nodes_m=axial_nodes_m,field_degree=field_degree)
    solve_cg=cg;mirror_diagnostics=None
    if mirror_reduction:
        q,mirror_diagnostics=_mirror_basis(system);solve_cg=_mirror_cg(system,q,cg)
    if time_integrator=='bdf2':
        result=_solve_bdf2_fem(system,density,specific_heat,times,time_step,history,solver,rtol,caps,
                               linear_preconditioner,solve_cg,observation_callback)
        if mirror_diagnostics is not None:result.diagnostics['mirror_reduction']=mirror_diagnostics
        return result
    capacity = density*specific_heat*system.mass
    vector = np.zeros_like(system.rhs)
    factors, steps, frames = {}, [], []
    time, power = 0., float(system.rhs.sum())
    started = perf_counter()
    for interval_index,target in enumerate(times):
        # Use an equal partition within each observation interval; avoids tiny
        # floating-point remainder steps and never exceeds the requested dt.
        cap = time_step if caps is None or interval_index==0 else caps[interval_index-1]
        count = int(np.ceil(max(0.,target-time)/cap))
        if count:
            dt = (target-time)/count
            start_time = time
            step_matrix = (capacity/dt+system.matrix).tocsr()
            if solver=='direct' and dt not in factors:
                factors[dt] = factorized(step_matrix.tocsc())
            preconditioner = None
            if solver=='cg':
                preconditioner = (diags(1/step_matrix.diagonal()) if linear_preconditioner=='jacobi'
                    else _axial_line_preconditioner(step_matrix,system.Mxy.shape[0],field_degree=system.field_degree))
            for j in range(count):
                time = float(start_time+(j+1)*dt) if j+1<count else float(target)
                load = system.rhs*history.value(time)
                rhs = capacity@vector/dt+load
                previous = vector
                if solver=='direct':
                    vector = factors[dt](rhs)
                else:
                    vector, info = solve_cg(step_matrix,rhs,x0=previous,M=preconditioner,rtol=rtol,atol=0.,maxiter=10000)
                    if info!=0:
                        raise RuntimeError(f'Transient FEM CG failed: info={info}')
                if not np.all(np.isfinite(vector)):
                    raise RuntimeError('Transient FEM has nonfinite temperatures')
                storage = float(np.sum(capacity@(vector-previous))/dt)
                loss = float(np.sum(system.matrix@vector))
                incoming = float(load.sum())
                residual = float(np.linalg.norm(step_matrix@vector-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny))
                steps.append({'time_s':time,'dt_s':float(dt),'input_W':incoming,'storage_rate_W':storage,
                              'heat_loss_W':loss,'stored_energy_J':float(np.sum(capacity@vector)),
                              'balance_absolute_error_W':abs(storage+loss-incoming),
                              'relative_linear_residual':residual})
        diagnostic = {'time_s':float(target),'stored_energy_J':float(np.sum(capacity@vector))}
        frame = FemSolution(system.disk_mesh,system.z_m,vector.reshape(system.z_m.size,-1).copy(),diagnostic,field_degree=system.field_degree)
        if frames:
            # Geometry is identical; reuse the first frame's spatial locator.
            frame._finder,frame._origins,frame._inverse = frames[0]._finder,frames[0]._origins,frames[0]._inverse
        frames.append(frame)
        if observation_callback is not None:observation_callback(frame)
    resolution = {} if caps is None else {'time_step_by_interval_s':caps.tolist()}
    if linear_preconditioner!='jacobi':
        resolution['linear_preconditioner'] = linear_preconditioner
    if mirror_diagnostics is not None:resolution['mirror_reduction']=mirror_diagnostics
    return TransientFemSolution(times,frames,steps,{**system.diagnostics,**resolution,'time_integrator':'backward_euler',
        'time_step_max_s':float(time_step),'solver':solver,'solver_rtol':rtol,'density_kg_per_m3':density,
        'specific_heat_J_per_kg_K':specific_heat,'full_input_W':power,'solve_seconds':perf_counter()-started})


def _solve_bdf2_fem(system,density,specific_heat,times,time_step,history,solver,rtol,caps,preconditioner_kind,cg_solver=None,observation_callback=None):
    """Variable-step BDF2 of the independently assembled FEM ODE C T'+K T=f.

    BE starts the zero initial state, and restarts when dt/previous_dt>1.5.
    The conservative restart avoids large step-ratio transitions between
    observation intervals; there is no claim of adaptive error control.
    Legacy storage_rate_W is still the actual energy increment divided by dt.
    bdf_storage_rate_W records the method's three-state discrete derivative.
    """
    capacity=density*specific_heat*system.mass
    solve_cg=cg if cg_solver is None else cg_solver
    vector=np.zeros_like(system.rhs);older=None;previous_dt=None
    frames,steps,factors=[],[],{}
    time=0.;last_key=None;preconditioner=None
    started=perf_counter()
    for interval_index,target in enumerate(times):
        cap=time_step if caps is None or interval_index==0 else caps[interval_index-1]
        count=int(np.ceil(max(0.,target-time)/cap))
        if count:
            dt=(target-time)/count;start_time=time
            for j in range(count):
                time=float(start_time+(j+1)*dt) if j+1<count else float(target)
                ratio=0. if previous_dt is None else dt/previous_dt
                restart=older is None or ratio>1.5*(1+16*np.finfo(float).eps)
                if restart:
                    a0,a1,a2,order=1.,-1.,0.,1
                else:
                    a0=(1+2*ratio)/(1+ratio);a1=-(1+ratio);a2=ratio**2/(1+ratio);order=2
                key=(float(dt),float(a0))
                if key!=last_key:
                    step_matrix=(a0*capacity/dt+system.matrix).tocsr()
                    if solver=='direct':
                        if key not in factors:
                            factors[key]=factorized(step_matrix.tocsc())
                    else:
                        preconditioner=(diags(1/step_matrix.diagonal()) if preconditioner_kind=='jacobi'
                            else _axial_line_preconditioner(step_matrix,system.Mxy.shape[0],field_degree=system.field_degree))
                    last_key=key
                previous=vector
                history_term=previous if restart else (-a1)*previous-a2*older
                load=system.rhs*history.value(time)
                rhs=capacity@history_term/dt+load
                if solver=='direct':
                    vector=factors[key](rhs)
                else:
                    vector,info=solve_cg(step_matrix,rhs,x0=previous,M=preconditioner,rtol=rtol,atol=0.,maxiter=10000)
                    if info!=0:
                        raise RuntimeError(f'BDF2 FEM CG failed: info={info}')
                if not np.all(np.isfinite(vector)):
                    raise RuntimeError('BDF2 FEM has nonfinite temperatures')
                storage=float(np.sum(capacity@(vector-previous))/dt)
                derivative=a0*(vector-previous)
                if not restart:
                    derivative=derivative+a2*(older-previous)
                bdf_storage=float(np.sum(capacity@derivative)/dt)
                loss=float(np.sum(system.matrix@vector));incoming=float(load.sum())
                residual=float(np.linalg.norm(step_matrix@vector-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny))
                steps.append(dict(time_s=time,dt_s=float(dt),input_W=incoming,storage_rate_W=storage,
                    heat_loss_W=loss,stored_energy_J=float(np.sum(capacity@vector)),
                    balance_absolute_error_W=abs(storage+loss-incoming),relative_linear_residual=residual,
                    bdf_storage_rate_W=bdf_storage,bdf_balance_absolute_error_W=abs(bdf_storage+loss-incoming),
                    bdf_alpha0=float(a0),bdf_alpha1=float(a1),bdf_alpha2=float(a2),time_order=order))
                older=previous;previous_dt=dt
        frame=FemSolution(system.disk_mesh,system.z_m,vector.reshape(system.z_m.size,-1).copy(),
                          {'time_s':float(target),'stored_energy_J':float(np.sum(capacity@vector))},field_degree=system.field_degree)
        if frames:
            frame._finder,frame._origins,frame._inverse=frames[0]._finder,frames[0]._origins,frames[0]._inverse
        frames.append(frame)
        if observation_callback is not None:observation_callback(frame)
    resolution={} if caps is None else {'time_step_by_interval_s':caps.tolist()}
    if preconditioner_kind!='jacobi':
        resolution['linear_preconditioner']=preconditioner_kind
    return TransientFemSolution(times,frames,steps,{**system.diagnostics,**resolution,'time_integrator':'bdf2',
        'bdf2_startup':'backward_euler','bdf2_max_step_ratio':1.5,'time_step_max_s':float(time_step),
        'solver':solver,'solver_rtol':rtol,'density_kg_per_m3':density,'specific_heat_J_per_kg_K':specific_heat,
        'full_input_W':float(system.rhs.sum()),'solve_seconds':perf_counter()-started})
