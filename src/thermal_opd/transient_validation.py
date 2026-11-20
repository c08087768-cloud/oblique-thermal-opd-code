"""Transient constraints, independent references and saved experiment review.

Boundary sidecars retain their original schema and flat-run reading support.
"""
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from thermal_opd.run_io import data_directory

from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.model import PreparedModel,ModalResolution
from thermal_opd.transient import PowerHistory
from scipy.linalg import eigh_tridiagonal
from thermal_opd.projection import reconstruct_cosine_field
from thermal_opd.run_io import figure_directory,resolve_run_path,read_json,read_numeric_data,write_json,validation_report,validation_check
from thermal_opd.run_io import relative_metadata_paths,file_sha256
from thermal_opd.convergence import field_errors,within_tolerance
from thermal_opd.validation_common import (require,validate_field_errors,mesh_boundary_edges,points_inside_polygon,
    finite_difference_laplacian,saved_validation_checks,saved_validation_result,quadratic_fem_integral_weights)

TRANSIENT_RUN_KINDS=frozenset(('transient_modal_fem_study','transient_joint_truncation_selection','transient_fine_fields',
                             'transient_fem_convergence_validation'))


BOUNDARY_FILES = ('boundary_checks.npz','boundary_summary.json','boundary_residuals.csv')
RESIDUAL_KEYS = ('front_modal_residual_W_per_m2','rear_residual_W_per_m2','side_residual_W_per_m2')
CSV_COLUMNS = ('time_s','front_modal_max_abs_W_per_m2','rear_max_abs_W_per_m2','side_max_abs_W_per_m2',
               'front_modal_over_full_power_peak_flux','rear_over_full_power_peak_flux','side_over_full_power_peak_flux')


def solution_from_saved_transient(summary, fields):
    """Restore exactly the saved physical parameters, history, J, C and roots."""
    kind = summary['kind']
    if kind not in ('transient_fine_fields','transient_modal_fem_study'):
        raise ValueError('Boundary check needs a transient field/FEM run, not a selection run')
    history = summary['precision_scope']['history'] if kind=='transient_fine_fields' else summary['history']
    J = summary['selected_axial_modes'] if kind=='transient_fine_fields' else summary['axial_counts'][-1]
    coefficients = {int(m):fields[f'source_coefficients_m{m}_W_per_m2'] for m in fields['orders']}
    roots = {int(m):fields[f'radial_roots_m{m}_per_m'] for m in fields['orders']}
    prepared=PreparedModel(PhysicalParameters(**summary['physical_parameters_SI']),summary['angle_deg'],
                           ModalResolution(**summary['modal_settings']),coefficients,roots)
    return prepared.solve_transient(num_axial_modes=J,history=PowerHistory(**history))


def boundary_metrics(fields):
    return np.stack([np.max(np.abs(fields[key]),axis=(1,2)) for key in RESIDUAL_KEYS],axis=1)


def write_transient_boundary_check(directory,model,times,r,phi,side_z,*,
                                   relative_tolerance,absolute_tolerance,provenance=None):
    """Write actual boundary values and CSV/JSON metrics; return PASS/FAIL report.

    Relative errors use the full-power prescribed Gaussian peak, fixed across
    time. The absolute tolerance handles zero source without dividing by zero.
    This does not test pointwise FEM fluxes or source-projection convergence.
    """
    directory = Path(directory)
    angle_deg=model.prepared.angle_deg
    if not all(np.isfinite(t) and t>=0 for t in (relative_tolerance,absolute_tolerance)):
        raise ValueError('Boundary tolerances must be finite and nonnegative')
    f = transient_boundary_fields(model,times,r,phi,side_z)
    p = model.prepared.parameters
    peak = float(oblique_gaussian_heat_flux(0.,0.,p.beam_radius,np.deg2rad(angle_deg),p.absorbed_power))
    scale = abs(peak)
    absolute = boundary_metrics(f)
    relative = absolute/scale if scale>0 else np.zeros_like(absolute)
    limit = absolute_tolerance+relative_tolerance*scale
    report = dict(version=1,status='PASS' if np.max(absolute)<=limit else 'FAIL',
        checked_at_utc=datetime.now(timezone.utc).isoformat(),physical_parameters_SI=asdict(p),
        history=asdict(model.history),num_axial_modes=len(model.axial.beta),angle_deg=float(angle_deg),
        source_target='g(t)*q_modal; projection error against the Gaussian is separate',
        derivative_method='analytic Fourier-Bessel and axial derivatives',
        equations=['h*T-k*dT/dz-g(t)*q_modal','h*T+k*dT/dz','h*T+k*dT/dr'],
        boundary_order=['front_modal','rear','side'],times_s=f['times_s'].tolist(),
        grid=dict(r=len(r),phi=len(phi),side_z=len(side_z)),
        full_power_peak_gaussian_flux_W_per_m2=scale,
        relative_tolerance=float(relative_tolerance),absolute_tolerance_W_per_m2=float(absolute_tolerance),
        allowed_absolute_residual_W_per_m2=float(limit),
        max_absolute_residuals_W_per_m2=absolute.tolist(),
        max_residuals_over_full_power_peak_flux=relative.tolist(),
        zero_source_note='At zero prescribed power, normalized values are 0; the absolute residual gate still applies.',
        scope='Sampled semi-analytical Robin consistency only; not PDE/J convergence or FEM pointwise flux validation.',
        provenance=provenance)
    np.savez_compressed(data_directory(directory)/BOUNDARY_FILES[0],**f)
    with (data_directory(directory)/BOUNDARY_FILES[2]).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(np.column_stack((f['times_s'],absolute,relative)))
    write_json(directory/BOUNDARY_FILES[1],report)
    return report


def validate_transient_boundary_check(directory,summary):
    """Recompute stored residuals, reconstruct the model and check snapshot T."""
    directory = Path(directory)
    locations=(data_directory(directory)/BOUNDARY_FILES[0],directory/BOUNDARY_FILES[1],data_directory(directory)/BOUNDARY_FILES[2])
    present = [path.is_file() for path in locations]
    if not any(present) and not summary.get('boundary_validation_required',False):
        return dict(status='NOT_RUN',note='Legacy run has no saved transient boundary check.')
    require(all(present),'Missing transient boundary check files')
    report = read_json(directory/BOUNDARY_FILES[1])
    require(report['version']==1,'Unsupported transient boundary check version')
    with read_numeric_data(data_directory(directory)/'fields.npz') as saved, read_numeric_data(locations[0]) as f:
        model = solution_from_saved_transient(summary,saved)
        p = model.prepared.parameters
        require(report['physical_parameters_SI']==asdict(p) and report['history']==asdict(model.history)
                and report['num_axial_modes']==len(model.axial.beta) and report['angle_deg']==summary['angle_deg'],
                'Transient boundary model differs from saved run')
        snapshot_times=saved['times_s']
        times=f['times_s']
        indices=np.searchsorted(times,snapshot_times)
        require(np.array_equal(snapshot_times,summary['times_s']) and np.all(indices<times.size)
                and np.array_equal(times[indices],snapshot_times)
                and np.array_equal(report['times_s'],times),'Transient boundary times differ')
        if summary.get('boundary_history_required',False):
            with read_numeric_data(data_directory(directory)/'history.npz') as history:
                require(np.array_equal(times,history['times_s']),'Transient boundary history times differ')
        fine = summary['kind']=='transient_fine_fields'
        r,phi = (saved[key] for key in (('temperature_r_m','temperature_phi_rad') if fine else ('r_m','phi_rad')))
        require(np.array_equal(f['r_m'],r) and np.array_equal(f['phi_rad'],phi),'Transient boundary grid differs from saved fields')
        require(report['grid']==dict(r=r.size,phi=phi.size,side_z=f['side_z_m'].size),'Transient boundary grid metadata differs')
        expected = transient_boundary_fields(model,times,r,phi,f['side_z_m'])
        require(set(f.files)==set(expected),'Transient boundary array keys differ')
        for key,value in expected.items():
            require(f[key].shape==value.shape and np.all(np.isfinite(f[key]))
                    and np.allclose(f[key],value,rtol=1e-11,atol=1e-9),f'Transient boundary array differs: {key}')
        T = saved['temperature_K'] if fine else saved['temperature_sa_K']
        for key,plane in (('front_temperature_K',0),('rear_temperature_K',2)):
            require(np.allclose(f[key][indices],T[:,plane],rtol=1e-11,atol=1e-12),'Transient boundary temperature differs from actual saved snapshot')
        residuals = transient_residual_fields(f,p)
        for key,value in residuals.items():
            require(np.allclose(f[key],value,rtol=1e-11,atol=1e-12),'Saved transient boundary residual differs from T/gradient/load')
        absolute = boundary_metrics(residuals)
        peak = abs(float(oblique_gaussian_heat_flux(0.,0.,p.beam_radius,np.deg2rad(summary['angle_deg']),p.absorbed_power)))
        relative = absolute/peak if peak>0 else np.zeros_like(absolute)
        rel_tol,abs_tol = report['relative_tolerance'],report['absolute_tolerance_W_per_m2']
        require(all(np.isfinite(t) and t>=0 for t in (rel_tol,abs_tol)),'Invalid transient boundary tolerance')
        limit = abs_tol+rel_tol*peak
        require(np.isclose(report['full_power_peak_gaussian_flux_W_per_m2'],peak,rtol=1e-12,atol=0.)
                and np.isclose(report['allowed_absolute_residual_W_per_m2'],limit,rtol=1e-12,atol=0.),
                'Transient boundary normalization differs')
        for key,value in (('max_absolute_residuals_W_per_m2',absolute),('max_residuals_over_full_power_peak_flux',relative)):
            require(np.asarray(report[key]).shape==value.shape and np.allclose(report[key],value,rtol=1e-11,atol=1e-14),
                    'Transient boundary summary metrics differ')
        with locations[2].open(newline='',encoding='utf-8') as stream:
            rows = list(csv.reader(stream))
        require(tuple(rows[0])==CSV_COLUMNS,'Transient boundary CSV columns differ')
        table = np.asarray(rows[1:],dtype=float)
        expected_table = np.column_stack((times,absolute,relative))
        require(table.shape==expected_table.shape and np.allclose(table,expected_table,rtol=1e-11,atol=1e-14),
                'Transient boundary CSV differs')
        require(report['status']=='PASS' and np.max(absolute)<=limit,'Transient boundary residual exceeds tolerance')
    return dict(status='PASS',max_absolute_residuals_W_per_m2=np.max(absolute,axis=0).tolist(),
                boundary_order=report['boundary_order'],allowed_absolute_residual_W_per_m2=limit,
                report_file=BOUNDARY_FILES[1],table_file=str(locations[2].relative_to(directory)))


def transient_boundary_fields(model, times, r, phi, side_z):
    """Sample signed Robin residuals against g(t)*q_modal, in W/m².

    Boundary temperatures and analytic normal-coordinate derivatives are saved
    separately so validators can reconstruct residuals. This checks the retained
    model, not source projection accuracy or transient PDE/J convergence.
    At t=0 both T and the history's load are zero, including a step convention.
    """
    p = model.prepared.parameters
    times,r,phi,side_z = (np.asarray(a,dtype=float) for a in (times,r,phi,side_z))
    if any(a.ndim!=1 or not a.size or not np.all(np.isfinite(a)) for a in (times,r,phi,side_z)):
        raise ValueError('Require finite nonempty 1D boundary coordinates/times')
    if times[0]<0 or np.any(np.diff(times)<=0) or r[0]<0 or r[-1]>p.radius or np.any(np.diff(r)<=0):
        raise ValueError('Invalid boundary times/radii')
    if phi[0]!=0 or phi[-1]>=2*np.pi or np.any(np.diff(phi)<=0):
        raise ValueError('Invalid boundary azimuths')
    if side_z[0]!=0 or side_z[-1]!=p.thickness or np.any(np.diff(side_z)<=0):
        raise ValueError('Side boundary must cover z=0 to L')
    source = reconstruct_cosine_field(model.prepared.coefficients,model.prepared.roots,r,phi)
    front,rear,side,front_dz,rear_dz,side_dr = [],[],[],[],[],[]
    for t in times:
        front.append(model.temperature(r[:,None],phi[None,:],0.,t))
        rear.append(model.temperature(r[:,None],phi[None,:],p.thickness,t))
        side.append(model.temperature(p.radius,phi[None,:],side_z[:,None],t))
        front_dz.append(model.temperature_gradient(r[:,None],phi[None,:],0.,t)[1])
        rear_dz.append(model.temperature_gradient(r[:,None],phi[None,:],p.thickness,t)[1])
        side_dr.append(model.temperature_gradient(p.radius,phi[None,:],side_z[:,None],t)[0])
    f = dict(times_s=times,r_m=r,phi_rad=phi,side_z_m=side_z,modal_source_W_per_m2=source,
        front_temperature_K=np.array(front),rear_temperature_K=np.array(rear),side_temperature_K=np.array(side),
        front_dT_dz_K_per_m=np.array(front_dz),rear_dT_dz_K_per_m=np.array(rear_dz),side_dT_dr_K_per_m=np.array(side_dr),
        load_factors=np.array([model.history.value(t) for t in times]))
    f['front_load_W_per_m2'] = f['load_factors'][:,None,None]*source
    f.update(transient_residual_fields(f,p))
    return f


def transient_residual_fields(f,p):
    """Reconstruct signed residuals from boundary T, derivatives and load."""
    return dict(
        front_modal_residual_W_per_m2=p.h_eff*f['front_temperature_K']-p.conductivity*f['front_dT_dz_K_per_m']-f['front_load_W_per_m2'],
        rear_residual_W_per_m2=p.h_eff*f['rear_temperature_K']+p.conductivity*f['rear_dT_dz_K_per_m'],
        side_residual_W_per_m2=p.h_eff*f['side_temperature_K']+p.conductivity*f['side_dT_dr_K_per_m'])


def transient_pde_residual(temperature,p,r,phi,z,time,*,space_step,time_step):
    """Independent rho cp T_t-k Laplacian(T) residual, W/m³."""
    r,phi,z=np.broadcast_arrays(np.asarray(r,dtype=float),np.asarray(phi,dtype=float),np.asarray(z,dtype=float))
    h,dt=space_step,time_step
    if not all(np.isfinite(v) and v>0 for v in (h,dt)) or time<=dt:
        raise ValueError('Require positive finite steps and t>dt')
    laplacian=finite_difference_laplacian(lambda r,phi,z:temperature(r,phi,z,time),p,r,phi,z,space_step=h)
    temporal=(temperature(r,phi,z,time+dt)-temperature(r,phi,z,time-dt))/(2*dt)
    return p.density*p.specific_heat*temporal-p.conductivity*laplacian


def transient_solution_acceptance(model, times, config):
    """Physical gates shared by selection and independent result verification.

    Numerical settings/thresholds are supplied by the caller. Analytic basis
    relations establish retained-flux Robin constraints independently of time;
    sampled residuals additionally check field assembly and roundoff. Neither
    establishes projection accuracy against the untruncated Gaussian load.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.special import jv,jvp
    p,prepared = model.prepared.parameters,model.prepared
    times = np.asarray(times,dtype=float)
    require(times.ndim==1 and times.size>0 and np.all(np.isfinite(times))
            and np.all(times>0) and np.all(np.diff(times)>0),'Require increasing positive acceptance times')
    checks,fields = [],{}
    def check(name,values,threshold,scope):
        values=np.asarray(values)
        fields[name]=values
        metric=float(np.max(np.abs(values)))
        checks.append(validation_check(name,'passed' if np.isfinite(metric) and metric<=threshold else 'failed',
            metrics={'max_error':metric},threshold=threshold,scope=scope,evidence=name))
    rr,pp,zz=np.meshgrid(np.linspace(.15*p.radius,.8*p.radius,5),np.arange(8)*2*np.pi/8,
                        np.linspace(.15*p.thickness,.85*p.thickness,5),indexing='ij')
    rr,pp,zz=rr.ravel(),pp.ravel(),zz.ravel()
    fields['initial_temperature_K']=model.temperature(rr,pp,zz,0.)
    fields['initial_opd_m']=model.opd([0.,p.radius/4],[0.,0.],0.)
    check('initial_condition',np.r_[fields['initial_temperature_K'],fields['initial_opd_m']],0.,
          'Volume and refracted-ray samples at t=0; early positive accuracy is checked separately')
    radial=[]
    for m,lam in prepared.roots.items():
        radial.extend((p.h_eff*jv(m,lam*p.radius)+p.conductivity*lam*jvp(m,lam*p.radius))
                      /np.maximum(p.h_eff,p.conductivity*lam))
    X=model.axial.values(np.array([0.,p.thickness]))
    dX=model.axial.derivatives(np.array([0.,p.thickness]))
    scale=np.maximum(p.h_eff,p.conductivity*model.axial.beta)
    scale=np.maximum(scale,np.finfo(float).tiny)
    fields['basis_residuals']=np.r_[radial,(p.conductivity*dX[:,0]-p.h_eff*X[:,0])/scale,
                                    (p.conductivity*dX[:,1]+p.h_eff*X[:,1])/scale]
    check('boundary_basis',fields['basis_residuals'],config['basis_tolerance'],
          'Dimensionless radial/axial Robin identities for every retained mode, independent of time')
    long_time=max(times[-1],12/model.slowest_decay_rate,
                  12*model.history.tau if model.history.kind!='step' else 0.)
    boundary_times=np.unique(np.r_[0.,times,np.geomspace(times[0],long_time,41)])
    bc=transient_boundary_fields(model,boundary_times,np.linspace(0,p.radius,25),
                                 np.arange(32)*2*np.pi/32,np.linspace(0,p.thickness,33))
    fields['boundary_times_s']=boundary_times
    for key in RESIDUAL_KEYS:
        fields[key]=bc[key]
    check('thermal_boundaries',np.r_[tuple(bc[k].ravel() for k in RESIDUAL_KEYS)],
          config['boundary_tolerance_W_per_m2'],'Three assembled Robin residuals; retained front heat flux')
    if 'projection_tolerance' in config:
        from thermal_opd.model import project_modal_source
        r,phi=np.linspace(0,p.radius,25),np.arange(32)*2*np.pi/32
        projected=[reconstruct_cosine_field(prepared.coefficients,prepared.roots,r,phi)]
        for factor in (2,4):
            c=project_modal_source(p,prepared.angle_deg,prepared.roots,
                num_r=prepared.resolution.projection_r*factor,num_phi=prepared.resolution.projection_phi*factor)
            projected.append(reconstruct_cosine_field(c,prepared.roots,r,phi))
        fields['projection_grid_normalized_changes']=(np.array(projected[:-1])-projected[-1])/max(np.max(abs(projected[-1])),1e-30)
        check('projection_quadrature_convergence',fields['projection_grid_normalized_changes'],config['projection_tolerance'],
              'Fixed roots/M/N; projection quadrature factors 1/2/4; not Gaussian modal truncation error')
    a,w=leggauss(48);rg=(a+1)*p.radius/2;zg=(a+1)*p.thickness/2
    wr=w*p.radius/2*rg;wz=w*p.thickness/2;ph=np.arange(64)*2*np.pi/64
    q=reconstruct_cosine_field(prepared.coefficients,prepared.roots,rg,ph)
    input_power=2*np.pi*np.dot(wr,q.mean(axis=1))
    peak=2*p.absorbed_power*np.cos(np.deg2rad(prepared.angle_deg))/(np.pi*p.beam_radius**2)
    h=config['space_step_ratio']*min(p.radius,p.thickness)
    pde,energy=[],[]
    for t in times:
        dt=min(config['time_step_s'],t/100)
        pde.append(np.stack([transient_pde_residual(model.temperature,p,rr,pp,zz,t,
            space_step=step,time_step=dt)/max(peak/p.thickness,1.) for step in (h,h/2)]))
        Tp=model.temperature(rg[:,None,None],ph[None,:,None],zg[None,None,:],t+dt)
        Tm=model.temperature(rg[:,None,None],ph[None,:,None],zg[None,None,:],t-dt)
        storage=p.density*p.specific_heat*2*np.pi*np.einsum('rz,r,z->',((Tp-Tm)/(2*dt)).mean(axis=1),wr,wz)
        faces=model.temperature(rg[:,None],ph[None,:],0.,t)+model.temperature(rg[:,None],ph[None,:],p.thickness,t)
        side=model.temperature(p.radius,ph[None,:],zg[:,None],t)
        loss=p.h_eff*2*np.pi*(np.dot(wr,faces.mean(axis=1))+p.radius*np.dot(wz,side.mean(axis=1)))
        energy.append((storage+loss-model.history.value(t)*input_power)/max(abs(input_power),1e-30))
    fields['pde_normalized_residuals']=np.asarray(pde)
    fields['energy_normalized_residuals']=np.asarray(energy)
    check('pde_residual',fields['pde_normalized_residuals'],config['pde_tolerance'],
          'Independent interior finite differences at h/h2; dt<=min(configured dt,t/100)')
    if 'early_space_step_ratio' in config:
        early=[]
        for t in times[times<config['early_pde_before_s']]:
            diffusion_depth=np.sqrt(p.thermal_diffusivity*t)
            er,ep,ez=np.meshgrid(np.array([.2,.6])*p.radius,np.array([0.,np.pi/4,np.pi/2]),
                                np.array([.5,1.,2.])*diffusion_depth,indexing='ij')
            eh=min(h,config['early_space_step_ratio']*diffusion_depth)
            edt=min(config['time_step_s'],config['early_time_step_ratio']*t)
            require(np.all(ez+eh<p.thickness),'Early PDE stencil exceeds slab; adjust depth samples')
            early.append(np.stack([transient_pde_residual(model.temperature,p,er.ravel(),ep.ravel(),ez.ravel(),t,
                space_step=step,time_step=edt)/max(peak/p.thickness,1.) for step in (eh,eh/2)]))
        if early:
            fields['early_pde_normalized_residuals']=np.asarray(early)
            check('early_pde_residual',fields['early_pde_normalized_residuals'],config['pde_tolerance'],
                  'Independent finite differences inside the moving front diffusion layer; h/h2')
    check('energy_balance',fields['energy_normalized_residuals'],config['energy_tolerance'],
          'Independent volume finite-time storage and three surface quadratures')
    steady=prepared.solve_steady()
    Ts=steady.temperature(rr,pp,zz)
    fields['long_time_temperature_error']=(model.temperature(rr,pp,zz,long_time)-Ts)/max(np.max(abs(Ts)),1e-30)
    Os=steady.opd([0.,p.radius/4],[0.,0.])
    fields['long_time_opd_error']=(model.opd([0.,p.radius/4],[0.,0.],long_time)-Os)/max(np.max(abs(Os)),1e-30)
    check('long_time_steady_limit',np.r_[fields['long_time_temperature_error'],fields['long_time_opd_error']],
          config['limit_tolerance'],'Late temperature and OPD against the same retained steady solution')
    report=validation_report(checks,config={**config,'times_s':times.tolist(),'long_time_s':long_time},run_directory='.')
    return report,fields


def load_transient_selection(directory, parameters, angle, history='step'):
    """Read a revalidated selection; derive a missing legacy flag in memory only."""
    directory=Path(directory).resolve()
    reviewed=validate_saved_transient_run(directory)
    summary=read_json(directory/'summary.json')
    require(summary['kind']=='transient_joint_truncation_selection'
            and summary['physical_parameters_SI']==asdict(parameters),'Need selection with identical physical parameters')
    require(angle in summary['angles_deg'] and history=='step' and summary['history']['kind']=='step',
            'Angle/history is outside selected scope')
    joint=next((row for row in reviewed['verification']['checks']
                if row['name']=='joint_modal_truncation'),None)
    require(reviewed['status']=='PASS' and joint is not None and joint['status']=='passed',
            'Selected transient truncation is not established by saved checks')
    require('transient_truncation_established' not in summary
            or summary['transient_truncation_established'] is True,
            'Selected transient truncation explicitly not established')
    # The legacy producer omitted this field; no original JSON is rewritten.
    summary['transient_truncation_established']=True
    return summary


def transient_selection_errors(T,R,weights,O,RO,gradient,reference_gradient,depth,reference_depth,*,normalization):
    """Retain optical metrics and add gradient/front-layer accuracy evidence."""
    errors=field_errors(T,R,weights,O,RO,temperature_normalization=normalization)
    # Components have the same units: compare the gradient vector norm.
    # Separately normalizing a nearly zero component can turn roundoff into
    # a large relative error. Raw dr/dz fields remain available for review.
    scale=max(np.max(np.linalg.norm(reference_gradient,axis=0)),np.finfo(float).tiny)
    errors['gradient_relative_linf']=float(np.max(np.linalg.norm(gradient-reference_gradient,axis=0))/scale)
    errors['depth_temperature_linf_over_reference_peak']=float(np.max(abs(depth-reference_depth))
        /max(np.max(abs(reference_depth)),np.finfo(float).tiny))
    return errors


def transient_uniform_flux_1d_reference(thickness,conductivity,volumetric_capacity,flux,h_front,h_back,times,*,num_intervals):
    """Independent 1D finite-volume diffusion with exact-in-time step response.

    For a cylinder with insulated side and spatially uniform front flux, T
    depends only on z. Half control volumes at the faces impose Robin losses.
    Symmetric tridiagonal diagonalization avoids the prism FEM assembly and
    the semi-analytical Fourier-Bessel/axial characteristic equations.
    This reference still requires its own spatial refinement.
    """
    from operator import index
    n=index(num_intervals)
    times=np.asarray(times,dtype=float)
    if n<2 or times.ndim!=1 or not np.all(np.isfinite(times)) or np.any(times<0):
        raise ValueError('Invalid 1D reference grid or times')
    if not all(np.isfinite(v) and v>0 for v in (thickness,conductivity,volumetric_capacity)) or min(h_front,h_back)<0 or h_front+h_back<=0:
        raise ValueError('Require positive material/geometry and dissipative Robin faces')
    z=np.linspace(0,thickness,n+1)
    dz=thickness/n
    mass=np.full(n+1,volumetric_capacity*dz);mass[[0,-1]]*=.5
    diagonal=np.full(n+1,2*conductivity/dz)
    diagonal[0]=conductivity/dz+h_front
    diagonal[-1]=conductivity/dz+h_back
    off=np.full(n,-conductivity/dz)
    rates,vectors=eigh_tridiagonal(diagonal/mass,off/np.sqrt(mass[:-1]*mass[1:]))
    forcing=flux*vectors[0]/np.sqrt(mass[0])
    response=-np.expm1(-times[:,None]*rates[None,:])/rates[None,:]*forcing[None,:]
    temperature=(response@vectors.T)/np.sqrt(mass)[None,:]
    return z,temperature


def validate_transient_fem(path, summary):
    """Recompute mode sums, field errors and nodal storage/loss from saved files."""
    p,h = summary['physical_parameters_SI'],summary['h_eff_W_per_m2_K']
    tolerance = summary['sanity_tolerance']
    require(summary['sanity_pass'],'Transient sanity failed')
    evidence=None
    if summary['fem_convergence_established']:
        evidence=load_transient_fem_certificate(resolve_run_path(path,summary['fem_convergence_directory']),
            PhysicalParameters(**p),summary['angle_deg'],PowerHistory(**summary['history']),summary['times_s'])
        require(evidence['certificate_summary_sha256']==summary['fem_convergence_summary_sha256']
            and evidence['certificate_verification_sha256']==summary['fem_convergence_verification_sha256'],
            'Formal FEM certificate changed since case generation')
    if summary['transient_truncation_established']:
        selection=load_transient_selection(resolve_run_path(path,summary['selection_directory']),
            PhysicalParameters(**p),summary['angle_deg'],summary['history']['kind'])
        require(selection.get('transient_truncation_established') is True,'Selected transient truncation is not established')
        require(summary['modal_settings']==selection['selected_settings']['modal']
            and summary['axial_counts'][-1]==selection['selected_settings']['J'],
            'Saved transient case does not use selected modal settings')
        require(all(t==0 or t in selection['times_s'] for t in summary['times_s']),
            'Transient selection time scope does not cover every saved positive time')
    with read_numeric_data(data_directory(path)/'fields.npz') as f:
        require(all(np.all(np.isfinite(f[key])) for key in f.files),'Nonfinite transient field')
        times,weights = f['times_s'],f['area_weights_m2']
        require(np.array_equal(times,summary['times_s']) and times[0]==0 and np.all(np.diff(times)>0),'Bad transient times')
        temperature,opd = f['temperature_sa_K'],f['opd_sa_m']
        require(np.all(temperature[0]==0) and np.all(opd[0]==0),'Nonzero initial SA field')
        for total,orders in (('temperature_sa_K','order_temperature_K'),('opd_sa_m','order_opd_m'),
                             ('steady_temperature_K','steady_order_temperature_K'),('steady_opd_m','steady_order_opd_m'),
                             ('source_projected_W_per_m2','source_order_W_per_m2')):
            require(np.allclose(f[total],f[orders].sum(axis=0),rtol=1e-11,atol=1e-18),'Thermal mode sum mismatch')
        energies = np.sum(f['order_temperature_K']**2*weights[None,None,None],axis=(-2,-1))
        shares = np.divide(energies,energies.sum(axis=0),out=np.zeros_like(energies),where=energies.sum(axis=0)!=0)
        require(np.allclose(shares,f['temperature_plane_l2_squared_shares'],rtol=1e-11,atol=1e-14),'Modal norm shares differ')
        require(np.allclose(np.ptp(f['order_opd_m'],axis=-1),f['order_sampled_opd_pv_m'],rtol=1e-11,atol=1e-20),'Modal optical PV differs')
        long_errors = field_errors(f['long_time_temperature_K'].reshape(3,-1),f['steady_temperature_K'].reshape(3,-1),
                                  weights.ravel(),f['long_time_opd_m'],f['steady_opd_m'])
        validate_field_errors(long_errors,summary['long_time_errors_vs_steady'])
        for j,t in enumerate(times):
            if t==0:
                continue
            normalization=summary.get('temperature_normalization_by_time',['per_plane']*len(times))[j]
            for i in range(len(summary['axial_counts'])):
                err = field_errors(f['axial_check_temperature_K'][i,j].reshape(3,-1),temperature[j].reshape(3,-1),weights.ravel(),
                                   f['axial_check_opd_m'][i,j],opd[j],temperature_normalization=normalization)
                validate_field_errors(err,summary['axial_errors_vs_largest_J'][i][j])
            err = field_errors(temperature[j].reshape(3,-1),f['reference_temperature_K'][j].reshape(3,-1),weights.ravel(),
                               opd[j],f['reference_opd_m'][j],temperature_normalization=normalization)
            validate_field_errors(err,summary['transverse_errors_vs_finite_reference'][j])
            err = field_errors(temperature[j].reshape(3,-1),temperature[j].reshape(3,-1),weights.ravel(),
                               opd[j],f['higher_ray_quadrature_opd_m'][j],temperature_normalization=normalization)
            validate_field_errors(err,summary['ray_quadrature_errors'][j])
        mask,common,tm = f['opd_physical_mask'],f['opd_fem_common_mask'],f['temperature_fem_common_mask']
        require(mask.dtype==common.dtype==tm.dtype==bool,'Bad transient masks')
        require(np.all(~common|mask),'FEM optical aperture exceeds physical aperture')
        x,y = f['opd_x_m'],f['opd_y_m']
        angle = np.deg2rad(summary['refraction_angle_deg'])
        shift = p['thickness']*np.tan(angle)
        require(np.all(x[mask]**2+y[mask]**2<=(p['radius']*summary['aperture_radius_ratio'])**2*(1+1e-14)),
                'Ray entrance outside aperture')
        require(np.all(x[mask]**2+(y[mask]+shift)**2<=p['radius']**2*(1+1e-14)),'Ray exits cylinder')
        require(np.array_equal(f['opd_sa_fem_common_m'],opd[:,common[mask]]),'Common SA OPD differs')
        require(f['temperature_fem_K'].shape[0]==len(summary['fem_runs']),'Missing transient FEM run')
        for i,row in enumerate(summary['fem_runs']):
            require(row['max_relative_linear_residual']<tolerance and row['max_energy_balance_error_relative_to_full_input']<tolerance,
                    'Transient FEM solver/energy failed')
            tf,of = f['temperature_fem_K'][i],f['opd_fem_m'][i]
            difference = tf-temperature[...,tm]
            require(np.allclose(np.max(np.abs(difference),axis=-1),row['temperature_absolute_max_difference_K'],rtol=1e-10,atol=1e-13),
                    'Absolute temperature errors differ')
            rms = np.sqrt(np.sum(difference**2*weights[tm],axis=-1)/weights[tm].sum())
            require(np.allclose(rms,row['temperature_area_rms_difference_K'],rtol=1e-10,atol=1e-13),'Temperature RMS errors differ')
            require(np.all(tf[0]==0) and np.all(of[0]==0),'Nonzero initial FEM field')
            for j,t in enumerate(times):
                if t>0:
                    normalization=summary.get('temperature_normalization_by_time',['per_plane']*len(times))[j]
                    err = field_errors(tf[j],temperature[j][...,tm],weights[tm],of[j],f['opd_sa_fem_common_m'][j],
                                       temperature_normalization=normalization)
                    validate_field_errors(err,row['errors_vs_sa'][j])
            if evidence is not None:
                require(len(summary['fem_runs'])==1,'Formal case must reuse exactly the accepted FEM candidate')
                header=_transient_fem_case_header(path,dict(source_run=str(path),case_index=i))
                require(header['file']==evidence['header']['file'] and header['sha256']==evidence['header']['sha256'],
                    'Formal case did not reuse the certificate candidate nodes')
                require(np.array_equal(header['times'],evidence['header']['times']),
                    'Formal FEM source time schedule differs')
                tx=f['r_m'][:,None]*np.cos(f['phi_rad']);ty=f['r_m'][:,None]*np.sin(f['phi_rad'])
                depths=np.broadcast_to(f['z_m'],(header['times'].size,f['z_m'].size))
                samples=_transient_fem_case_samples(header,tx[tm],ty[tm],x[common],y[common],depths,angle)
                indices=[int(np.flatnonzero(header['times']==t)[0]) for t in times]
                require(np.allclose(samples['temperature'][indices],tf,rtol=1e-10,atol=1e-12),
                    'Saved formal FEM sampled temperature differs from candidate nodes')
                require(np.allclose(samples['opd'][indices],of,rtol=1e-10,atol=1e-20),
                    'Saved formal FEM sampled OPD differs from candidate nodes')
                require(samples['initial']==0 and samples['energy_error']<1e-7 and samples['linear_residual']<1e-7
                    and all(samples[key]<=1 for key in ('nodal_energy_ratio','storage_ratio','loss_ratio','load_ratio','balance_ratio')),
                    'Formal FEM saved nodal physical checks failed')
                continue
            with read_numeric_data(data_directory(path)/f'fem_run{i}.npz') as mesh:
                xy,tri,z,T = (mesh[key] for key in ('disk_nodes_xy_m','disk_triangles','axial_nodes_z_m','temperature_K'))
                require(T.shape==(times.size,z.size,xy.shape[1]) and np.all(np.isfinite(T)),'Bad transient FEM nodal fields')
                require(np.array_equal(mesh['times_s'],times) and np.all(T[0]==0),'Bad FEM initial time')
                require(z[0]==0 and z[-1]==p['thickness'] and np.all(np.diff(z)>0),'Bad FEM layers')
                require(tri.min()>=0 and tri.max()<xy.shape[1],'Bad FEM mesh topology')
                v1,v2 = xy[:,tri[1]]-xy[:,tri[0]],xy[:,tri[2]]-xy[:,tri[0]]
                areas = np.abs(v1[0]*v2[1]-v1[1]*v2[0])/2
                edges = mesh_boundary_edges(tri)
                lengths = np.linalg.norm(xy[:,edges[:,1]]-xy[:,edges[:,0]],axis=0)
                stored = p['density']*p['specific_heat']*np.trapezoid(np.sum(T[:,:,tri].mean(axis=-2)*areas,axis=-1),x=z,axis=-1)
                require(np.all(points_inside_polygon(x[common],y[common],xy,edges)&points_inside_polygon(x[common],y[common]+shift,xy,edges)),
                        'Transient ray outside FEM polygon')
                steps = {key:mesh['steps'][:,k] for k,key in enumerate(mesh['step_keys'])}
                require(all(np.all(np.isfinite(v)) for v in steps.values()),'Nonfinite transient FEM step')
                require(np.all(steps['dt_s']>0) and steps['time_s'][-1]==times[-1],'Invalid time integration range')
                require(np.allclose(np.diff(np.r_[0.,steps['time_s']]),steps['dt_s'],rtol=1e-10,atol=1e-12),'Incorrect step times')
                actual_storage = np.diff(np.r_[0.,steps['stored_energy_J']])/steps['dt_s']
                require(np.allclose(actual_storage,steps['storage_rate_W'],rtol=1e-8,atol=1e-12),'Stored-energy rate mismatch')
                balance = abs(actual_storage+steps['heat_loss_W']-steps['input_W'])
                scale = max(abs(row['full_input_W']),np.finfo(float).tiny)
                require(np.max(balance)/scale<tolerance,'Saved transient energy failed')
                history = PowerHistory(**summary['history'])
                expected_input = row['full_input_W']*np.array([history.value(t) for t in steps['time_s']])
                require(np.allclose(steps['input_W'],expected_input,rtol=1e-12,atol=1e-14),'Time-dependent load differs')
                for j,t in enumerate(times[1:],1):
                    k = np.flatnonzero(np.isclose(steps['time_s'],t,rtol=0,atol=1e-12))
                    require(k.size==1,'FEM missed observation time')
                    require(np.isclose(stored[j],steps['stored_energy_J'][k[0]],rtol=1e-10,atol=1e-12),'Nodal stored energy mismatch')
                    loss = h*(np.sum(areas*T[j,0,tri].mean(axis=0))+np.sum(areas*T[j,-1,tri].mean(axis=0))
                              +np.trapezoid(np.sum(T[j,:,edges].mean(axis=1)*lengths[:,None],axis=0),x=z))
                    require(np.isclose(loss,steps['heat_loss_W'][k[0]],rtol=1e-9,atol=1e-12),'Nodal heat loss mismatch')
        require((figure_directory(path)/'transient_modes.png').is_file() and (data_directory(path)/'modal_contributions.csv').is_file(),'Missing modal study artifacts')
    return evidence['verification'] if evidence is not None else None


def validate_transient_selection(path,summary):
    require(len(summary['cases'])==len(summary['angles_deg']),'Missing transient selection angle')
    angles=summary['angles_deg'];actual=[case['angle_deg'] for case in summary['cases']]
    require(len(set(angles))==len(angles) and len(set(actual))==len(actual)
            and set(angles)==set(actual),'Transient selection angle scope differs from saved cases')
    with read_numeric_data(data_directory(path)/'fields.npz') as f:
        require(all(np.all(np.isfinite(f[key])) for key in f.files),'Nonfinite transient selection field')
        weights = f['temperature_area_weights']
        for case in summary['cases']:
            label = f'theta{case["angle_deg"]:g}'
            selected = next(row for row in case['candidates'] if row['settings']==summary['selected_settings'])
            require(case['reference_pass'] and selected['accepted'] and case['denser_common_grid_pass'] and case['sampling_pass'],
                    'Transient selection gate failed')
            for j in range(len(summary['times_s'])):
                for left,right,pv_left,pv_right,stored,limit in (
                    ('reference','enlarged','reference','enlarged',case['reference_errors_by_time'],summary['reference_stability_tolerance']),
                    ('selected','reference','selected','reference',selected['errors_by_time'],summary['tolerance'])):
                    if summary.get('acceptance_version')==1:
                        err = transient_selection_errors(f[label+'_'+left+'_temperature_K'][j],f[label+'_'+right+'_temperature_K'][j],weights,
                            f[label+'_'+left+'_opd_m'][j],f[label+'_'+right+'_opd_m'][j],
                            f[label+'_'+left+'_gradient_K_per_m'][j],f[label+'_'+right+'_gradient_K_per_m'][j],
                            f[label+'_'+left+'_depth_temperature_K'][j],f[label+'_'+right+'_depth_temperature_K'][j],
                            normalization=summary['temperature_normalization_by_time'][j])
                    else:
                        err = field_errors(f[label+'_'+left+'_temperature_K'][j],f[label+'_'+right+'_temperature_K'][j],weights,
                                           f[label+'_'+left+'_opd_m'][j],f[label+'_'+right+'_opd_m'][j])
                    a,b = f[label+'_'+pv_left+'_continuous_pv_m'][j],f[label+'_'+pv_right+'_continuous_pv_m'][j]
                    require(a>0 and b>0,'Invalid continuous transient PV')
                    err['opd_relative_pv'] = abs(a-b)/b
                    validate_field_errors(err,stored[j])
                    require(within_tolerance(err,limit),'Transient reference/candidate above target')
                if summary.get('acceptance_version')==1:
                    err = transient_selection_errors(f[label+'_selected_temperature_K'][j],f[label+'_reference_temperature_K'][j],weights,
                        f[label+'_fine_selected_opd_m'][j],f[label+'_fine_reference_opd_m'][j],
                        f[label+'_selected_gradient_K_per_m'][j],f[label+'_reference_gradient_K_per_m'][j],
                        f[label+'_selected_depth_temperature_K'][j],f[label+'_reference_depth_temperature_K'][j],
                        normalization=summary['temperature_normalization_by_time'][j])
                else:
                    err = field_errors(f[label+'_selected_temperature_K'][j],f[label+'_reference_temperature_K'][j],weights,
                                       f[label+'_fine_selected_opd_m'][j],f[label+'_fine_reference_opd_m'][j])
                err['opd_relative_pv'] = abs(f[label+'_selected_continuous_pv_m'][j]-f[label+'_reference_continuous_pv_m'][j])/f[label+'_reference_continuous_pv_m'][j]
                validate_field_errors(err,case['denser_errors_by_time'][j])
                require(within_tolerance(err,summary['tolerance']),'Dense transient comparison above target')
                pv = f[label+'_selected_continuous_pv_m'][j]
                sampling = abs(np.ptp(f[label+'_figure_opd_m'][j])-pv)/pv
                require(np.isclose(sampling,case['sampling_errors_by_time'][j],rtol=1e-9,atol=1e-12),'Transient sampling metric differs')
                require(sampling<=summary['tolerance'],'Transient figure sampling above target')
    if 'acceptance_version' in summary:
        require(summary['acceptance_version']==1,'Unsupported transient acceptance version')
        from thermal_opd.model import ThermalModel
        p=PhysicalParameters(**summary['physical_parameters_SI'])
        settings=ModalResolution(**summary['selected_settings']['modal'])
        with read_numeric_data(data_directory(path)/'acceptance.npz') as evidence:
            for case in summary['cases']:
                label=f'theta{case["angle_deg"]:g}'
                model=ThermalModel(p,case['angle_deg'],settings).prepare().solve_transient(
                    num_axial_modes=summary['selected_settings']['J'],history=PowerHistory(**summary['history']))
                report,actual=transient_solution_acceptance(model,summary['times_s'],summary['acceptance_config'])
                require(report['status']=='passed' and case['physical_validation']['status']=='passed',
                        'Transient acceptance gate failed')
                stored_checks=case['physical_validation']['checks']
                require(len(stored_checks)==len(report['checks']),'Transient acceptance checks missing')
                for current,stored in zip(report['checks'],stored_checks):
                    require(current['name']==stored['name'] and current['status']==stored['status']
                        and current['threshold']==stored['threshold']
                        and np.isclose(current['metrics']['max_error'],stored['metrics']['max_error'],rtol=1e-8,atol=1e-12),
                        'Transient acceptance report differs')
                for key,value in actual.items():
                    require(label+'_'+key in evidence and evidence[label+'_'+key].shape==value.shape
                        and np.allclose(evidence[label+'_'+key],value,rtol=1e-10,atol=1e-12),
                        f'Transient acceptance data differs: {key}')


def validate_transient_fields(path,summary):
    p = summary['physical_parameters_SI']
    if summary['transient_truncation_established']:
        selected_path = resolve_run_path(path,summary['selection_directory'])
        validate_saved_transient_run(selected_path)
        selected = read_json(selected_path/'summary.json')
        require(selected['kind']=='transient_joint_truncation_selection' and selected['physical_parameters_SI']==p,
                'Fine transient selection has different physics')
        require(summary['angle_deg'] in selected['angles_deg'] and summary['modal_settings']==selected['selected_settings']['modal']
                and summary['selected_axial_modes']==selected['selected_settings']['J'],'Fine transient outside selected scope')
        require(summary['precision_scope']['times_s']==selected['times_s'],'Selected times differ')
    require(not summary['fem_convergence_established'],'Fine SA fields do not establish FEM convergence')
    with read_numeric_data(data_directory(path)/'fields.npz') as f:
        times,r,phi,z = (f[key] for key in ('times_s','temperature_r_m','temperature_phi_rad','temperature_z_m'))
        T,O,mask = f['temperature_K'],f['opd_m'],f['aperture_mask']
        s = summary['numerical_settings']
        require(np.array_equal(times,summary['times_s']) and times[0]==0 and np.all(np.diff(times)>0),'Invalid fine transient times')
        require(T.shape==(times.size,3,s['temperature_r'],s['temperature_phi']) and np.all(np.isfinite(T)),'Bad fine temperature shape/values')
        require(np.all(np.isfinite(r)) and np.all(np.diff(r)>0) and r[0]==0 and r[-1]<=p['radius'],'Bad fine radii')
        require(np.all(np.isfinite(phi)) and np.all(np.diff(phi)>0) and phi[0]==0 and phi[-1]<2*np.pi,'Bad fine angles')
        require(np.array_equal(z,[0.,p['thickness']/2,p['thickness']]),'Bad fine axial planes')
        require(mask.dtype==bool and mask.shape==(s['opd_xy'],s['opd_xy']) and O.shape==(times.size,)+mask.shape,'Bad fine OPD shape')
        require(np.all(np.isnan(O[:,~mask])),'Unexpected exterior fine OPD values')
        require(np.all(np.isfinite(O[:,mask])) and np.all(T[0]==0) and np.all(O[0,mask]==0),'Bad fine initial/optical values')
        x,y = f['opd_x_m'],f['opd_y_m']
        shift = p['thickness']*np.tan(np.deg2rad(summary['refraction_angle_deg']))
        require(np.all(x[mask]**2+y[mask]**2<=(p['radius']*summary['aperture_radius_ratio'])**2*(1+1e-14)), 'Fine entry outside aperture')
        require(np.all(x[mask]**2+(y[mask]+shift)**2<=p['radius']**2*(1+1e-14)), 'Fine ray leaves cylinder')
        require(np.allclose(T,f['order_temperature_K'].sum(axis=0),rtol=1e-11,atol=1e-14),'Fine thermal mode sum differs')
        require(np.allclose(O[:,mask],f['order_opd_m'][...,mask].sum(axis=0),rtol=1e-11,atol=1e-20),'Fine optical mode sum differs')
        sampled = np.ptp(O[:,mask],axis=1)
        require(np.allclose(sampled,f['sampled_pv_m'],rtol=1e-11,atol=1e-20),'Fine sampled PV differs')
        require(len(summary['snapshots'])==times.size,'Missing fine snapshot')
        for j,row in enumerate(summary['snapshots']):
            require(row['time_s']==times[j] and np.isclose(sampled[j],row['sampled_pv_m'],rtol=1e-11,atol=1e-20),'Saved snapshot PV differs')
            if summary['transient_truncation_established'] and times[j] in summary['precision_scope']['times_s']:
                require(row['continuous_pv_m']>0 and abs(sampled[j]-row['continuous_pv_m'])/row['continuous_pv_m']<=summary['precision_target'],
                        'Fine snapshot PV sampling above target')
        if summary['transient_truncation_established']:
            model = solution_from_saved_transient(summary,f)
            ii,jj = np.array([0,r.size//2,r.size-1]),np.array([0,phi.size//4,phi.size//2])
            for j in (1,times.size-2,times.size-1):
                for plane,depth in enumerate(z):
                    expected = model.temperature(r[ii],phi[jj],depth,times[j])
                    require(np.allclose(T[j,plane,ii,jj],expected,rtol=1e-11,atol=1e-13),'Fine field does not match saved model')
            w = np.broadcast_to(np.maximum(r[:,None],p['radius']*1e-9),T.shape[-2:]).ravel()
            errors = field_errors(T[-1].reshape(3,-1),f['steady_temperature_K'].reshape(3,-1),w,O[-1,mask],f['steady_opd_m'][mask])
            validate_field_errors(errors,summary['long_time_errors_vs_same_modes_steady'])
            require(within_tolerance(errors,summary['precision_target']),'Fine long-time limit above target')
            with read_numeric_data(data_directory(path)/'history.npz') as history:
                require(all(np.all(np.isfinite(history[key])) for key in history.files),'Nonfinite fine history')
                ht,ho,hT = history['times_s'],history['opd_m'],history['temperature_K']
                require(ht[0]==0 and np.all(np.diff(ht)>0) and ht[-1]==times[-1],'Invalid fine history times')
                require(np.allclose(np.ptp(ho,axis=1),history['sampled_pv_m'],rtol=1e-11,atol=1e-20),'Fine history PV differs')
                for j,time in enumerate(times):
                    k = np.flatnonzero(ht==time)
                    require(k.size==1,'Fine history missed snapshot')
                    require(np.allclose(hT[k[0],:,0],T[j,:,0,0],rtol=1e-11,atol=1e-13),'Fine center history differs')
                    require(np.allclose(ho[k[0]],O[j,mask],rtol=1e-11,atol=1e-20),'Fine history optical field differs')
    figure_path = resolve_run_path(path,summary['figure_directory']) if summary.get('figure_directory') else figure_directory(path)
    for filename in summary['figures']:
        require((figure_path/filename).is_file(),'Missing fine transient figure')


def transient_saved_validation_report(directory,summary,boundary_check=None,*,fem_validation=None):
    """Record initial, time, long-time and FEM evidence in the original report schema."""
    checks=saved_validation_checks(directory,summary)
    kind=summary.get('kind')
    if kind=='transient_joint_truncation_selection' or summary.get('transient_truncation_established',False):
        checks['joint_modal_truncation']['status']='passed'
    if kind=='transient_joint_truncation_selection' and summary.get('acceptance_version')==1:
        for name in ('initial_condition','thermal_boundaries','pde_residual','long_time_steady_limit'):
            rows=[next(c for c in case['physical_validation']['checks'] if c['name']==name) for case in summary['cases']]
            checks[name].update(status='passed',evidence='data/acceptance.npz',threshold=rows[0]['threshold'],
                metrics={'max_error':max(c['metrics']['max_error'] for c in rows)},
                scope='Recomputed selected-solution physical gates at recorded scope')
        for name in ('boundary_basis','energy_balance','early_pde_residual'):
            rows=[c for case in summary['cases'] for c in case['physical_validation']['checks'] if c['name']==name]
            if rows:
                checks[name]=validation_check(name,'passed',threshold=rows[0]['threshold'],
                    metrics={'max_error':max(c['metrics']['max_error'] for c in rows)},
                    evidence='data/acceptance.npz',scope=rows[0]['scope'])
        checks['initial_neighborhood'].update(status='passed',evidence='data/fields.npz',
            threshold=summary['tolerance'],scope=f'Finite-reference temperature/gradient/OPD/depth samples from {summary["times_s"][0]} s; not t->0+')
        if 'projection_tolerance' in summary['acceptance_config']:
            checks['projection_quadrature_convergence'].update(status='passed',evidence='data/acceptance.npz',
                threshold=summary['acceptance_config']['projection_tolerance'],scope='Recomputed fixed-root quadrature factors 1/2/4')
    if kind in ('transient_fine_fields','transient_modal_fem_study'):
        checks['initial_condition'].update(status='passed',metrics={'max_initial_temperature_K':0.,'max_initial_opd_m':0.},threshold=0.)
    long_errors=summary.get('long_time_errors_vs_same_modes_steady',summary.get('long_time_errors_vs_steady'))
    if long_errors is not None:
        checks['long_time_steady_limit'].update(metrics=long_errors,
            status='passed' if within_tolerance(long_errors,checks['long_time_steady_limit']['threshold']) else 'failed')
    checks['thermal_boundaries'].update(metrics=boundary_check or {},
        threshold=(boundary_check or {}).get('allowed_absolute_residual_W_per_m2'))
    if boundary_check is not None and boundary_check['status']=='PASS':
        checks['thermal_boundaries']['status']='passed'
    if kind=='transient_modal_fem_study':
        checks['fem_energy_balance']['status']='passed'
        checks['fem_weak_equation_residual']['status']='passed'
        if fem_validation is not None:
            prefix=f'theta{summary["angle_deg"]:g}_'
            for name,directions in (('fem_space_convergence',('xy','z','joint')),('fem_time_convergence',('time',))):
                rows=[c for c in fem_validation['checks'] if c['name'] in {prefix+d+'_convergence' for d in directions}]
                states=[row['status'] for row in rows]
                status=('failed' if 'failed' in states else
                    'passed' if len(rows)==len(directions) and all(s=='passed' for s in states) else 'not_run')
                checks[name].update(status=status,metrics={'reread_certificate_checks':rows},
                    evidence=summary.get('fem_convergence_directory'),
                    scope='Independent saved FEM certificate, including its diagnostic-only 1ms policy')
            checks['fem_independent_reference'].update(status=fem_validation['status'],
                evidence=summary.get('fem_convergence_directory'),scope='Independently reread FEM refinement certificate')
    if summary.get('mesh_convergence_established',False):
        checks['fem_space_convergence']['status']='passed'
    return validation_report(list(checks.values()),config={'physical_parameters_SI':summary.get('physical_parameters_SI'),
        'kind':kind,'scope':summary.get('precision_scope')},run_directory=directory)


def _transient_fem_case_header(directory,reference):
    """Read mesh/schedule metadata; large nodal arrays are sampled one case at a time."""
    source=resolve_run_path(directory,reference['source_run'])
    summary=read_json(source/'summary.json')
    for relative,sha in summary.get('birth_evidence_files',{}).items():
        require(file_sha256(resolve_run_path(source,relative))==sha,'FEM birth evidence hash differs')
    index=reference['case_index']
    require(isinstance(index,int) and not isinstance(index,bool) and index>=0,'Invalid FEM case index')
    rows=summary.get('fem_runs',summary.get('cases',[]))
    row=next((row for i,row in enumerate(rows) if row.get('index',i)==index),None)
    require(row is not None,'Missing FEM case diagnostics')
    diagnostics=row.get('diagnostics',row)
    file=(resolve_run_path(source,row['data_file']) if 'data_file' in row
          else data_directory(source)/f'fem_run{index}.npz')
    data_format=row.get('data_format','thermal_opd_fem')
    require(data_format in ('thermal_opd_fem','thermal_opd_probe_nodes_v1'),'Unsupported FEM data format')
    mesh_keys=(('disk_nodes_xy_m','disk_triangles','axial_nodes_z_m') if data_format=='thermal_opd_fem'
               else ('xy_vertices_m','triangles','z_dofs_m'))
    degree=diagnostics.get('field_degree_xy',1)
    require(degree in (1,2) and diagnostics.get('field_degree_z',degree)==degree,'Invalid FEM field degree')
    if row.get('data_sha256'):require(file_sha256(file)==row['data_sha256'],'Source FEM data hash differs')
    with read_numeric_data(file) as f:
        xy,tri,z=(f[k] for k in mesh_keys)
        times=f['times_s'] if 'times_s' in f else np.asarray(summary['times_s'])
        keys=f['step_keys'].tolist();values=f['steps']
        xy_dofs=f['xy_dof_coordinates_m'] if 'xy_dof_coordinates_m' in f else xy
        require('field_degree' not in f or f['field_degree'].item()==degree,'Saved FEM field degree differs')
    require(xy.ndim==2 and xy.shape[0]==2 and np.all(np.isfinite(xy)),'Invalid FEM xy nodes')
    require(tri.ndim==2 and tri.shape[0]==3 and tri.dtype.kind in 'iu'
        and tri.min()>=0 and tri.max()<xy.shape[1],'Invalid FEM triangles')
    require(z.ndim==1 and z.size>=2 and np.all(np.isfinite(z)) and np.all(np.diff(z)>0),'Invalid FEM z nodes')
    require(xy_dofs.ndim==2 and xy_dofs.shape[0]==2 and np.all(np.isfinite(xy_dofs)),'Invalid FEM DOF coordinates')
    require(times.ndim==1 and times.size>=2 and times[0]==0 and np.all(np.isfinite(times))
        and np.all(np.diff(times)>0),'Saved FEM must include zero and increasing times')
    require(len(keys)==len(set(keys)) and values.ndim==2 and values.shape[1]==len(keys)
        and values.shape[0]>0 and np.all(np.isfinite(values)),'Invalid FEM step data')
    steps={key:values[:,i] for i,key in enumerate(keys)}
    required={'time_s','dt_s','input_W','storage_rate_W','stored_energy_J','heat_loss_W',
              'balance_absolute_error_W','relative_linear_residual'}
    require(required<=set(steps),'Missing FEM step evidence')
    require(np.all(steps['dt_s']>0) and steps['time_s'][-1]==times[-1]
        and np.allclose(np.diff(np.r_[0.,steps['time_s']]),steps['dt_s'],rtol=1e-10,atol=1e-12),
        'Invalid FEM integration schedule')
    require(np.all(steps['relative_linear_residual']>=0),'Invalid FEM linear residual')
    p=PhysicalParameters(**summary['physical_parameters_SI'])
    require(z[0]==0 and z[-1]==p.thickness,'FEM axial domain differs from physics')
    require(diagnostics['axial_layers']*degree==z.size-1
        and diagnostics['degrees_of_freedom']==z.size*xy_dofs.shape[1],'FEM mesh diagnostics differ from saved nodes')
    if degree==2:
        from skfem import MeshTri,Basis,ElementTriP2
        require(diagnostics.get('geometry_degree',1)==1,'Only affine P2 polygon geometry is supported')
        basis=Basis(MeshTri(xy,tri),ElementTriP2())
        require(xy_dofs.shape==basis.doflocs.shape and np.allclose(xy_dofs,basis.doflocs,rtol=0,atol=1e-14*p.radius),
            'Saved P2 DOF coordinates differ from quadratic mesh')
        require(np.allclose(z[1::2],(z[:-2:2]+z[2::2])/2,rtol=1e-14,atol=0),'Invalid P2 axial midpoint DOFs')
        require(diagnostics['quadrature_order']>=4,'P2 mass/Robin quadrature is underintegrated')
    method=diagnostics['time_integrator']
    require(method in ('backward_euler','bdf2'),'Unsupported FEM time integrator')
    if method=='bdf2':
        extra={'bdf_storage_rate_W','bdf_balance_absolute_error_W','bdf_alpha0','bdf_alpha1','bdf_alpha2','time_order'}
        require(extra<=set(steps),'Missing BDF2 evidence')
        require(diagnostics.get('bdf2_startup')=='backward_euler'
            and diagnostics.get('bdf2_max_step_ratio')==1.5,'Unsupported BDF2 startup/ratio policy')
        dt=steps['dt_s'];prior=np.r_[dt[0],dt[:-1]]
        restart=np.r_[True,(dt[1:]/dt[:-1])>1.5*(1+16*np.finfo(float).eps)]
        require(np.array_equal(steps['time_order'],np.where(restart,1,2)),'Invalid BDF2 method order')
        a0,a1,a2=(steps['bdf_alpha'+str(i)] for i in range(3))
        require(np.all(a0[restart]==1.) and np.all(a1[restart]==-1.) and np.all(a2[restart]==0.),
            'Invalid BDF2 startup/restart coefficients')
        # Independent polynomial exactness at t_n, t_n-h, t_n-h-p; no solver helper.
        use=~restart;offset=1+prior[use]/dt[use]
        require(np.allclose(a0[use]+a1[use]+a2[use],0.,rtol=0,atol=1e-12)
            and np.allclose(-a1[use]-a2[use]*offset,1.,rtol=0,atol=1e-12)
            and np.allclose(a1[use]+a2[use]*offset**2,0.,rtol=0,atol=1e-12),
            'BDF2 coefficients do not differentiate quadratic polynomials correctly')
    else:
        require(not any(key.startswith('bdf_') for key in steps),'BE metadata contains BDF2 evidence')
    a,b=xy[:,tri[1]]-xy[:,tri[0]],xy[:,tri[2]]-xy[:,tri[0]]
    areas=np.abs(a[0]*b[1]-a[1]*b[0])/2
    require(np.all(areas>0),'Degenerate FEM triangle')
    edges=mesh_boundary_edges(tri)
    require(np.all(np.linalg.norm(xy,axis=0)<=p.radius*(1+1e-12))
        and np.allclose(np.linalg.norm(xy[:,np.unique(edges)],axis=0),p.radius,rtol=1e-12,atol=1e-14),
        'FEM polygon differs from the physical cylinder radius')
    lengths=np.linalg.norm(xy[:,edges[:,1]]-xy[:,edges[:,0]],axis=0)
    area_weights=np.bincount(tri.ravel(),weights=np.tile(areas/3,3),minlength=xy.shape[1])
    side_weights=np.bincount(edges.ravel(),weights=np.repeat(lengths/2,2),minlength=xy.shape[1])
    z_weights=None
    if degree==2:
        area_weights,side_weights,z_weights=quadratic_fem_integral_weights(xy,tri,z,xy_dofs)
    max_edge=max(float(np.max(np.linalg.norm(xy[:,tri[j]]-xy[:,tri[k]],axis=0))) for j,k in ((0,1),(1,2),(2,0)))
    interval_dt=[];observation_steps=[]
    for left,right in zip(times[:-1],times[1:]):
        ids=np.flatnonzero((steps['time_s']>left)&(steps['time_s']<=right))
        require(ids.size>0 and steps['time_s'][ids[-1]]==right,'FEM missed observation time')
        interval_dt.append(float(np.max(steps['dt_s'][ids])));observation_steps.append(int(ids[-1]))
    return dict(source=source,file=file,summary=summary,diagnostics=diagnostics,p=p,xy=xy,tri=tri,z=z,
        times=times,steps=steps,areas=areas,area_weights=area_weights,side_weights=side_weights,
        max_xy_edge_m=max_edge,interval_dt=np.array(interval_dt),observation_steps=observation_steps,
        sha256=file_sha256(file),field_degree=degree,xy_dof_coordinates=xy_dofs,z_integration_weights=z_weights)


def _transient_fem_case_samples(header,tx,ty,ox,oy,depths,internal):
    from thermal_opd.fem import FemSolution
    from skfem import MeshTri,Basis,ElementTriP1,ElementTriP2
    p=header['p'];mesh=MeshTri(header['xy'],header['tri']);times=header['times']
    with read_numeric_data(header['file']) as f:
        nodal=f['temperature_K']
        if nodal.ndim==2:
            require(times.size==2 and 'temperature_initial_K' in f,'Missing historical FEM initial nodes')
            nodal=np.stack((f['temperature_initial_K'],nodal))
        require(nodal.shape==(times.size,header['z'].size,header['xy_dof_coordinates'].shape[1])
            and np.all(np.isfinite(nodal)),'Invalid saved FEM temperatures')
        temperature,opd,energy,loss=[],[],[],[]
        for i,field in enumerate(nodal):
            frame=FemSolution(mesh,header['z'],field,{},field_degree=header['field_degree'])
            temperature.append(np.stack([frame.temperature_at(tx,ty,z) for z in depths[i]]))
            opd.append(frame.refracted_opd(ox,oy,internal,p.dn_dT,point_chunk_size=128))
            # Integral of the actual linear shape functions. These weights only
            # integrate saved fields; they do not lump the solver's capacity.
            planar=field@header['area_weights']
            if header['field_degree']==1:
                energy.append(p.density*p.specific_heat*np.trapezoid(planar,x=header['z']))
                loss.append(p.h_eff*(planar[0]+planar[-1]+np.trapezoid(field@header['side_weights'],x=header['z'])))
            else:
                weights=header['z_integration_weights']
                energy.append(p.density*p.specific_heat*(weights@planar))
                loss.append(p.h_eff*(planar[0]+planar[-1]+weights@(field@header['side_weights'])))
        initial=float(np.max(np.abs(nodal[0])))
    require(file_sha256(header['file'])==header['sha256'],
            'Source FEM data changed during validation')
    basis=Basis(mesh,ElementTriP1() if header['field_degree']==1 else ElementTriP2(),intorder=header['diagnostics']['quadrature_order'])
    xyq=basis.global_coordinates()
    power=float(np.sum(oblique_gaussian_heat_flux(xyq[0],xyq[1],p.beam_radius,
        np.deg2rad(header['summary']['angle_deg']),p.absorbed_power)*basis.dx))
    require(power>0,'Validation requires nonzero prescribed Gaussian power')
    steps=header['steps'];ids=header['observation_steps'];energy=np.asarray(energy);loss=np.asarray(loss)
    stored_rate=np.diff(np.r_[energy[0],steps['stored_energy_J']])/steps['dt_s']
    history=PowerHistory(**header['summary']['history'])
    expected_input=power*np.array([history.value(t) for t in steps['time_s']])
    nodal_energy_ratio=np.max(abs(energy[1:]-steps['stored_energy_J'][ids])/(1e-12+1e-10*abs(steps['stored_energy_J'][ids])))
    storage_ratio=np.max(abs(stored_rate-steps['storage_rate_W'])/(1e-12+1e-8*abs(steps['storage_rate_W'])))
    energy_error=max(float(np.max(abs(stored_rate+steps['heat_loss_W']-expected_input))/power),
        float(np.max(abs(steps['storage_rate_W'][ids]+loss[1:]-expected_input[ids]))/power))
    loss_ratio=float(np.max(abs(loss[1:]-steps['heat_loss_W'][ids])/(1e-12+1e-9*abs(steps['heat_loss_W'][ids]))))
    load_ratio=float(np.max(abs(steps['input_W']-expected_input)/(1e-14+1e-12*abs(expected_input))))
    load_ratio=max(load_ratio,float(abs(header['diagnostics']['full_input_W']-power)/(1e-14+1e-12*power)))
    recorded_balance=abs(steps['storage_rate_W']+steps['heat_loss_W']-steps['input_W'])
    balance_ratio=float(np.max(abs(steps['balance_absolute_error_W']-recorded_balance)/(1e-12+1e-8*recorded_balance)))
    if header['diagnostics']['time_integrator']=='bdf2':
        previous=np.r_[energy[0],steps['stored_energy_J'][:-1]]
        older=np.r_[energy[0],previous[:-1]]
        bdf_rate=(steps['bdf_alpha0']*(steps['stored_energy_J']-previous)
                  +steps['bdf_alpha2']*(older-previous))/steps['dt_s']
        bdf_ratio=float(np.max(abs(bdf_rate-steps['bdf_storage_rate_W'])
            /(1e-12+1e-8*abs(steps['bdf_storage_rate_W']))))
        require(bdf_ratio<=1.,'Recorded BDF2 energy derivative differs from saved energies')
        bdf_balance=abs(steps['bdf_storage_rate_W']+steps['heat_loss_W']-steps['input_W'])
        bdf_balance_ratio=float(np.max(abs(steps['bdf_balance_absolute_error_W']-bdf_balance)
            /(1e-12+1e-8*bdf_balance)))
        require(bdf_balance_ratio<=1.,'Recorded BDF2 balance error differs from step evidence')
        balance_ratio=max(balance_ratio,bdf_balance_ratio)
        storage_ratio=max(float(storage_ratio),bdf_ratio)
        energy_error=max(float(np.max(abs(bdf_rate+steps['heat_loss_W']-expected_input))/power),
            float(np.max(abs(steps['bdf_storage_rate_W'][ids]+loss[1:]-expected_input[ids]))/power))
    return dict(temperature=np.array(temperature),opd=np.array(opd),energy=energy,loss=loss,
        initial=initial,energy_error=energy_error,nodal_energy_ratio=float(nodal_energy_ratio),
        storage_ratio=float(storage_ratio),loss_ratio=loss_ratio,load_ratio=load_ratio,
        balance_ratio=balance_ratio,linear_residual=float(np.max(steps['relative_linear_residual'])))


def transient_fem_convergence_report(directory,manifest):
    """Recompute four isolated/joint refinement gates from independent saved FEM.

    Missing references/coverage are not_run. P1 FEM Robin boundaries are assessed
    in their weak form and via independent surface integrals, not pointwise flux.
    Thresholds and early/late temperature normalizations retain existing rules.
    New manifests can record 1ms as diagnostic-only; missing policy replays the
    historical gates, and all-step physical checks always remain mandatory.
    """
    from thermal_opd.fem import FemSolution
    from skfem import MeshTri
    from thermal_opd.opd import refracted_angle,transmitted_ray_mask
    require(manifest.get('tolerance',1e-3)==1e-3,'Legacy single FEM tolerance must remain 0.1%; use explicit space/joint/time limits')
    # The legacy single tolerance retains its original meaning when replaying
    # old reports. New runs record the approved space/joint/time limits apart.
    space_tolerance=manifest.get('space_tolerance',manifest.get('tolerance',1e-3))
    time_tolerance=manifest.get('time_tolerance',manifest.get('tolerance',1e-3))
    joint_tolerance=manifest.get('joint_tolerance',space_tolerance)
    require(space_tolerance in (1e-3,5e-3) and joint_tolerance==space_tolerance,
        'Supported FEM spatial policies are legacy 0.1% or approved 0.5%, with the same joint limit')
    require(time_tolerance==1e-3,'FEM time tolerance must remain 0.1%')
    require(manifest.get('sanity_tolerance',1e-7)==1e-7,'FEM sanity tolerance must remain 1e-7')
    requested=np.asarray(manifest['times_s'],dtype=float);angles=manifest['angles_deg']
    require(requested.ndim==1 and requested.size>=2 and requested[0]==0
        and np.all(np.isfinite(requested)) and np.all(np.diff(requested)>0),'Invalid requested FEM times')
    # Missing policy replays historical reports exactly. New CLI runs record
    # the user's explicit 1ms diagnostic-only policy in their manifest.
    diagnostic_times=np.asarray(manifest.get('diagnostic_only_times_s',[]),dtype=float)
    require(diagnostic_times.ndim==1 and (diagnostic_times.size==0 or
        (diagnostic_times.size==1 and diagnostic_times[0]==.001)),
        'Only 0.001 s may be excluded as a diagnostic convergence time')
    gate_requested=requested[~np.isin(requested,diagnostic_times)]
    require(angles and len(set(angles))==len(angles) and all(np.isfinite(a) and abs(a)<90 for a in angles),
            'Invalid requested FEM angles')
    cases=manifest['cases'];case_angles=[c['angle_deg'] for c in cases]
    require(len(set(case_angles))==len(cases) and set(case_angles)<=set(angles),'Invalid FEM case angles')
    nt=manifest.get('temperature_xy_points',201);no=manifest.get('opd_xy_points',101)
    require(all(isinstance(n,int) and not isinstance(n,bool) and n>=5 for n in (nt,no)),'Invalid sampling resolution')
    checks=[];fields={};hashes={};coverage=set(case_angles)==set(angles)
    roles=('candidate','xy_reference','z_reference','time_reference','joint_reference')
    for case in cases:
        angle=case['angle_deg'];label=f'theta{angle:g}'
        require(isinstance(case.get('candidate'),dict),'Missing candidate FEM case')
        headers={role:_transient_fem_case_header(directory,case[role]) for role in roles if case.get(role) is not None}
        base=headers['candidate'];p=base['p'];times=base['times']
        coverage &= all(np.all(np.isin(gate_requested,h['times'])) for h in headers.values())
        for role,h in headers.items():
            require(h['summary']['angle_deg']==angle,'Source FEM angle differs from manifest')
            require(h['summary']['physical_parameters_SI']==base['summary']['physical_parameters_SI']
                and h['summary']['history']==base['summary']['history'],'FEM references have different physics/history')
            require(np.array_equal(h['times'],times),'FEM references cover different observation times')
            hashes[label+'_'+role]=dict(source_run=str(h['source']),case_index=case[role]['case_index'],
                sha256=h['sha256'],summary_sha256=file_sha256(h['source']/'summary.json'))
        internal=refracted_angle(np.deg2rad(angle),p.incident_refractive_index,p.substrate_refractive_index)
        ta=np.linspace(-p.radius,p.radius,nt);tx,ty=np.meshgrid(ta,ta);tm=tx*tx+ty*ty<=p.radius**2
        oa=np.linspace(-.9*p.radius,.9*p.radius,no);ox,oy=np.meshgrid(oa,oa)
        om=(ox*ox+oy*oy<=(.9*p.radius)**2)&transmitted_ray_mask(ox,oy,p.radius,p.thickness,internal)
        for h in headers.values():
            mesh=MeshTri(h['xy'],h['tri'])
            geometry=FemSolution(mesh,np.array([0.,p.thickness]),np.zeros((2,mesh.nvertices)),{})
            tm &= geometry.transmitted_mask(tx,ty,0.);om &= geometry.transmitted_mask(ox,oy,internal)
        require(tm.sum()>1 and om.sum()>1,'Insufficient common thermal/optical points')
        depths=np.array([np.r_[0.,p.thickness/2,p.thickness,
            np.clip(np.array([0.,.25,.5,1.,2.,4.,8.])*np.sqrt(p.thermal_diffusivity*t),0.,p.thickness)] for t in times])
        fields.update({label+'_times_s':times,label+'_temperature_x_m':tx[tm],label+'_temperature_y_m':ty[tm],
            label+'_opd_x_m':ox[om],label+'_opd_y_m':oy[om],label+'_temperature_z_by_time_m':depths})
        samples={}
        for role,h in headers.items():
            samples[role]=_transient_fem_case_samples(h,tx[tm],ty[tm],ox[om],oy[om],depths,internal)
            for key,unit in (('temperature','temperature_K'),('opd','opd_m'),('energy','nodal_energy_J'),('loss','nodal_loss_W')):
                fields[label+'_'+role+'_'+unit]=samples[role][key]
        def add(name,passed,metrics,threshold,scope):
            checks.append(validation_check(label+'_'+name,'passed' if passed else 'failed',metrics=metrics,
                threshold=threshold,scope=scope,evidence='data/comparison_fields.npz'))
        states=[h['summary']['status'] for h in headers.values()]
        state='passed' if all(s=='COMPLETE' for s in states) else ('failed' if 'FAILED' in states else 'not_run')
        checks.append(validation_check(label+'_source_completion',state,metrics={'source_states':states},
            scope='Saved source runs must be complete; incomplete/failed source data are not a final certificate'))
        initial=max(s['initial'] for s in samples.values())
        add('initial_condition',initial==0,{'max_initial_node_temperature_K':initial},0.,'All saved FEM nodes at t=0')
        energy=max(s['energy_error'] for s in samples.values());storage=max(s['storage_ratio'] for s in samples.values())
        nodal=max(s['nodal_energy_ratio'] for s in samples.values())
        balance=max(s['balance_ratio'] for s in samples.values())
        add('energy_balance',energy<1e-7 and storage<=1 and nodal<=1 and balance<=1,
            {'max_input_normalized_error':energy,'storage_rate_consistency_ratio':storage,
             'nodal_storage_consistency_ratio':nodal,'balance_diagnostic_consistency_ratio':balance},
            1e-7,'Recorded secant storage and method-specific discrete energy derivative; independent Robin losses at saved observations')
        linear=max(s['linear_residual'] for s in samples.values())
        add('weak_equation_residual',linear<1e-7,{'max_recorded_relative_residual':linear},1e-7,
            'Solver-recorded discrete weak residual at all steps, including natural Robin boundaries; not pointwise flux')
        loss=max(s['loss_ratio'] for s in samples.values())
        add('robin_loss_diagnostics',loss<=1,{'max_consistency_ratio':loss},1.,
            'Independent nodal boundary integral vs recorded loss; original rtol=1e-9 and atol=1e-12 W retained')
        load=max(s['load_ratio'] for s in samples.values())
        add('prescribed_load',load<=1,{'max_consistency_ratio':load},1.,
            'Independent XY Gaussian quadrature and power history vs all saved step loads; rtol=1e-12, atol=1e-14 W')
        if diagnostic_times.size:
            missing={role:diagnostic_times[~np.isin(diagnostic_times,h['times'])].tolist()
                for role,h in headers.items()}
            checks.append(validation_check(label+'_diagnostic_recording',
                'not_run' if any(missing.values()) else 'passed',
                metrics={'diagnostic_only_times_s':diagnostic_times.tolist(),'missing_times_by_role_s':missing},
                scope='Required saved diagnostic frames; their numerical differences do not gate convergence',
                evidence='data/comparison_fields.npz'))
        for direction in ('xy','z','time','joint'):
            role=direction+'_reference';name=label+'_'+direction+'_convergence'
            tolerance=time_tolerance if direction=='time' else (joint_tolerance if direction=='joint' else space_tolerance)
            if role not in headers:
                checks.append(validation_check(name,'not_run',threshold=tolerance,scope='Missing '+role));continue
            ref=headers[role]
            same_xy=np.array_equal(base['xy'],ref['xy']) and np.array_equal(base['tri'],ref['tri'])
            same_z=np.array_equal(base['z'],ref['z'])
            same_dt=np.array_equal(base['steps']['time_s'],ref['steps']['time_s']) and np.array_equal(base['steps']['dt_s'],ref['steps']['dt_s'])
            finer_xy=ref['xy'].shape[1]>base['xy'].shape[1] and ref['tri'].shape[1]>base['tri'].shape[1] and ref['max_xy_edge_m']<base['max_xy_edge_m']
            finer_z=ref['z'].size>base['z'].size and np.max(np.diff(ref['z']))<np.max(np.diff(base['z']))
            finer_dt=bool(np.all(ref['interval_dt']<base['interval_dt']))
            isolated={'xy':finer_xy and same_z and same_dt,'z':same_xy and finer_z and same_dt,
                'time':same_xy and same_z and finer_dt,'joint':finer_xy and finer_z and finer_dt}[direction]
            isolated=bool(isolated and base['diagnostics']['quadrature_order']==ref['diagnostics']['quadrature_order']
                          and base['diagnostics']['time_integrator']==ref['diagnostics']['time_integrator']
                          and base['field_degree']==ref['field_degree'])
            errors=[]
            ids=np.flatnonzero((times>0)&(np.isin(times,requested)|np.isin(times,diagnostic_times)))
            check_times=times[ids];used_for_gate=~np.isin(check_times,diagnostic_times)
            for j in ids:
                t=times[j]
                errors.append(field_errors(samples['candidate']['temperature'][j],samples[role]['temperature'][j],np.ones(tm.sum()),
                    samples['candidate']['opd'][j],samples[role]['opd'][j],temperature_normalization='global' if t<.625 else 'per_plane'))
            gate_errors=[e for e,used in zip(errors,used_for_gate) if used]
            passed=isolated and all(within_tolerance(e,tolerance) for e in gate_errors)
            status=('passed' if passed else 'failed') if gate_errors else 'not_run'
            metrics={'isolated_parameter_change':isolated,'times_s':check_times.tolist(),
                'temperature_normalization_by_time':['global' if t<.625 else 'per_plane' for t in check_times],
                'errors_by_time':errors,'temperature_points':int(tm.sum()),'opd_points':int(om.sum())}
            if diagnostic_times.size:
                metrics.update(used_for_gate_by_time=used_for_gate.tolist(),
                    diagnostic_only_times_s=diagnostic_times.tolist(),
                    diagnostic_times_not_recorded_s=diagnostic_times[~np.isin(diagnostic_times,times)].tolist())
            checks.append(validation_check(name,status,threshold=tolerance,
                metrics=metrics,
                scope='Same common full thermal disk, three planes and moving diffusion depths; separate refracted optical pupil',
                evidence='data/comparison_fields.npz'))
    scope_metrics={'requested_angles_deg':angles,'requested_times_s':requested.tolist()}
    if diagnostic_times.size:
        scope_metrics.update(acceptance_times_s=gate_requested.tolist(),diagnostic_only_times_s=diagnostic_times.tolist())
    checks.append(validation_check('requested_scope','passed' if coverage else 'not_run',
        metrics=scope_metrics,
        scope=('Every requested angle and acceptance time must be present; 0.001 s is recorded as diagnostic only'
            if diagnostic_times.size else 'Every requested angle and time must be present in all provided comparisons')))
    report=validation_report(checks,config=manifest,run_directory=directory)
    report.update(fem_convergence_established=report['status']=='passed',evidence_hashes=hashes,
        limitations=['Finite recorded time/depth/XY sampling; no claim for every continuous point or t approaching zero.',
            'Robin boundaries assessed in weak form and integral balance, not pointwise normal flux.',
            'Gaussian load quadrature order is fixed during refinement; its separate convergence is not established here.',
            'This FEM certificate does not establish semi-analytical truncation or cross-method agreement.'])
    if diagnostic_times.size:
        report['limitations'].append('0.001 s field changes are recorded as diagnostics and excluded from refinement gates; all-step physical checks remain required.')
    return relative_metadata_paths(report,directory),fields


def _transient_fem_records_equal(actual,stored):
    if isinstance(actual,dict):
        return isinstance(stored,dict) and actual.keys()==stored.keys() and all(_transient_fem_records_equal(v,stored[k]) for k,v in actual.items())
    if isinstance(actual,list):
        return isinstance(stored,list) and len(actual)==len(stored) and all(_transient_fem_records_equal(a,b) for a,b in zip(actual,stored))
    if isinstance(actual,(int,float)) and not isinstance(actual,bool):
        return isinstance(stored,(int,float)) and not isinstance(stored,bool) and np.isclose(actual,stored,rtol=1e-8,atol=1e-12)
    return actual==stored


def validate_saved_transient_fem_convergence(path,summary):
    report,fields=transient_fem_convergence_report(path,summary['config'])
    stored=read_json(path/'verification.json')
    require(report['evidence_hashes']==stored['evidence_hashes'],'FEM source evidence changed')
    require(report['status']==stored['status'] and report['counts']==stored['counts']
        and _transient_fem_records_equal(report['checks'],stored['checks']),'Saved FEM convergence metric/status differs')
    require(summary['fem_convergence_established']==report['fem_convergence_established'],
            'FEM convergence flag differs from complete scientific gates')
    with read_numeric_data(data_directory(path)/'comparison_fields.npz') as saved:
        require(set(saved.files)==set(fields),'Missing FEM convergence sampled field')
        for key,value in fields.items():
            atol=1e-20 if key.endswith('opd_m') else 1e-12
            require(saved[key].shape==value.shape and np.allclose(saved[key],value,rtol=1e-10,atol=atol),
                    f'Saved FEM convergence field differs: {key}')
    return saved_validation_result(summary,report)


def load_transient_fem_certificate(directory,parameters,angle,history,times,*,repository_root=None):
    """Independently review a formal certificate and restore its candidate nodes.

    This consumer reuses the original FEM interpolation and refracted ray
    integrator. It neither solves a PDE nor copies the source nodal archive.
    Solver-source hashes describe applicability; source metadata/hashes retain
    the actual birth lineage of reused, independently checked probe nodes.
    """
    from skfem import MeshTri
    from thermal_opd.fem import FemSolution,TransientFemSolution
    directory=Path(directory).resolve();summary=read_json(directory/'summary.json')
    summary_hash=file_sha256(directory/'summary.json');verification_hash=file_sha256(directory/'verification.json')
    require(summary.get('kind')=='transient_fem_convergence_validation'
        and summary.get('status')=='COMPLETE','Need a complete FEM convergence certificate')
    reviewed=validate_saved_transient_fem_convergence(directory,summary)
    report=reviewed['verification']
    require(summary.get('fem_convergence_established') is True
        and reviewed.get('fem_convergence_established') is True
        and report.get('status')=='passed' and report.get('fem_convergence_established') is True,
        'Formal FEM convergence certificate must be independently reread and passed')
    manifest=summary['config'];requested=np.asarray(times,dtype=float)
    require(angle in manifest['angles_deg'],'Requested angle is outside FEM certificate scope')
    require(np.all(np.isin(requested,manifest['times_s'])),'Requested times are outside FEM certificate scope')
    root=Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    expected={f'src/thermal_opd/{name}' for name in ('fem.py','parameters.py','heat_source.py','transient.py')}
    hashes={str(k).replace('\\','/'):v for k,v in manifest.get('solver_source_sha256',{}).items()}
    require(expected<=hashes.keys(),'FEM certificate is missing solver source applicability hashes')
    require(all(file_sha256(root/key)==hashes[key] for key in expected),
        'FEM certificate solver source applicability differs from current production source')
    case=next(case for case in manifest['cases'] if case['angle_deg']==angle)
    reference=case['candidate'];header=_transient_fem_case_header(directory,reference)
    evidence=report['evidence_hashes'][f'theta{angle:g}_candidate']
    require(header['sha256']==evidence['sha256']
        and file_sha256(header['source']/'summary.json')==evidence['summary_sha256'],
        'FEM candidate evidence changed after independent certificate review')
    require(header['summary']['physical_parameters_SI']==asdict(parameters),'FEM certificate physical parameters differ')
    require(header['summary']['history']==asdict(history),'FEM certificate history differs')
    require(header['summary']['angle_deg']==angle,'FEM candidate angle differs')
    require(header['field_degree']==2,'Formal FEM candidate requires quadratic P2 fields')
    require(np.all(np.isin(requested,header['times'])),'FEM candidate is missing requested times')
    indices=[int(np.flatnonzero(header['times']==t)[0]) for t in requested]
    mesh=MeshTri(header['xy'],header['tri'])
    with read_numeric_data(header['file']) as saved:
        nodal=saved['temperature_K']
        require(nodal.shape==(header['times'].size,header['z'].size,header['xy_dof_coordinates'].shape[1])
            and np.all(np.isfinite(nodal)),'Invalid formal FEM nodal fields')
        frames=[FemSolution(mesh,header['z'],nodal[index],header['diagnostics'],field_degree=2) for index in indices]
    require(file_sha256(header['file'])==header['sha256'],'Source FEM data changed during candidate restoration')
    require(file_sha256(directory/'summary.json')==summary_hash
        and file_sha256(directory/'verification.json')==verification_hash,'FEM certificate changed during candidate restoration')
    steps=[dict(zip(header['steps'],values)) for values in zip(*header['steps'].values())]
    solution=TransientFemSolution(requested,frames,steps,header['diagnostics'])
    source_rows=header['summary'].get('fem_runs',header['summary'].get('cases',[]))
    row=next(row for i,row in enumerate(source_rows) if row.get('index',i)==reference['case_index'])
    return dict(header=header,solution=solution,verification=report,
        certificate_directory=str(directory),certificate_summary_sha256=summary_hash,
        certificate_verification_sha256=verification_hash,
        source_run=str(header['source']),case_index=reference['case_index'],data_file=str(header['file']),
        data_format=row.get('data_format','thermal_opd_fem'),data_sha256=header['sha256'])


def validate_saved_transient_run(directory,summary=None):
    """Reload transient fields, recompute gates and independently inspect boundaries."""
    path=Path(directory)
    summary=read_json(path/'summary.json') if summary is None else summary
    kind=summary.get('kind')
    require(summary.get('status')=='COMPLETE','Run is not complete')
    if kind=='transient_fem_convergence_validation':
        return validate_saved_transient_fem_convergence(path,summary)
    fem_validation=None
    if kind=='transient_modal_fem_study':
        fem_validation=validate_transient_fem(path,summary)
    elif kind=='transient_joint_truncation_selection':
        validate_transient_selection(path,summary)
    elif kind=='transient_fine_fields':
        validate_transient_fields(path,summary)
    else:
        raise ValueError(f'Unsupported run kind: {kind}')
    boundary_check=None
    if kind in ('transient_modal_fem_study','transient_fine_fields'):
        boundary_check=validate_transient_boundary_check(path,summary)
    return saved_validation_result(summary,transient_saved_validation_report(path,summary,boundary_check,
        fem_validation=fem_validation),boundary_check)
