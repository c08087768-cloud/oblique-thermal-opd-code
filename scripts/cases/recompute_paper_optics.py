"""Rebuild manuscript optical evidence after the approved material-input change.

Uses the existing SA and independently assembled P2 FEM implementations. It
never alters a historical run. Run stages separately in an existing isolated
outputs/optical_parameter_update directory; completed stage files are not
overwritten. No solver or acceptance criterion is reimplemented here.

Controls: test_optical_material_and_omission.py, test_opd.py, test_fem_bdf2.py
and test_fem_p2.py. The SA checks compare finite modal references and doubled
ray quadrature. Historical TEMPERATURE refinement is inherited separately;
new FEM optical samples are cross-method comparisons, not a new FEM optical
refinement certificate. See the resulting manifest for actual checks/scope.
Added 2026-11-03 for the author-approved matched optical inputs and Section5.3.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import csv
import importlib.util
import json
import time
import hashlib
import numpy as np
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.model import ThermalModel, ModalResolution, PreparedModel
from thermal_opd.opd import transmitted_ray_mask
from thermal_opd.metrics import azimuthal_omission_metrics
from thermal_opd.convergence import continuous_aperture_pv

ROOT = Path(__file__).resolve().parents[2]
OBS = np.array([0., .001, .0025, .01, .05, .1, .625, 2.5, 20., 50., 75.])
FINE = ModalResolution(10, 20, 1200, 128, 32)
STEADY = ModalResolution(6, 8, 400, 32, 16)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def grid(p, angle, n):
    axis = np.linspace(-.9*p.radius, .9*p.radius, n)
    x, y = np.meshgrid(axis, axis)
    mask = (x*x+y*y <= (.9*p.radius)**2) & transmitted_ray_mask(
        x, y, p.radius, p.thickness, angle)
    return x, y, mask


def discrepancy(values, reference):
    pv = np.ptp(reference, axis=-1)
    difference = values-reference
    difference -= difference.mean(axis=-1, keepdims=True)
    return dict(pv_relative=(np.abs(np.ptp(values,axis=-1)-pv)/pv).tolist(),
                rms_over_reference_pv=(np.sqrt(np.mean(difference**2,axis=-1))/pv).tolist(),
                linf_over_reference_pv=(np.max(np.abs(difference),axis=-1)/pv).tolist())


def sa_case(run, angle):
    folder = run/f'sa_theta{angle:g}'
    folder.mkdir(exist_ok=False)
    start = time.perf_counter()
    p = PhysicalParameters()
    with (ROOT/f'data/reference/theta{angle:g}_history.csv').open(encoding='utf-8') as f:
        old_history = [dict(row) for row in csv.DictReader(f)]
    times = np.unique(np.r_[OBS, [float(v['time_s']) for v in old_history
                                if 0 < float(v['time_s']) <= 75.000001]])
    prepared = ThermalModel(p, angle, FINE).prepare()
    model = prepared.solve_transient(num_axial_modes=256)
    x, y, mask = grid(p, prepared.refraction_angle, 201)
    ids = np.searchsorted(times, OBS)
    assert np.array_equal(times[ids], OBS)
    full = np.zeros((len(times), int(mask.sum())))
    saved_orders = []
    zero = None
    for m in prepared.coefficients:
        one = PreparedModel(p, angle, FINE, {m:prepared.coefficients[m]}, {m:prepared.roots[m]})
        one_model = one.solve_transient(num_axial_modes=256)
        values = one_model.opd_series(x[mask], y[mask], times, point_chunk_size=512)
        full += values
        if m == 0:
            zero = values.copy()
        saved_orders.append(values[ids])
        print('SA_ORDER', angle, m, round(time.perf_counter()-start,2), flush=True)
    history = []
    all_mask = np.ones(int(mask.sum()), dtype=bool)
    for i,t in enumerate(times):
        row = dict(time_s=float(t), sampled_pv_nm=float(np.ptp(full[i])*1e9))
        if t > 0:
            row.update(azimuthal_omission_metrics(full[i],zero[i],all_mask))
        history.append(row)
    dense = np.full((len(OBS),201,201), np.nan)
    dense[:,mask] = full[ids]
    dense_zero = np.full_like(dense, np.nan)
    dense_zero[:,mask] = zero[ids]
    order_fields = np.full((len(saved_orders),len(OBS),201,201),np.nan)
    order_fields[:,:,mask] = np.array(saved_orders)
    assert np.allclose(order_fields[:,:,mask].sum(axis=0),dense[:,mask],rtol=2e-13,atol=1e-20)
    # Independent whole-dictionary assembly: catches omitted/doubled orders.
    direct = model.opd_series(x[mask][::211],y[mask][::211],OBS,point_chunk_size=256)
    assert np.allclose(direct,dense[:,mask][:,::211],rtol=2e-12,atol=1e-20)
    sp = ThermalModel(p, angle, STEADY).prepare()
    steady = sp.solve_steady()
    steady_opd = np.full_like(x,np.nan)
    steady_opd[mask] = steady.opd(x[mask],y[mask],point_chunk_size=512)
    r = np.linspace(0,p.radius,161)
    phi = np.arange(180)*2*np.pi/180
    temp = np.stack([steady.temperature(r[:,None],phi[None,:],z) for z in [0,p.thickness/2,p.thickness]])
    # Re-estimate the same continuous-pupil extrema as the existing supplement.
    extrema = {}
    for label,at in [('steady',steady.prepare_opd(point_chunk_size=256)),
                     ('0.625',model.prepare_opd(.625,point_chunk_size=256)),
                     ('75',model.prepare_opd(75.,point_chunk_size=256))]:
        extrema[label] = continuous_aperture_pv(at,radius=p.radius,thickness=p.thickness,
            refraction_angle=prepared.refraction_angle,aperture_radius=.9*p.radius,
            num_seed_radius=9,num_seed_phi=48)
    np.savez_compressed(folder/'fields.npz',times_s=OBS,opd_x_m=x,opd_y_m=y,
        mask=mask,opd_m=dense,opd_m0_m=dense_zero,orders=np.array(list(prepared.coefficients)),
        order_opd_m=order_fields,steady_opd_m=steady_opd,temperature_r_m=r,
        temperature_phi_rad=phi,steady_temperature_K=temp)
    write(folder/'summary.json',dict(status='COMPLETE',angle_deg=angle,
        physical_parameters_SI=asdict(p),modal=asdict(FINE),steady_modal=asdict(STEADY),
        axial_modes=256,refraction_angle_rad=prepared.refraction_angle,
        history=history,continuous_extrema=extrema,steady_sampled_pv_nm=float(np.ptp(steady_opd[mask])*1e9),
        elapsed_seconds=time.perf_counter()-start,order_assembly_check=True))
    print('SA_COMPLETE',angle,round(time.perf_counter()-start,2),flush=True)


def check_sa(run,angle):
    p = PhysicalParameters()
    prepared = ThermalModel(p,angle,FINE).prepare()
    model = prepared.solve_transient(num_axial_modes=256)
    reference = ThermalModel(p,angle,ModalResolution(14,28,1800,256,64)).prepare()
    ref_model = reference.solve_transient(num_axial_modes=384)
    x,y,mask = grid(p,prepared.refraction_angle,101)
    times = OBS[1:]
    selected = model.opd_series(x[mask],y[mask],times,point_chunk_size=512)
    doubled_ray = model.opd_series(x[mask],y[mask],times,num_ray_points=64,point_chunk_size=512)
    ref = ref_model.opd_series(x[mask],y[mask],times,point_chunk_size=512)
    checks = dict(modal_reference=discrepancy(selected,ref),ray_quadrature=discrepancy(selected,doubled_ray))
    s=ThermalModel(p,angle,STEADY).prepare().solve_steady()
    checks['steady_reference']=discrepancy(s.opd(x[mask],y[mask])[None,:],
        prepared.solve_steady().opd(x[mask],y[mask])[None,:])
    # Same 0.1% finite-reference target; zero time has no relative norm.
    passed = all(max(values) <= .001 for group in checks.values() for values in group.values())
    write(run/f'sa_check_theta{angle:g}.json',dict(times_s=times.tolist(),checks=checks,
        relative_target=.001,passed=passed,scope='Finite-reference SA optical checks, not exact-solution error bounds'))
    if not passed:
        raise RuntimeError('New optical SA finite-reference check failed')
    print('SA_CHECK_PASS',angle,flush=True)


def sa_limits(run,angle):
    from thermal_opd.steady import axial_response
    from thermal_opd.opd import modal_refracted_opd_series
    p=PhysicalParameters()
    prepared=ThermalModel(p,angle,FINE).prepare()
    model=prepared.solve_transient(num_axial_modes=256)
    late=max(75.,12/model.slowest_decay_rate)
    response=model._response(late)
    x,y,mask=grid(p,prepared.refraction_angle,201)
    values=modal_refracted_opd_series(x[mask],y[mask],prepared.coefficients,prepared.roots,
        lambda m,n,z:np.stack([axial_response(z,prepared.roots[m][n],prepared.coefficients[m][n],
            p.conductivity,p.thickness,p.h_eff,p.h_eff)[0],response(m,n,z)]),
        2,p.thickness,p.dn_dT,radius=p.radius,refraction_angle=prepared.refraction_angle,
        num_ray_points=32,point_chunk_size=512)
    pvs=np.ptp(values,axis=1)
    error=float(abs(pvs[1]-pvs[0])/pvs[0])
    assert error<=1e-5
    np.savez_compressed(run/f'sa_limits_theta{angle:g}.npz',opd_x_m=x,opd_y_m=y,mask=mask,
        steady_opd_m=values[0],long_time_opd_m=values[1])
    write(run/f'sa_limits_theta{angle:g}.json',dict(long_time_s=late,steady_sampled_pv_nm=float(pvs[0]*1e9),
        long_time_sampled_pv_nm=float(pvs[1]*1e9),relative_pv_difference=error,passed=True))
    print('SA_LIMIT_PASS',angle,late,error,flush=True)


def original_fem_driver():
    spec=importlib.util.spec_from_file_location('original_fem_refinement',
        ROOT/'scripts/convergence/check_transient_fem_convergence.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fem_case(run,angle,role):
    from thermal_opd.run_io import create_run
    mod=original_fem_driver()
    spec=next(v for v in mod.make_plan('full',[angle]) if v['role']==role)
    folder=create_run(ROOT,'optical_parameter_update')
    write(run/f'fem_theta{angle:g}_{role}_path.json',dict(path=folder.relative_to(ROOT).as_posix()))
    mod.solve_case(ROOT,spec,folder)


def steady_fem_case(run,angle):
    from thermal_opd.fem import solve_steady_fem
    from thermal_opd.heat_source import oblique_gaussian_heat_flux
    p=PhysicalParameters()
    prepared=ThermalModel(p,angle,STEADY).prepare()
    steady=prepared.solve_steady()
    start=time.perf_counter()
    solution=solve_steady_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,
        lambda x,y:oblique_gaussian_heat_flux(x,y,p.beam_radius,np.deg2rad(angle),p.absorbed_power),
        disk_refinements=5,axial_layers=64,quadrature_order=10,solver='cg',rtol=1e-10,
        field_degree=2,mirror_reduction=True)
    x,y,mask=grid(p,prepared.refraction_angle,101)
    common=mask & solution.transmitted_mask(x,y,prepared.refraction_angle)
    sa=steady.opd(x[common],y[common],point_chunk_size=512)
    fe=solution.refracted_opd(x[common],y[common],prepared.refraction_angle,p.dn_dT,point_chunk_size=128)
    axis=np.linspace(-p.radius,p.radius,101)
    tx,ty=np.meshgrid(axis,axis)
    tm=(tx*tx+ty*ty<=p.radius**2)&solution.transmitted_mask(tx,ty,0.)
    ts=np.stack([steady.temperature(np.hypot(tx[tm],ty[tm]),np.arctan2(ty[tm],tx[tm]),z)
                 for z in [0,p.thickness/2,p.thickness]])
    tf=np.stack([solution.temperature_at(tx[tm],ty[tm],z)for z in [0,p.thickness/2,p.thickness]])
    temperature_linf=(np.max(abs(tf-ts),axis=1)/np.max(abs(ts),axis=1)).tolist()
    np.savez_compressed(run/f'steady_fem_theta{angle:g}.npz',opd_x_m=x,opd_y_m=y,
        mask=common,opd_sa_m=sa,opd_fem_m=fe,temperature_x_m=tx,temperature_y_m=ty,
        temperature_mask=tm,temperature_sa_K=ts,temperature_fem_K=tf,
        disk_nodes_xy_m=solution.disk_mesh.p,disk_triangles=solution.disk_mesh.t,
        axial_nodes_z_m=solution.z_m,nodal_temperature_K=solution.temperature_K)
    write(run/f'steady_fem_theta{angle:g}.json',dict(status='COMPLETE',angle_deg=angle,
        physical_parameters_SI=asdict(p),diagnostics=solution.diagnostics,
        temperature_relative_linf=temperature_linf,optical=discrepancy(fe[None,:],sa[None,:]),
        elapsed_seconds=time.perf_counter()-start))
    print('STEADY_FEM_COMPLETE',angle,round(time.perf_counter()-start,2),flush=True)


def supplement_sweeps(run):
    source=ROOT/('outputs/experiment_audit/20261017T121521368371Z_f107298c/data/snapshot/'
        'outputs/convergence_diagnostic/20261017T093231672765Z_4809a8b0_modal_sum_continuation_sweeps_a1/summary.json')
    original=json.loads(source.read_text(encoding='utf-8'))
    p=PhysicalParameters()
    cache={}
    result={}
    started=time.perf_counter()
    for name,sweep in original['sweeps'].items():
        cases=[]
        for old in sweep['cases']:
            config={k:old[k]for k in ('M','N','projection_r','projection_phi','opd_xy','ray_points')}
            key=tuple(config.values())
            if key not in cache:
                modal=ModalResolution(*(config[k]for k in ('M','N','projection_r','projection_phi','ray_points')))
                prepared=ThermalModel(p,60.,modal).prepare()
                x,y,mask=grid(p,prepared.refraction_angle,config['opd_xy'])
                values=prepared.solve_steady().opd(x[mask],y[mask],point_chunk_size=512)
                cache[key]=float(np.ptp(values))
            cases.append(dict(**config,varied_value=old['varied_value'],opd_pv_m=cache[key]))
        reference=cases[-1]['opd_pv_m']
        for row in cases:row['relative_pv_change_vs_reference']=abs(row['opd_pv_m']-reference)/reference
        result[name]=dict(varied_setting=sweep['varied_setting'],reference_settings=sweep['reference_settings'],cases=cases)
        print('SWEEP_COMPLETE',name,round(time.perf_counter()-started,2),flush=True)
    write(run/'supplement_sweeps.json',dict(status='COMPLETE',physical_parameters_SI=asdict(p),sweeps=result,
        historical_settings_source=source.relative_to(ROOT).as_posix(),
        historical_settings_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started))


def postprocess_fem(run,angle):
    from skfem import MeshTri, Basis, ElementTriP2
    from thermal_opd.fem import FemSolution
    from thermal_opd.convergence import field_errors
    from numpy.polynomial.legendre import leggauss
    link=json.loads((run/f'fem_theta{angle:g}_candidate_path.json').read_text())
    folder=ROOT/link['path']
    header=json.loads((folder/'summary.json').read_text())
    p=PhysicalParameters()
    assert header['status']=='COMPLETE' and header['physical_parameters_SI']==asdict(p)
    source=folder/'data/fem_run0.npz'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha==header['cases'][0]['data_sha256']
    assert header['maximum_relative_energy_error']<1e-7 and header['maximum_relative_linear_residual']<1e-7
    f=np.load(source,allow_pickle=False)
    assert np.array_equal(f['times_s'],OBS) and int(f['field_degree'])==2
    assert np.max(abs(f['temperature_K'][0]))==0
    mesh=MeshTri(f['disk_nodes_xy_m'],f['disk_triangles'])
    assert np.array_equal(Basis(mesh,ElementTriP2()).doflocs,f['xy_dof_coordinates_m'])
    frames=[FemSolution(mesh,f['axial_nodes_z_m'],values,{'time_s':float(t)},field_degree=2)
            for t,values in zip(OBS,f['temperature_K'])]
    for frame in frames[1:]:
        frame._finder,frame._origins,frame._inverse=frames[0]._finder,frames[0]._origins,frames[0]._inverse
    prepared=ThermalModel(p,angle,FINE).prepare()
    model=prepared.solve_transient(num_axial_modes=256)
    x,y,physical=grid(p,prepared.refraction_angle,101)
    common=physical & frames[0].transmitted_mask(x,y,prepared.refraction_angle)
    sa_physical=model.opd_series(x[physical],y[physical],OBS,point_chunk_size=512)
    sa_common=sa_physical[:,common[physical]]
    fem=[]
    for t,frame in zip(OBS,frames):
        fem.append(frame.refracted_opd(x[common],y[common],prepared.refraction_angle,p.dn_dT,point_chunk_size=128))
        print('FEM_NEW_RAY_INTEGRAL',angle,t,flush=True)
    fem=np.array(fem)
    nodes,weights=leggauss(201)
    r=(nodes+1)*p.radius/2
    phi=np.arange(201)*2*np.pi/201
    tx,ty=r[:,None]*np.cos(phi),r[:,None]*np.sin(phi)
    tm=frames[0].transmitted_mask(tx,ty,0.)
    aw=np.broadcast_to((weights*p.radius/2*r*2*np.pi/201)[:,None],tm.shape)[tm]
    errors=[]
    for i,(t,frame)in enumerate(zip(OBS,frames)):
        if t==0:errors.append(None);continue
        ts=np.stack([model.temperature(r[:,None],phi[None,:],z,t)[tm]for z in [0,p.thickness/2,p.thickness]])
        tf=np.stack([frame.temperature_at(tx[tm],ty[tm],z)for z in [0,p.thickness/2,p.thickness]])
        errors.append(field_errors(tf,ts,aw,fem[i],sa_common[i],
            temperature_normalization='global'if t<.625 else 'per_plane'))
    np.savez_compressed(run/f'new_opd_fem_theta{angle:g}.npz',times_s=OBS,
        opd_x_m=x,opd_y_m=y,physical_mask=physical,common_mask=common,
        opd_sa_physical_m=sa_physical,opd_sa_common_m=sa_common,opd_fem_m=fem)
    write(run/f'new_opd_fem_theta{angle:g}.json',dict(status='COMPLETE',angle_deg=angle,
        physical_parameters_SI=asdict(p),source_file=source.relative_to(ROOT).as_posix(),
        source_sha256=sha,times_s=OBS.tolist(),errors_vs_sa=errors,
        maximum_relative_energy_error=header['maximum_relative_energy_error'],
        maximum_relative_linear_residual=header['maximum_relative_linear_residual'],
        scope='Independent method-to-method differences on current refracted paths; no new FEM optical-refinement certificate'))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    f.close()
    print('FEM_OPTICAL_POSTPROCESS_COMPLETE',angle,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--angle',type=float,choices=(0.,30.,60.),required=True)
    parser.add_argument('--stage',choices=('sa','check-sa','sa-limits','fem','steady-fem','sweeps','postprocess-fem'),required=True)
    parser.add_argument('--role',default='candidate',choices=('candidate','xy_reference','z_reference','time_reference','joint_reference'))
    args=parser.parse_args()
    run=args.run_directory.resolve()
    run.relative_to(ROOT/'outputs/optical_parameter_update')
    if args.stage=='sa':sa_case(run,args.angle)
    elif args.stage=='check-sa':check_sa(run,args.angle)
    elif args.stage=='sa-limits':sa_limits(run,args.angle)
    elif args.stage=='steady-fem':steady_fem_case(run,args.angle)
    elif args.stage=='sweeps':supplement_sweeps(run)
    elif args.stage=='postprocess-fem':postprocess_fem(run,args.angle)
    else:fem_case(run,args.angle,args.role)


if __name__=='__main__':
    main()
