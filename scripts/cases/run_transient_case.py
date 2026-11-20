"""Transient/modal-field study on one angle; smoke is an implementation check.

Reuses source projections, steady temperature, refracted rays and prism FEM.
Saved mode OPDs are contributions from thermal orders, not a Fourier analysis
of the cropped entrance-plane optical field. Single-order PVs are not additive.
"""
from thermal_opd.run_io import script_defaults
from thermal_opd.model import ThermalModel, PreparedModel
from thermal_opd.run_io import data_directory
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
from thermal_opd.plotting import save_transient_diagnostic_figure
import numpy as np
from numpy.polynomial.legendre import leggauss

from thermal_opd.convergence import field_errors
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.model import ModalResolution
from thermal_opd.modal_evaluation import modal_temperature
from thermal_opd.opd import refracted_angle, transmitted_ray_mask
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.projection import reconstruct_cosine_field
from thermal_opd.run_io import create_run, provenance, write_summary
from thermal_opd.transient import PowerHistory
from thermal_opd.transient_validation import write_transient_boundary_check,load_transient_fem_certificate


parameters = PhysicalParameters()
# Steady-selected settings are a starting point, not a transient certificate.
modal_settings = ModalResolution(6,8,400,32,32)
reference_settings = ModalResolution(10,16,800,128,64)
axial_counts = (12,24,48)
observation_times_s = (0.,.625,2.5,20.,50.,75.)
formal_observation_times_s = (0.,.001,.0025,.01,.05,.1,.625,2.5,20.,50.,75.)
long_time_decay_exponent = 12.
temperature_r_points, temperature_phi_points = 32,48
opd_xy_points = 25
formal_temperature_r_points,formal_temperature_phi_points,formal_opd_xy_points = 201,201,101
aperture_radius_ratio = .9
smoke_meshes = ((2,8),(3,12))
smoke_time_steps_s = (.25,.125,.0625)
full_meshes = ((3,16),(4,32),(5,48))
full_time_steps_s = (.1,.05,.025)
load_quadrature_order = 6
solver_rtol = 1e-10
sanity_tolerance = 1e-7
boundary_z_points = 65
boundary_relative_tolerance,boundary_absolute_tolerance_W_per_m2 = 1e-10,1e-9
point_chunk_size = 128
repository_root = Path(__file__).resolve().parents[2]


def evaluate(model, times, r, phi, z, x, y, *, ray_points=None):
    # This modal-contribution study uses independent m subtotals. Explicitly
    # select the same grouping when evaluating its total; model defaults and
    # historical physical-evidence replays keep their original accumulation.
    temperature = np.stack([np.stack([modal_temperature(r[:,None],phi[None,:],plane,
                                      model.prepared.coefficients,model.prepared.roots,model._response(t),
                                      group_by_order=True)
                                      for plane in z]) for t in times])
    opd = model.opd_series(x, y, times, num_ray_points=ray_points, point_chunk_size=point_chunk_size)
    return temperature,opd


def errors_by_time(temperature, reference, weights, opd, reference_opd, *, times=None):
    # The zero field has no meaningful relative error or PV normalization.
    return [None if not np.any(reference[i]) else field_errors(temperature[i].reshape(3,-1),
             reference[i].reshape(3,-1),weights.ravel(),opd[i],reference_opd[i],
                 temperature_normalization='global' if times is not None and 0<times[i]<.625 else 'per_plane')
            for i in range(len(temperature))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','full','fields'),default='smoke')
    parser.add_argument('--selection',type=Path,help='Completed transient joint truncation selection')
    parser.add_argument('--fem-convergence',type=Path,help='Independently passed transient FEM convergence certificate (required for full)')
    parser.add_argument('--angle',type=float,default=60.)
    parser.add_argument('--history',choices=('step','linear','exponential'),default='step')
    parser.add_argument('--ramp-time',type=float,default=5.)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    p = parameters
    if args.mode=='full' and args.fem_convergence is None:
        raise ValueError('--mode full requires --fem-convergence pointing to independently passed FEM evidence')
    if args.mode=='fields':
        from transient_fields import run_fine_fields
        run_fine_fields(args,p,repository_root)
        return
    settings = modal_settings
    counts,ref_settings,observations=axial_counts,reference_settings,observation_times_s
    if args.selection is not None:
        from transient_fields import load_selection
        selected = load_selection(args.selection,p,args.angle,args.history)
        settings = ModalResolution(**selected['selected_settings']['modal'])
        selected_J=selected['selected_settings']['J']
        counts=tuple(sorted(set(j for j in (*axial_counts,selected_J) if j<=selected_J)))
        ref_settings=ModalResolution(**selected['reference']['modal'])
        observations=(0.,*selected['times_s'])
    history = PowerHistory(args.history,args.ramp_time)
    if args.history=='step' and (args.mode=='full' or args.fem_convergence is not None):
        history=PowerHistory()
    fem_evidence=None
    if args.mode=='full':
        observations=tuple(sorted(set((*formal_observation_times_s,*observations))))
    if args.fem_convergence is not None:
        fem_evidence=load_transient_fem_certificate(args.fem_convergence,p,args.angle,history,observations,
            repository_root=repository_root)
    internal = refracted_angle(np.deg2rad(args.angle),p.incident_refractive_index,p.substrate_refractive_index)
    directory = create_run(repository_root,f'transient_{args.mode}_theta{args.angle:g}_{args.history}',args.output_dir)
    summary = {'kind':'transient_modal_fem_study','status':'RUNNING','mode':args.mode,
        'angle_deg':args.angle,'refraction_angle_deg':float(np.rad2deg(internal)),
        'physical_parameters_SI':asdict(p),'h_eff_W_per_m2_K':p.h_eff,
        'thermal_diffusivity_m2_per_s':p.thermal_diffusivity,'history':asdict(history),
        'modal_settings':asdict(settings),'reference_settings':asdict(ref_settings),
        'axial_counts':list(counts),'aperture_radius_ratio':aperture_radius_ratio,
        'sanity_tolerance':sanity_tolerance,'transient_truncation_established':False,
        'fem_convergence_established':False,'boundary_validation_required':True,'provenance':provenance(repository_root,__file__),
        'selection_directory':str(args.selection.resolve()) if args.selection is not None else None}
    if fem_evidence is not None:
        summary.update(fem_convergence_established=True,
            fem_convergence_directory=fem_evidence['certificate_directory'],
            fem_convergence_summary_sha256=fem_evidence['certificate_summary_sha256'],
            fem_convergence_verification_sha256=fem_evidence['certificate_verification_sha256'])
        if args.history=='step':
            summary['history_metadata_policy']='Formal step uses canonical PowerHistory() tau=1; tau does not affect step power'
    if args.selection is not None:
        covered=all(t==0 or t in selected['times_s'] for t in observations)
        summary['transient_truncation_established']=bool(selected.get('transient_truncation_established',False) and covered)
        summary['transient_truncation_scope']=dict(selected_times_s=selected['times_s'],
            requested_times_s=list(observations),all_positive_times_covered=covered,
            note='FEM diagnostic-only 1ms policy does not extend semi-analytical selection coverage')
    write_summary(directory, summary, defaults=script_defaults(globals()))
    started = perf_counter()
    print(f'Run: {directory}',flush=True)
    try:
        prepared = ThermalModel(p,args.angle,settings).prepare()
        c,roots = prepared.coefficients,prepared.roots
        steady = prepared.solve_steady()
        model = prepared.solve_transient(num_axial_modes=counts[-1],history=history)
        times = np.array(observations)
        summary['temperature_normalization_by_time']=['global' if 0<t<.625 else 'per_plane' for t in times]
        long_time = max(times[-1],long_time_decay_exponent/model.slowest_decay_rate,
                        long_time_decay_exponent*history.tau if history.kind!='step' else 0.)
        summary['long_time_s'] = float(long_time)
        summary['slowest_decay_rate_per_s'] = model.slowest_decay_rate
        summary['times_s'] = times.tolist()
        nr,nphi,no=((formal_temperature_r_points,formal_temperature_phi_points,formal_opd_xy_points)
            if args.mode=='full' else (temperature_r_points,temperature_phi_points,opd_xy_points))
        summary['sampling']=dict(temperature_coordinates='polar Gauss radius / uniform azimuth',
            temperature_r_points=nr,temperature_phi_points=nphi,opd_coordinates='Cartesian entrance pupil',
            opd_xy_points=no,fem_certificate_coordinates='Separate common Cartesian thermal disk and refracted pupil')
        nodes,wr = leggauss(nr)
        r = (nodes+1)*p.radius/2
        phi = np.arange(nphi)*2*np.pi/nphi
        z = np.array([0.,p.thickness/2,p.thickness])
        area_weights = np.broadcast_to((wr*p.radius/2*r*2*np.pi/phi.size)[:,None],(r.size,phi.size)).copy()
        axis = np.linspace(-aperture_radius_ratio*p.radius,aperture_radius_ratio*p.radius,no)
        x,y = np.meshgrid(axis,axis)
        mask = (x*x+y*y<=(aperture_radius_ratio*p.radius)**2)&transmitted_ray_mask(x,y,p.radius,p.thickness,internal)
        ox,oy = x[mask],y[mask]
        temperature,opd = evaluate(model,times,r,phi,z,ox,oy)
        fields = {'times_s':times,'r_m':r,'phi_rad':phi,'z_m':z,'area_weights_m2':area_weights,
                  'opd_x_m':x,'opd_y_m':y,'opd_physical_mask':mask,
                  'temperature_sa_K':temperature,'opd_sa_m':opd,'orders':np.array(list(c),dtype=int)}
        for m in c:
            fields[f'source_coefficients_m{m}_W_per_m2'] = c[m]
            fields[f'radial_roots_m{m}_per_m'] = roots[m]
        thermal_args = (c,roots,p.conductivity,p.thickness,p.h_eff,p.h_eff)
        fields['steady_temperature_K'] = np.stack([steady.temperature(r[:,None],phi[None,:],plane) for plane in z])
        fields['steady_opd_m'] = steady.opd(ox,oy,point_chunk_size=point_chunk_size)
        tl,ol = evaluate(model,[long_time],r,phi,z,ox,oy)
        fields['long_time_temperature_K'],fields['long_time_opd_m'] = tl[0],ol[0]
        summary['long_time_errors_vs_steady'] = field_errors(tl[0].reshape(3,-1),fields['steady_temperature_K'].reshape(3,-1),
            area_weights.ravel(),ol[0],fields['steady_opd_m'])

        # Full retained thermal orders, including their steady and optical fields.
        order_t,order_o,source_orders,steady_order_t,steady_order_o = [],[],[],[],[]
        for m in c:
            cm,rm = {m:c[m]},{m:roots[m]}
            one_prepared = PreparedModel(p,args.angle,settings,cm,rm)
            one = one_prepared.solve_transient(num_axial_modes=counts[-1],history=history)
            one_steady = one_prepared.solve_steady()
            tm,om = evaluate(one,times,r,phi,z,ox,oy)
            order_t.append(tm); order_o.append(om)
            source_orders.append(reconstruct_cosine_field(cm,rm,r,phi))
            one_args = (cm,rm,p.conductivity,p.thickness,p.h_eff,p.h_eff)
            steady_order_t.append(np.stack([one_steady.temperature(r[:,None],phi[None,:],plane) for plane in z]))
            steady_order_o.append(one_steady.opd(ox,oy,point_chunk_size=point_chunk_size))
        fields['order_temperature_K'],fields['order_opd_m'] = np.array(order_t),np.array(order_o)
        fields['steady_order_temperature_K'],fields['steady_order_opd_m'] = np.array(steady_order_t),np.array(steady_order_o)
        fields['source_order_W_per_m2'] = np.array(source_orders)
        fields['source_projected_W_per_m2'] = sum(source_orders)
        fields['source_prescribed_W_per_m2'] = oblique_gaussian_heat_flux(r[:,None]*np.cos(phi),r[:,None]*np.sin(phi),
            p.beam_radius,np.deg2rad(args.angle),p.absorbed_power)
        # Orthogonality holds on the full thermal disk, not the cropped ray domain.
        energies = np.sum(fields['order_temperature_K']**2*area_weights[None,None,None],axis=(-2,-1))
        shares = np.divide(energies,energies.sum(axis=0),out=np.zeros_like(energies),where=energies.sum(axis=0)!=0)
        fields['temperature_plane_l2_squared_shares'] = shares
        steady_energies = np.sum(fields['steady_order_temperature_K']**2*area_weights[None,None],axis=(-2,-1))
        fields['steady_temperature_plane_l2_squared_shares'] = steady_energies/steady_energies.sum(axis=0)
        source_energies = np.sum(fields['source_order_W_per_m2']**2*area_weights,axis=(-2,-1))
        fields['source_l2_squared_shares'] = source_energies/source_energies.sum()
        fields['order_sampled_opd_pv_m'] = np.ptp(fields['order_opd_m'],axis=-1)
        summary['mode_opd_interpretation'] = 'Ray integrals of thermal azimuthal orders; not entrance Fourier harmonics. Single-order PVs do not add.'
        with (data_directory(directory)/'modal_contributions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s','thermal_order_m','front_temperature_L2_squared_share','single_order_sampled_opd_PV_nm','total_sampled_opd_PV_nm'])
            for j,t in enumerate(times):
                for i,m in enumerate(c):
                    writer.writerow([t,m,shares[i,j,0],fields['order_sampled_opd_pv_m'][i,j]*1e9,np.ptp(opd[j])*1e9])
            for i,m in enumerate(c):
                writer.writerow(['steady',m,fields['steady_temperature_plane_l2_squared_shares'][i,0],
                                 np.ptp(fields['steady_order_opd_m'][i])*1e9,np.ptp(fields['steady_opd_m'])*1e9])

        print('Checking J, transverse truncation and ray quadrature separately',flush=True)
        j_fields_t,j_fields_o = [],[]
        for count in counts:
            candidate = prepared.solve_transient(num_axial_modes=count,history=history)
            tc,oc = evaluate(candidate,times,r,phi,z,ox,oy)
            j_fields_t.append(tc);j_fields_o.append(oc)
        fields['axial_check_temperature_K'],fields['axial_check_opd_m'] = np.array(j_fields_t),np.array(j_fields_o)
        summary['axial_errors_vs_largest_J'] = [errors_by_time(tc,temperature,area_weights,oc,opd,times=times) for tc,oc in zip(j_fields_t,j_fields_o)]
        ref_prepared = ThermalModel(p,args.angle,ref_settings).prepare()
        ref = ref_prepared.solve_transient(num_axial_modes=counts[-1],history=history)
        # Same ray count isolates M/N/projection; then increase ray quadrature.
        tr,orr = evaluate(ref,times,r,phi,z,ox,oy,ray_points=settings.ray_points)
        fields['reference_temperature_K'],fields['reference_opd_m'] = tr,orr
        summary['transverse_errors_vs_finite_reference'] = errors_by_time(temperature,tr,area_weights,opd,orr,times=times)
        tq,oq = evaluate(model,times,r,phi,z,ox,oy,ray_points=ref_settings.ray_points)
        fields['higher_ray_quadrature_opd_m'] = oq
        summary['ray_quadrature_errors'] = errors_by_time(temperature,tq,area_weights,opd,oq,times=times)
        # Only diagnostics: no enlarged reference or joint transient selection yet.
        summary['truncation_note'] = ('Uses selected M/N/J and recorded times; FEM convergence is separate.'
            if args.selection is not None else 'Diagnostic defaults; no final transient selection.')

        # FEM: spatial refinement at fixed dt, followed by dt refinement on
        # the finest spatial mesh. No duplicate steady assembly or ray formula.
        from thermal_opd.fem import solve_transient_fem
        meshes = smoke_meshes if args.mode=='smoke' else full_meshes
        steps = smoke_time_steps_s if args.mode=='smoke' else full_time_steps_s
        configs = [] if fem_evidence is not None else [(mesh,steps[0]) for mesh in meshes]+[(meshes[-1],dt) for dt in steps[1:]]
        fem_solutions,rows = [],[]
        if fem_evidence is not None:
            fem_solutions=[fem_evidence['solution']]
            checks={check['name']:check for check in fem_evidence['verification']['checks']}
            prefix=f'theta{args.angle:g}_'
            rows=[{**fem_evidence['header']['diagnostics'],
                'max_energy_balance_error_relative_to_full_input':checks[prefix+'energy_balance']['metrics']['max_input_normalized_error'],
                'max_relative_linear_residual':checks[prefix+'weak_equation_residual']['metrics']['max_recorded_relative_residual'],
                **{key:fem_evidence[key] for key in ('source_run','case_index','data_file','data_format','data_sha256')}}]
        for mesh,dt in configs:
            print(f'FEM mesh={mesh}, dt<={dt:g}s',flush=True)
            solution = solve_transient_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,
                lambda xx,yy:oblique_gaussian_heat_flux(xx,yy,p.beam_radius,np.deg2rad(args.angle),p.absorbed_power),
                density=p.density,specific_heat=p.specific_heat,times=times,time_step=dt,history=history,
                disk_refinements=mesh[0],axial_layers=mesh[1],quadrature_order=load_quadrature_order,
                solver='direct' if args.mode=='smoke' else 'cg',rtol=solver_rtol)
            fem_solutions.append(solution)
            scale = max(abs(solution.diagnostics['full_input_W']),np.finfo(float).tiny)
            row = {**solution.diagnostics,'max_energy_balance_error_relative_to_full_input':
                   max((v['balance_absolute_error_W']/scale for v in solution.steps),default=0.),
                   'max_relative_linear_residual':max((v['relative_linear_residual'] for v in solution.steps),default=0.)}
            rows.append(row)
        summary['fem_runs'] = rows
        common = mask.copy()
        for result in fem_solutions:
            common &= result.frames[0].transmitted_mask(x,y,internal)
        fields['opd_fem_common_mask'] = common
        # Interior Gauss nodes still need the common polygon check.
        tx,ty = r[:,None]*np.cos(phi),r[:,None]*np.sin(phi)
        thermal_mask = np.ones(tx.shape,dtype=bool)
        for result in fem_solutions:
            thermal_mask &= result.frames[0].transmitted_mask(tx,ty,0.)
        fields['temperature_fem_common_mask'] = thermal_mask
        sa_common_o = opd[:,common[mask]]
        fields['opd_sa_fem_common_m'] = sa_common_o
        fem_t,fem_o = [],[]
        for i,result in enumerate(fem_solutions):
            tf = np.stack([np.stack([f.temperature_at(tx[thermal_mask],ty[thermal_mask],plane) for plane in z]) for f in result.frames])
            of = np.stack([f.refracted_opd(x[common], y[common], internal, p.dn_dT, point_chunk_size=point_chunk_size) for f in result.frames])
            fem_t.append(tf);fem_o.append(of)
            st = temperature[...,thermal_mask]
            weights = area_weights[thermal_mask]
            rows[i]['errors_vs_sa'] = [None if t==0 else field_errors(tf[j],st[j],weights,of[j],sa_common_o[j],temperature_normalization=summary['temperature_normalization_by_time'][j]) for j,t in enumerate(times)]
            # Rear rise can be tiny at early times: retain absolute errors
            # alongside relative norms so a large ratio is interpretable.
            rows[i]['temperature_absolute_max_difference_K'] = np.max(np.abs(tf-st),axis=-1).tolist()
            rows[i]['temperature_area_rms_difference_K'] = np.sqrt(np.sum((tf-st)**2*weights,axis=-1)/weights.sum()).tolist()
            if fem_evidence is None:
                step_keys = list(result.steps[0])
                np.savez_compressed(data_directory(directory)/f'fem_run{i}.npz',disk_nodes_xy_m=result.frames[0].disk_mesh.p,
                    disk_triangles=result.frames[0].disk_mesh.t,axial_nodes_z_m=result.frames[0].z_m,
                    temperature_K=np.array([f.temperature_K for f in result.frames]),times_s=times,
                    step_keys=np.array(step_keys),steps=np.array([[s[key] for key in step_keys] for s in result.steps]))
        fields['temperature_fem_K'],fields['opd_fem_m'] = np.array(fem_t),np.array(fem_o)
        # Report changes against the finest mesh or smallest dt in that group.
        for i,row in enumerate(rows):
            if fem_evidence is not None:
                row['changes_vs_group_reference']=[None]*len(times)
                row['fem_convergence_directory']=fem_evidence['certificate_directory']
                row['changes_vs_group_reference_note']='No reference was rerun on these polar samples; changes are recorded on the separate common Cartesian certificate grid'
                continue
            reference_i = len(meshes)-1 if i<len(meshes) else len(rows)-1
            row['changes_vs_group_reference'] = [None if t==0 else field_errors(fem_t[i][j],fem_t[reference_i][j],
                area_weights[thermal_mask],fem_o[i][j],fem_o[reference_i][j],temperature_normalization=summary['temperature_normalization_by_time'][j]) for j,t in enumerate(times)]
        summary['sanity_pass'] = all(row['max_energy_balance_error_relative_to_full_input']<sanity_tolerance
                                   and row['max_relative_linear_residual']<sanity_tolerance for row in rows)
        if not summary['sanity_pass']:
            raise RuntimeError('Transient FEM energy/linear checks failed')
        np.savez_compressed(data_directory(directory)/'fields.npz',**fields)
        boundary = write_transient_boundary_check(directory,model,times,r,phi,np.linspace(0,p.thickness,boundary_z_points),
            relative_tolerance=boundary_relative_tolerance,
            absolute_tolerance=boundary_absolute_tolerance_W_per_m2,provenance=summary['provenance'])
        if boundary['status']!='PASS':
            raise RuntimeError('Transient Robin boundary residual failed')
        save_transient_diagnostic_figure(directory,fields)
        summary.update(status='COMPLETE',elapsed_seconds=perf_counter()-started)
        write_summary(directory, summary, defaults=script_defaults(globals()))
        print(f"COMPLETE: {directory}; transient truncation={summary['transient_truncation_established']}; FEM convergence={summary['fem_convergence_established']}",flush=True)
    except BaseException as error:
        summary.update(status='ABORTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
        write_summary(directory, summary, defaults=script_defaults(globals()))
        raise



if __name__=='__main__':
    main()
