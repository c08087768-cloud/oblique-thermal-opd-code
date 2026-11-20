"""Fine transient fields, called by run_transient_case --mode fields.

Numerical settings stay in this script. Shared physics and snapshot plotting
are imported from thermal_opd; this does not run a large FEM batch.
"""
from thermal_opd.run_io import script_defaults
from thermal_opd.model import ThermalModel, PreparedModel
from thermal_opd.run_io import data_directory, figure_directory
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from thermal_opd.convergence import continuous_aperture_pv,field_errors
from thermal_opd.model import ModalResolution
from thermal_opd.opd import refracted_angle, transmitted_ray_mask
from thermal_opd.plotting import save_temperature_opd_figure,save_evolution,save_history,save_early_depths,save_modal_contribution_figure
from thermal_opd.run_io import create_run,provenance,write_summary
from thermal_opd.transient_validation import validate_saved_transient_run,load_transient_selection as load_selection
from thermal_opd.transient_validation import write_transient_boundary_check
from thermal_opd.metrics import opd_metrics

temperature_r_points,temperature_phi_points,opd_xy_points = 161,180,201
aperture_radius_ratio = .9
point_chunk_size = 512
history_positive_points = 121
long_time_decay_exponent = 12.
pv_seed_radius,pv_seed_phi = 9,48
boundary_z_points = 65
boundary_relative_tolerance,boundary_absolute_tolerance_W_per_m2 = 1e-10,1e-9


def run_fine_fields(args,p,root):
    if args.selection is None:
        raise ValueError('--mode fields requires --selection pointing to a completed transient selection')
    selection = load_selection(args.selection,p,args.angle,args.history)
    modal = ModalResolution(**selection['selected_settings']['modal'])
    J = selection['selected_settings']['J']
    if opd_xy_points<selection['selected_figure_xy'] or selection['aperture_radius_ratio']!=aperture_radius_ratio:
        raise ValueError('Fine figure settings do not satisfy the selection sampling/aperture')
    directory = create_run(root,f'transient_fields_theta{args.angle:g}',args.output_dir)
    from thermal_opd.run_io import figure_directory
    figures = figure_directory(directory)
    summary = dict(kind='transient_fine_fields',status='RUNNING',angle_deg=args.angle,physical_parameters_SI=asdict(p),
        aperture_radius_ratio=aperture_radius_ratio,selection_directory=str(Path(args.selection).resolve()),
        transient_truncation_established=True,fem_convergence_established=False,boundary_validation_required=True,
        modal_settings=asdict(modal),selected_axial_modes=J,precision_target=selection['tolerance'],
        precision_scope=dict(angles_deg=selection['angles_deg'],times_s=selection['times_s'],history=selection['history']),
        numerical_settings=dict(temperature_r=temperature_r_points,temperature_phi=temperature_phi_points,opd_xy=opd_xy_points),
        figure_directory=str(figures),provenance=provenance(root,__file__),snapshots=[],figures=[])
    write_summary(directory, summary, defaults=script_defaults(globals()))
    print(f'Fine fields: {directory}\nFigures: {figures}',flush=True)
    started = perf_counter()
    try:
        internal = refracted_angle(np.deg2rad(args.angle),p.incident_refractive_index,p.substrate_refractive_index)
        summary['refraction_angle_deg'] = float(np.rad2deg(internal))
        prepared = ThermalModel(p,args.angle,modal).prepare()
        c,roots = prepared.coefficients,prepared.roots
        steady = prepared.solve_steady()
        model = prepared.solve_transient(num_axial_modes=J)
        long_time = max(selection['times_s'][-1],long_time_decay_exponent/model.slowest_decay_rate)
        times = np.array([0.,*selection['times_s'],long_time])
        summary['times_s'],summary['long_time_s'] = times.tolist(),float(long_time)
        r = np.linspace(0.,p.radius,temperature_r_points)
        phi = np.arange(temperature_phi_points)*2*np.pi/temperature_phi_points
        z = np.array([0.,p.thickness/2,p.thickness])
        axis = np.linspace(-aperture_radius_ratio*p.radius,aperture_radius_ratio*p.radius,opd_xy_points)
        x,y = np.meshgrid(axis,axis)
        mask = (x*x+y*y<=(aperture_radius_ratio*p.radius)**2)&transmitted_ray_mask(x,y,p.radius,p.thickness,internal)
        print('Computing dense snapshots on the selected modes',flush=True)
        T = np.stack([np.stack([model.temperature(r[:,None],phi[None,:],plane,t) for plane in z]) for t in times])
        O = np.full((times.size,)+x.shape,np.nan)
        O[:,mask] = model.opd_series(x[mask], y[mask], times, num_ray_points=modal.ray_points, point_chunk_size=point_chunk_size)
        sampled_pv = np.ptp(O[:,mask],axis=1)
        saved = dict(times_s=times,temperature_r_m=r,temperature_phi_rad=phi,temperature_z_m=z,temperature_K=T,
                     opd_x_m=x,opd_y_m=y,aperture_mask=mask,opd_m=O,sampled_pv_m=sampled_pv,orders=np.array(list(c)))
        for m in c:
            saved[f'source_coefficients_m{m}_W_per_m2'],saved[f'radial_roots_m{m}_per_m'] = c[m],roots[m]
        summary['sampled_opd_metrics']=[opd_metrics(frame,mask) for frame in O]
        # Source coefficients and axial/ray modules are exactly those selected.
        case = next(case for case in selection['cases'] if case['angle_deg']==args.angle)
        row = next(row for row in case['candidates'] if row['settings']==selection['selected_settings'])
        continuous = [None,*row['extrema']]
        evaluator = model.prepare_opd(long_time, num_ray_points=modal.ray_points, point_chunk_size=point_chunk_size)
        continuous.append(continuous_aperture_pv(evaluator,radius=p.radius,thickness=p.thickness,refraction_angle=internal,
            aperture_radius=aperture_radius_ratio*p.radius,num_seed_radius=pv_seed_radius,num_seed_phi=pv_seed_phi))
        for j,t in enumerate(times):
            if t==0:
                summary['snapshots'].append(dict(time_s=0.,sampled_pv_m=0.,continuous_pv_m=0.,figure=None))
                continue
            filename = f'transient_t{t:.6g}s.png'
            summary['snapshots'].append(dict(time_s=float(t),sampled_pv_m=float(sampled_pv[j]),
                continuous_pv_m=continuous[j]['pv'],extrema=continuous[j],figure=filename,
                relative_pv_sampling_error=float(abs(sampled_pv[j]-continuous[j]['pv'])/continuous[j]['pv'])))
            save_temperature_opd_figure(figures/filename,r,phi,T[j],x,y,mask,O[j],
                title=f'Transient: {args.angle:g} deg, t={t:.6g} s, M={modal.M}, N={modal.N}, J={J}',
                pv_m=sampled_pv[j],continuous_pv_m=continuous[j]['pv'])
            summary['figures'].append(filename)
        thermal_args = (c,roots,p.conductivity,p.thickness,p.h_eff,p.h_eff)
        saved['steady_temperature_K'] = np.stack([steady.temperature(r[:,None],phi[None,:],plane) for plane in z])
        saved['steady_opd_m'] = np.full(x.shape,np.nan)
        saved['steady_opd_m'][mask] = steady.opd(x[mask],y[mask],point_chunk_size=point_chunk_size)
        steady_integrator = steady.prepare_opd(point_chunk_size=point_chunk_size)
        steady_extrema = continuous_aperture_pv(steady_integrator,radius=p.radius,
            thickness=p.thickness,refraction_angle=internal,aperture_radius=aperture_radius_ratio*p.radius,
            num_seed_radius=pv_seed_radius,num_seed_phi=pv_seed_phi)
        summary['steady_extrema'] = steady_extrema
        save_temperature_opd_figure(figures/'steady_limit.png',r,phi,saved['steady_temperature_K'],x,y,mask,saved['steady_opd_m'],
            title=f'Same-mode steady limit: {args.angle:g} deg, M={modal.M}, N={modal.N}',
            pv_m=np.ptp(saved['steady_opd_m'][mask]),continuous_pv_m=steady_extrema['pv'])
        summary['figures'].append('steady_limit.png')
        long_errors = field_errors(T[-1].reshape(3,-1),saved['steady_temperature_K'].reshape(3,-1),
            np.broadcast_to(np.maximum(r[:,None],p.radius*1e-9),T.shape[-2:]).ravel(),O[-1,mask],saved['steady_opd_m'][mask])
        summary['long_time_errors_vs_same_modes_steady'] = long_errors
        # Same fine coordinates for the retained thermal order contributions.
        order_T,order_O = [],[]
        for m in c:
            one = PreparedModel(p,args.angle,modal,{m:c[m]},{m:roots[m]}).solve_transient(num_axial_modes=J)
            order_T.append(np.stack([np.stack([one.temperature(r[:,None],phi[None,:],plane,t) for plane in z]) for t in times]))
            optical = np.full_like(O,np.nan)
            optical[:,mask] = one.opd_series(x[mask], y[mask], times, num_ray_points=modal.ray_points, point_chunk_size=point_chunk_size)
            order_O.append(optical)
        saved['order_temperature_K'],saved['order_opd_m'] = np.array(order_T),np.array(order_O)
        wr = r*(r[1]-r[0]);wr[-1] *= .5
        energies = np.sum(saved['order_temperature_K']**2*wr[None,None,None,:,None],axis=(-2,-1))
        shares = np.divide(energies,energies.sum(axis=0),out=np.zeros_like(energies),where=energies.sum(axis=0)!=0)
        saved['temperature_plane_l2_squared_shares'] = shares
        saved['order_sampled_opd_pv_m'] = np.ptp(saved['order_opd_m'][...,mask],axis=-1)
        with (data_directory(directory)/'modal_contributions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s','thermal_order_m','front_temperature_L2_squared_share','single_order_sampled_opd_PV_nm','total_sampled_PV_nm'])
            for j,t in enumerate(times):
                for i,m in enumerate(c):
                    writer.writerow([t,m,shares[i,j,0],saved['order_sampled_opd_pv_m'][i,j]*1e9,sampled_pv[j]*1e9])
        save_modal_contribution_figure(figures/'transient_modes.png',times,list(c),shares,saved['order_sampled_opd_pv_m'],sampled_pv)
        summary['figures'].append('transient_modes.png')
        np.savez_compressed(data_directory(directory)/'fields.npz',**saved)
        print('Computing temperature histories and dense-aperture PV history',flush=True)
        ht = np.unique(np.r_[0.,times,np.geomspace(times[1],long_time,history_positive_points)])
        hx,hy = np.array([0.,p.radius/2,0.]),np.array([0.,0.,p.radius/2])
        history_T = np.stack([np.stack([model.temperature(np.hypot(hx,hy),np.arctan2(hy,hx),plane,t) for plane in z]) for t in ht])
        history_O = model.opd_series(x[mask], y[mask], ht, num_ray_points=modal.ray_points, point_chunk_size=point_chunk_size)
        history_pv = np.ptp(history_O,axis=1)
        history_steady_T = np.stack([steady.temperature(np.hypot(hx,hy),np.arctan2(hy,hx),plane) for plane in z])
        np.savez_compressed(data_directory(directory)/'history.npz',times_s=ht,point_x_m=hx,point_y_m=hy,z_m=z,
            temperature_K=history_T,opd_m=history_O,sampled_pv_m=history_pv,steady_temperature_K=history_steady_T,
            steady_sampled_pv_m=np.ptp(saved['steady_opd_m'][mask]))
        print('Checking front/rear/side Robin residuals at every snapshot and history time',flush=True)
        summary['boundary_history_required']=True
        boundary=write_transient_boundary_check(directory,model,ht,r,phi,np.linspace(0,p.thickness,boundary_z_points),
            relative_tolerance=boundary_relative_tolerance,
            absolute_tolerance=boundary_absolute_tolerance_W_per_m2,provenance=summary['provenance'])
        if boundary['status']!='PASS':raise RuntimeError('Transient Robin boundary residual failed')
        with (data_directory(directory)/'snapshots.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s','center_front_K','center_middle_K','center_rear_K','sampled_PV_nm','continuous_numerical_PV_nm'])
            for j,t in enumerate(times):
                writer.writerow([t,*T[j,:,0,0],sampled_pv[j]*1e9,summary['snapshots'][j]['continuous_pv_m']*1e9])
        with (data_directory(directory)/'history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s','center_front_K','center_middle_K','center_rear_K','x_half_radius_front_K','y_half_radius_front_K','sampled_PV_nm'])
            for j,t in enumerate(ht):
                writer.writerow([t,*history_T[j,:,0],history_T[j,0,1],history_T[j,0,2],history_pv[j]*1e9])
        save_evolution(figures,saved,times.size-1)
        save_early_depths(figures,saved)
        save_history(figures,ht,history_T,history_pv,history_steady_T,np.ptp(saved['steady_opd_m'][mask]),summary['snapshots'])
        summary['figures'] += ['transient_evolution.png','transient_early_depths.png','transient_history.png']
        summary.update(status='COMPLETE',elapsed_seconds=perf_counter()-started,
            precision_note='0.1% finite-reference selection at recorded angles/key times; zero exact, long-time steady limit checked separately. Intermediate history samples are not individually certified.',
            fem_note='This is the fine semi-analytical field run. FEM space/time convergence remains separate.')
        write_summary(directory, summary, defaults=script_defaults(globals()))
        print(f'COMPLETE: {directory}',flush=True)
    except BaseException as error:
        summary.update(status='ABORTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
        write_summary(directory, summary, defaults=script_defaults(globals()))
        raise
