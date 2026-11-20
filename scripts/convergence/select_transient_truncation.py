"""Select transient truncation after numerical and physical gates pass.

Step loading, nominal inputs, t=0 and positive samples from 1 ms. Finite
references do not certify arbitrary t->0+. Robin basis identities are time
independent; assembled residuals are sampled. No FEM or certified PV bound.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from numpy.polynomial.legendre import leggauss
from thermal_opd.convergence import within_tolerance,continuous_aperture_pv
from thermal_opd.model import ThermalModel,ModalResolution
from thermal_opd.opd import refracted_angle,transmitted_ray_mask
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.run_io import create_run,provenance,write_summary,script_defaults,data_directory
from thermal_opd.transient import PowerHistory
from thermal_opd.transient_validation import transient_solution_acceptance,transient_selection_errors

parameters = PhysicalParameters()
angles_deg = (0.,30.,60.)
times_s = (.001,.0025,.01,.05,.1,.625,2.5,20.,50.,75.)
early_normalization_before_s = .625
tolerance = 1e-3
reference_tolerance = tolerance/5
reference = ModalResolution(14,32,2400,256,64)
enlarged_reference = ModalResolution(16,40,3200,512,80)
reference_J,enlarged_reference_J = 512,768
candidates = (ModalResolution(6,8,400,32,16),ModalResolution(8,12,800,64,16),
              ModalResolution(8,16,1200,128,16),ModalResolution(10,20,1200,128,32),
              ModalResolution(12,24,1600,128,32))
candidate_axial_counts = (12,48,128,256)
acceptance_config = dict(boundary_tolerance_W_per_m2=1e-7,basis_tolerance=1e-10,
    pde_tolerance=1e-4,energy_tolerance=1e-5,limit_tolerance=1e-3,
    space_step_ratio=2e-4,time_step_s=1e-4,projection_tolerance=1e-3,
    early_space_step_ratio=.002,early_time_step_ratio=.0005,early_pde_before_s=.625)
temperature_r_points,temperature_phi_points = 96,128
common_xy_points,verification_xy_points,figure_xy_points = 61,121,201
aperture_radius_ratio = .9
pv_seed_radius,pv_seed_phi = 9,48
point_chunk_size = 512
repository_root = Path(__file__).resolve().parents[2]


def optical_coordinates(p,angle,count):
    axis=np.linspace(-aperture_radius_ratio*p.radius,aperture_radius_ratio*p.radius,count)
    x,y=np.meshgrid(axis,axis)
    mask=(x*x+y*y<=(aperture_radius_ratio*p.radius)**2)&transmitted_ray_mask(x,y,p.radius,p.thickness,angle)
    return x[mask],y[mask]


def optical(model,xy,angle,ray_points):
    return model.opd_series(*xy,times_s,num_ray_points=ray_points,point_chunk_size=point_chunk_size)


def compare(T,R,weights,O,RO,G,RG,depth,reference_depth,pv=None,ref_pv=None):
    errors=[]
    for i,t in enumerate(times_s):
        error=transient_selection_errors(T[i],R[i],weights,O[i],RO[i],G[i],RG[i],depth[i],reference_depth[i],
            normalization='global' if t<early_normalization_before_s else 'per_plane')
        if pv is not None:error['opd_relative_pv']=float(abs(pv[i]-ref_pv[i])/ref_pv[i])
        errors.append(error)
    return errors


def extrema(model,angle,ray_points):
    output=[]
    for t in times_s:
        result=continuous_aperture_pv(model.prepare_opd(t,num_ray_points=ray_points,point_chunk_size=point_chunk_size),
            radius=parameters.radius,thickness=parameters.thickness,refraction_angle=angle,
            aperture_radius=aperture_radius_ratio*parameters.radius,
            num_seed_radius=pv_seed_radius,num_seed_phi=pv_seed_phi)
        if result['successful_optimizations']<1:raise RuntimeError('No successful continuous-PV optimization')
        output.append(result)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    directory=create_run(repository_root,'transient_truncation',args.output_dir)
    p=parameters
    summary=dict(kind='transient_joint_truncation_selection',status='RUNNING',physical_parameters_SI=asdict(p),
        h_eff_W_per_m2_K=p.h_eff,history=asdict(PowerHistory()),angles_deg=list(angles_deg),times_s=list(times_s),
        tolerance=tolerance,reference_stability_tolerance=reference_tolerance,
        reference={'modal':asdict(reference),'J':reference_J},enlarged_reference={'modal':asdict(enlarged_reference),'J':enlarged_reference_J},
        candidate_axial_counts=list(candidate_axial_counts),acceptance_version=1,acceptance_config=acceptance_config,
        temperature_normalization_by_time=['global' if t<early_normalization_before_s else 'per_plane' for t in times_s],
        gradient_normalization='max norm(delta dr,dz) / max norm(reference dr,dz), over sampled points/depths',
        aperture_radius_ratio=aperture_radius_ratio,common_xy_points=common_xy_points,
        verification_xy_points=verification_xy_points,selected_figure_xy=figure_xy_points,
        scope_note='t=0 exact branch; positive accuracy at recorded samples >=1 ms; no t->0+ certificate',
        provenance=provenance(repository_root,__file__),cases=[])
    write_summary(directory,summary,defaults=script_defaults(globals()))
    print(f'Run: {directory}',flush=True)
    started=perf_counter()
    nodes,w=leggauss(temperature_r_points)
    r=(nodes+1)*p.radius/2;phi=np.arange(temperature_phi_points)*2*np.pi/temperature_phi_points
    rt,pt=np.broadcast_arrays(r[:,None],phi[None,:])
    weights=np.broadcast_to((w*p.radius/2*r)[:,None],rt.shape).ravel()
    planes=np.array([0.,p.thickness/2,p.thickness])
    # Include moving diffusion-layer depths; late clipped repeats are harmless.
    depth_z=np.array([np.clip(np.r_[0.,np.sqrt(p.thermal_diffusivity*t)*np.array([.25,.5,1.,2.,4.]),
                                   p.thickness/2,p.thickness],0,p.thickness) for t in times_s])
    saved=dict(temperature_r_m=rt.ravel(),temperature_phi_rad=pt.ravel(),temperature_z_m=planes,
               temperature_area_weights=weights,times_s=np.array(times_s),depth_z_m=depth_z)
    acceptance_saved={};contexts=[];prepared_cache={}
    def evaluate(angle,settings,J,xy):
        key=(angle,settings)
        if key not in prepared_cache:prepared_cache[key]=ThermalModel(p,angle,settings).prepare()
        model=prepared_cache[key].solve_transient(num_axial_modes=J)
        T=np.stack([np.stack([model.temperature(r[:,None],phi[None,:],z,t).ravel() for z in planes]) for t in times_s])
        G=np.stack([np.stack([np.stack(model.temperature_gradient(r[:,None],phi[None,:],z,t)).reshape(2,-1)
                             for z in depths],axis=1) for t,depths in zip(times_s,depth_z)])
        D=np.stack([model.temperature(r[:,None,None],phi[None,:,None],z[None,None,:],t).ravel()
                    for t,z in zip(times_s,depth_z)])
        O=optical(model,xy,0.,settings.ray_points)
        return model,T,O,G,D
    try:
        for angle in angles_deg:
            internal=refracted_angle(np.deg2rad(angle),p.incident_refractive_index,p.substrate_refractive_index)
            xy=optical_coordinates(p,internal,common_xy_points);label=f'theta{angle:g}'
            print(f'{label}: early-time references J{reference_J}/J{enlarged_reference_J}',flush=True)
            rm,RT,RO,RG,RD=evaluate(angle,reference,reference_J,xy)
            bm,BT,BO,BG,BD=evaluate(angle,enlarged_reference,enlarged_reference_J,xy)
            rp,bp=extrema(rm,internal,reference.ray_points),extrema(bm,internal,enlarged_reference.ray_points)
            rpv,bpv=np.array([v['pv'] for v in rp]),np.array([v['pv'] for v in bp])
            stability=compare(RT,BT,weights,RO,BO,RG,BG,RD,BD,rpv,bpv)
            case=dict(angle_deg=angle,refraction_angle_deg=float(np.rad2deg(internal)),
                reference_errors_by_time=stability,reference_pass=all(within_tolerance(e,reference_tolerance) for e in stability),
                reference_extrema=rp,enlarged_extrema=bp,candidates=[])
            summary['cases'].append(case)
            for prefix,T,O,G,D,pv in (('reference',RT,RO,RG,RD,rpv),('enlarged',BT,BO,BG,BD,bpv)):
                for suffix,value in (('temperature_K',T),('opd_m',O),('gradient_K_per_m',G),('depth_temperature_K',D),('continuous_pv_m',pv)):
                    saved[label+'_'+prefix+'_'+suffix]=value
            saved[label+'_opd_x_m'],saved[label+'_opd_y_m']=xy
            contexts.append((angle,internal,xy,rm,RT,RO,RG,RD,rpv))
            write_summary(directory,summary,defaults=script_defaults(globals()))
            if not case['reference_pass']:raise RuntimeError(f'{label}: reference not stable; enlarge references')
        chosen=None
        for settings in candidates:
            for J in candidate_axial_counts:
                rows=[];trial=[]
                for case,context in zip(summary['cases'],contexts):
                    angle,internal,xy,rm,RT,RO,RG,RD,rpv=context
                    model,T,O,G,D=evaluate(angle,settings,J,xy)
                    errors=compare(T,RT,weights,O,RO,G,RG,D,RD)
                    row=dict(settings={'modal':asdict(settings),'J':J},errors_by_time=errors,accepted=False,extrema=None)
                    case['candidates'].append(row);rows.append(row)
                    if not all(within_tolerance(e,tolerance) for e in errors):break
                    physical,physical_fields=transient_solution_acceptance(model,times_s,acceptance_config)
                    row['physical_validation']=physical
                    if physical['status']!='passed':break
                    ep=extrema(model,internal,settings.ray_points);pv=np.array([v['pv'] for v in ep])
                    errors=compare(T,RT,weights,O,RO,G,RG,D,RD,pv,rpv)
                    row.update(errors_by_time=errors,extrema=ep)
                    if not all(within_tolerance(e,tolerance) for e in errors):break
                    xy_fine=optical_coordinates(p,internal,verification_xy_points)
                    fine_r=optical(rm,xy_fine,internal,reference.ray_points)
                    fine_s=optical(model,xy_fine,internal,settings.ray_points)
                    fine_errors=compare(T,RT,weights,fine_s,fine_r,G,RG,D,RD,pv,rpv)
                    figure=optical(model,optical_coordinates(p,internal,figure_xy_points),internal,settings.ray_points)
                    sampling=abs(np.ptp(figure,axis=1)-pv)/pv
                    row.update(denser_errors_by_time=fine_errors,sampling_errors_by_time=sampling.tolist())
                    row['accepted']=all(within_tolerance(e,tolerance) for e in fine_errors) and bool(np.all(sampling<=tolerance))
                    if not row['accepted']:break
                    trial.append((T,O,G,D,pv,fine_r,fine_s,figure,physical_fields))
                print(f'M{settings.M}/N{settings.N}/J{J}: {len(trial)}/{len(contexts)} angles accepted',flush=True)
                write_summary(directory,summary,defaults=script_defaults(globals()))
                if len(trial)==len(contexts):chosen=(settings,J,rows,trial);break
            if chosen is not None:break
        if chosen is None:raise RuntimeError('No combination passes numerical and physical gates; enlarge candidate list')
        settings,J,rows,trial=chosen
        summary['selected_settings']={'modal':asdict(settings),'J':J}
        for case,row,result in zip(summary['cases'],rows,trial):
            label=f'theta{case["angle_deg"]:g}'
            T,O,G,D,pv,fine_r,fine_s,figure,physical_fields=result
            case.update(denser_errors_by_time=row['denser_errors_by_time'],denser_common_grid_pass=True,
                sampling_errors_by_time=row['sampling_errors_by_time'],sampling_pass=True,physical_validation=row['physical_validation'])
            for key,value in (('selected_temperature_K',T),('selected_opd_m',O),('selected_gradient_K_per_m',G),
                ('selected_depth_temperature_K',D),('selected_continuous_pv_m',pv),('fine_reference_opd_m',fine_r),
                ('fine_selected_opd_m',fine_s),('figure_opd_m',figure)):
                saved[label+'_'+key]=value
            for key,value in physical_fields.items():acceptance_saved[label+'_'+key]=value
        np.savez_compressed(data_directory(directory)/'fields.npz',**saved)
        np.savez_compressed(data_directory(directory)/'acceptance.npz',**acceptance_saved)
        summary.update(status='COMPLETE',transient_truncation_established=True,
                       elapsed_seconds=perf_counter()-started)
        write_summary(directory,summary,defaults=script_defaults(globals()))
        print(f'COMPLETE: selected {summary["selected_settings"]}; {directory}',flush=True)
    except BaseException as error:
        np.savez_compressed(data_directory(directory)/'partial_fields.npz',**saved)
        summary.update(status='ABORTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
        write_summary(directory,summary,defaults=script_defaults(globals()))
        raise


if __name__=='__main__':
    main()
