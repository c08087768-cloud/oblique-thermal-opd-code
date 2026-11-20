"""Opt-in independent semi-analytical checks; numerical settings below.

Reports unexecuted convergence studies explicitly. Never launched by solve().
"""
from thermal_opd.run_io import script_defaults
from thermal_opd.run_io import data_directory
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from thermal_opd.model import ThermalModel,PreparedModel,ModalResolution
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.transient import PowerHistory
from thermal_opd.run_io import create_run,provenance,write_summary
from thermal_opd.transient_validation import transient_solution_acceptance,load_transient_selection
from thermal_opd.steady_validation import steady_boundary_residuals,steady_energy_balance,steady_pde_residual
from thermal_opd.run_io import validation_check,validation_report

parameters=PhysicalParameters()
resolution=ModalResolution(6,8,400,32,32)
num_axial_modes=12
times_s=(.625,2.5,20.,50.,75.)
early_time_s=.001
early_axial_counts=(12,48,128,256,512)
reference_thickness_m,reference_time_s,reference_flux_W_per_m2=.1,.1,1000.
reference_radial_wavenumber_per_m,reference_axial_modes=1.,1200
reference_tolerance=2e-6
pde_tolerance,energy_tolerance,limit_tolerance=1e-4,1e-5,1e-3
boundary_tolerance_W_per_m2=1e-7
space_step_ratio,time_step_s=2e-4,1e-4
early_space_step_ratio,early_time_step_ratio,early_pde_before_s=.002,.0005,.625
repository_root=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle',type=float,default=60.)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--selection',type=Path,help='Validate the actual selected transient configuration')
    args=parser.parse_args()
    directory=create_run(repository_root,'semi_analytical_verification',args.output_dir)
    p=parameters
    settings,selected_J,check_times=resolution,num_axial_modes,times_s
    selected=None
    if args.selection is not None:
        selected=load_transient_selection(args.selection,p,args.angle)
        settings=ModalResolution(**selected['selected_settings']['modal'])
        selected_J=selected['selected_settings']['J']
        check_times=tuple(selected['times_s'])
    axial_reference_counts=tuple(sorted(set((*early_axial_counts,selected_J,
        *((selected['reference']['J'],selected['enlarged_reference']['J']) if selected else ())))))
    prepared=ThermalModel(p,args.angle,settings).prepare()
    steady=prepared.solve_steady()
    transient=prepared.solve_transient(num_axial_modes=selected_J,history=PowerHistory())
    config=dict(physical_parameters_SI=asdict(p),angle_deg=args.angle,resolution=asdict(settings),
                num_axial_modes=selected_J,times_s=list(check_times),early_time_s=early_time_s,
                early_axial_counts=list(axial_reference_counts),space_step_ratio=space_step_ratio,time_step_s=time_step_s)
    config['selection_directory']=str(args.selection.resolve()) if args.selection is not None else None
    config['physical_acceptance_config']=dict(boundary_tolerance_W_per_m2=boundary_tolerance_W_per_m2,
        basis_tolerance=1e-10,pde_tolerance=pde_tolerance,energy_tolerance=energy_tolerance,
        limit_tolerance=limit_tolerance,space_step_ratio=space_step_ratio,time_step_s=time_step_s,projection_tolerance=limit_tolerance,
        early_space_step_ratio=early_space_step_ratio,early_time_step_ratio=early_time_step_ratio,early_pde_before_s=early_pde_before_s)
    checks=[];fields={}
    def check(name,value,threshold,scope,field=None):
        checks.append(validation_check(name,'passed' if value<=threshold else 'failed',
            metrics={'max_error':float(value)},threshold=threshold,scope=scope,evidence=field))
    r,phi,z=np.linspace(0,p.radius,25),np.arange(32)*2*np.pi/32,np.linspace(0,p.thickness,33)
    physical_config=config['physical_acceptance_config']
    physical,physical_fields=transient_solution_acceptance(transient,check_times,physical_config)
    aliases={'pde_residual':'transient_pde','energy_balance':'transient_energy_balance'}
    for item in physical['checks']:
        if item['name']=='thermal_boundaries':continue
        item={**item,'name':aliases.get(item['name'],item['name'])}
        item['evidence']='acceptance_'+item['evidence']
        checks.append(item)
    fields.update({'acceptance_'+key:value for key,value in physical_fields.items()})
    for name,key in (('front_boundary','front_modal_residual_W_per_m2'),('rear_boundary','rear_residual_W_per_m2'),('side_boundary','side_residual_W_per_m2')):
        fields[name]=physical_fields[key]
        check(name,np.max(abs(fields[name])),boundary_tolerance_W_per_m2,'Retained heat flux; sampled Robin consistency',name)
    sb=steady_boundary_residuals(r,phi,z,prepared.coefficients,prepared.roots,p.radius,p.conductivity,p.thickness,p.h_eff,p.h_eff,p.h_eff)
    fields['steady_boundaries']=np.array(list(sb.values()))
    check('steady_boundaries',max(sb.values()),boundary_tolerance_W_per_m2,'Steady retained heat flux','steady_boundaries')
    rr,pp,zz=np.meshgrid(np.linspace(.15*p.radius,.8*p.radius,5),np.arange(8)*2*np.pi/8,
                         np.linspace(.15*p.thickness,.85*p.thickness,5),indexing='ij')
    rr,pp,zz=rr.ravel(),pp.ravel(),zz.ravel()
    fields['initial_temperature']=physical_fields['initial_temperature_K']
    peak=2*p.absorbed_power*np.cos(np.deg2rad(args.angle))/(np.pi*p.beam_radius**2)
    scale=max(peak/p.thickness,1.)
    h=space_step_ratio*min(p.radius,p.thickness)
    fields['pde_normalized_residuals']=physical_fields['pde_normalized_residuals']
    steady_res=np.stack([steady_pde_residual(steady.temperature,p,rr,pp,zz,space_step=step)/scale for step in (h,h/2)])
    fields['steady_pde_normalized_residuals']=steady_res
    check('steady_pde',np.max(abs(steady_res)),pde_tolerance,'Independent interior finite differences','steady_pde_normalized_residuals')
    fields['long_time_normalized_difference']=physical_fields['long_time_temperature_error']
    zero_p=replace(p,absorbed_power=0.)
    zero=PreparedModel(zero_p,args.angle,settings,{m:np.zeros_like(v) for m,v in prepared.coefficients.items()},prepared.roots)
    zero_t=zero.solve_transient(num_axial_modes=selected_J)
    fields['zero_source']=np.r_[zero.solve_steady().temperature(rr,pp,zz),zero_t.temperature(rr,pp,zz,.625),zero_t.opd([0.,.001],[0.,-.001],.625)]
    check('zero_source',np.max(abs(fields['zero_source'])),0.,'Steady/transient and optical zero baseline','zero_source')
    # Independent short-time, deep-slab limit; this tests axial load normalization
    # and rho*cp, rather than providing a general finite-cylinder certificate.
    reference_p=replace(p,thickness=reference_thickness_m,h_conv=0.,emissivity=0.)
    reference_prepared=PreparedModel(reference_p,0.,settings,
        {0:np.array([reference_flux_W_per_m2])},{0:np.array([reference_radial_wavenumber_per_m])})
    config['known_axial_reference']=dict(physical_parameters_SI=asdict(reference_p),
        source_coefficient_W_per_m2=reference_flux_W_per_m2,radial_wavenumber_per_m=reference_radial_wavenumber_per_m,
        num_axial_modes=reference_axial_modes,time_s=reference_time_s)
    reference_solution=reference_prepared.solve_transient(num_axial_modes=reference_axial_modes)
    actual=reference_solution.temperature(0.,0.,0.,reference_time_s)
    expected=2*reference_flux_W_per_m2/reference_p.conductivity*np.sqrt(reference_p.thermal_diffusivity*reference_time_s/np.pi)
    fields['semi_infinite_reference_values_K']=np.array([actual,expected])
    fields['semi_infinite_reference_relative_error']=np.array([(actual-expected)/expected])
    check('known_short_time_axial_reference',abs(fields['semi_infinite_reference_relative_error'][0]),reference_tolerance,
          'Near-uniform single radial mode at r=0, insulated thick slab, short-time constant-flux analytic limit; not a finite-cylinder boundary benchmark',
          'semi_infinite_reference_relative_error')
    fields['transient_energy_normalized_residual']=physical_fields['energy_normalized_residuals']
    energy=steady_energy_balance(prepared.coefficients,prepared.roots,p.radius,p.conductivity,p.thickness,p.h_eff,p.h_eff,p.h_eff,num_r=48,num_z=48,num_phi=64)
    fields['steady_energy_normalized_residual']=np.array([energy['balance_absolute_error_W']/max(abs(energy['modal_input_W']),1e-30)])
    check('steady_energy_balance',fields['steady_energy_normalized_residual'][0],energy_tolerance,'Independent surface quadrature','steady_energy_normalized_residual')
    fields['projection_grid_normalized_changes']=physical_fields['projection_grid_normalized_changes']
    early=np.stack([prepared.solve_transient(num_axial_modes=J).temperature(r[:,None],phi[None,:],0.,early_time_s) for J in axial_reference_counts])
    fields['early_front_temperature_K']=early
    early_scale=max(np.max(abs(early[-1])),np.finfo(float).tiny)
    errors=np.max(abs(early-early[-1]),axis=(1,2))/early_scale
    fields['early_selected_J_error']=np.array([errors[axial_reference_counts.index(selected_J)]])
    fields['early_reference_stability']=np.array([errors[-2]])
    check('initial_neighborhood_selected_J',fields['early_selected_J_error'][0],limit_tolerance,
          f'Early-time t={early_time_s}s, actual selected J={selected_J}; finite axial reference','early_selected_J_error')
    check('initial_neighborhood_reference_stability',errors[-2],limit_tolerance,
          f'J{axial_reference_counts[-2]} vs J{axial_reference_counts[-1]} at t={early_time_s}s','early_reference_stability')
    if selected is not None:
        rows=[next(row for row in case['candidates'] if row['settings']==selected['selected_settings']) for case in selected['cases']]
        metric=max(float(np.max(value)) for row in rows for error in row['errors_by_time'] for value in error.values())
        fields['selected_joint_truncation_error']=np.array([metric])
        check('joint_modal_truncation',metric,selected['tolerance'],
              'Revalidated saved joint selection; recorded angles/times/history only','selected_joint_truncation_error')
    else:
        checks.append(validation_check('joint_modal_truncation','not_run',scope='No saved joint selection supplied'))
    for name in ('fem_independent_verification','cross_method_comparison'):
        checks.append(validation_check(name,'not_run',scope='Separate experiment/evidence required; this run does not certify it'))
    report=validation_report(checks,config=config,run_directory=directory)
    np.savez_compressed(data_directory(directory)/'verification_fields.npz',**fields)
    (directory/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_summary(directory, dict(kind='scientific_verification', status='COMPLETE', verification_type='semi_analytical', physical_parameters_SI=asdict(p), config=config, verification_status=report['status'], provenance=provenance(repository_root, __file__)), defaults=script_defaults(globals()))
    print(f"COMPLETE: {directory}; scientific status={report['status']}; counts={report['counts']}")


if __name__=='__main__':
    main()
