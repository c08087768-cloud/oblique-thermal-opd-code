"""Independent uniform-front-flux benchmark for prism FEM, not SA comparison."""
from thermal_opd.run_io import script_defaults
from thermal_opd.run_io import data_directory
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from thermal_opd.fem import solve_steady_fem,solve_transient_fem
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.transient import PowerHistory
from thermal_opd.run_io import create_run,provenance,write_summary
from thermal_opd.transient_validation import transient_uniform_flux_1d_reference
from thermal_opd.run_io import validation_check,validation_report

parameters=PhysicalParameters()
uniform_flux_W_per_m2=1000.
reference_intervals=(256,512)
axial_layers=(8,16,32)
time_steps_s=(.25,.125,.0625)
times_s=(0.,.625,2.5,20.)
reference_tolerance,local_fem_tolerance=1e-4,2e-2
field_degree=2
quadrature_order=4
space_time_step_s=.005
space_control_time_step_s=.0025
repository_root=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--field-degree',type=int,choices=(1,2),default=field_degree,
        help='Prism field degree: P2/BDF2 by default; P1 retains backward Euler')
    args=parser.parse_args()
    degree=args.field_degree
    time_integrator='bdf2' if degree==2 else 'backward_euler'
    balance_diagnostic='bdf_balance_absolute_error_W' if time_integrator=='bdf2' else 'balance_absolute_error_W'
    directory=create_run(repository_root,'fem_verification',args.output_dir)
    p=parameters;q=uniform_flux_W_per_m2
    config=dict(physical_parameters_SI=asdict(p),front_flux_W_per_m2=q,h_front=p.h_eff,h_back=p.h_eff,h_side=0.,
        reference_intervals=list(reference_intervals),axial_layers=list(axial_layers),time_steps_s=list(time_steps_s),times_s=list(times_s),
        reference_tolerance=reference_tolerance,local_fem_tolerance=local_fem_tolerance,
        field_degree=degree,time_integrator=time_integrator,quadrature_order=quadrature_order,
        disk_refinements=1,solver='direct',mirror_reduction=False,energy_balance_diagnostic=balance_diagnostic,
        fem_configurations=[])
    fields={};checks=[]
    def check(name,value,threshold,scope):
        fields[name]=np.array([value])
        checks.append(validation_check(name,'passed' if value<=threshold else 'failed',metrics={'max_error':float(value)},threshold=threshold,scope=scope,evidence=name))
    refs=[]
    for n in reference_intervals:
        z,T=transient_uniform_flux_1d_reference(p.thickness,p.conductivity,p.density*p.specific_heat,q,p.h_eff,p.h_eff,times_s,num_intervals=n)
        refs.append((z,T));fields[f'reference_{n}_z_m']=z;fields[f'reference_{n}_T_K']=T
    z,T=refs[-1]
    previous=np.stack([np.interp(z,refs[0][0],row) for row in refs[0][1]])
    check('independent_reference_stability',np.max(abs(previous[1:]-T[1:]))/np.max(abs(T[1:])),reference_tolerance,'1D finite-volume space refinement; exact modal time response')
    thermal=(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,0.,lambda x,y:q+0*x)
    steady=solve_steady_fem(*thermal,disk_refinements=1,axial_layers=axial_layers[-1],quadrature_order=quadrature_order,
        solver='direct',rtol=1e-10,field_degree=degree,mirror_reduction=False)
    config['steady_fem']=dict(field_degree=steady.field_degree,degrees_of_freedom=int(steady.temperature_K.size),
        axial_dofs=int(steady.temperature_K.shape[0]),xy_dofs=int(steady.temperature_K.shape[1]))
    expected=q*(p.conductivity+p.h_eff*(p.thickness-z))/(p.conductivity*(2*p.h_eff)+p.h_eff**2*p.thickness)
    check('steady_known_linear_solution',np.max(abs(steady.temperature_at(0*z,0*z,z)-expected))/np.max(abs(expected)),1e-10,'Exact uniform-flux insulated-side linear axial solution')
    configurations=[(n,time_steps_s[0]) for n in axial_layers]+[(axial_layers[-1],dt) for dt in time_steps_s[1:]]
    time_indices=list(range(len(axial_layers)-1,len(configurations)))
    space_indices=list(range(len(axial_layers)))
    if degree==2:
        # Retain the original coupled sequence as a diagnostic. Its coarse
        # BDF2 time error can cancel spatial error, so it cannot isolate space.
        space_indices=list(range(len(configurations),len(configurations)+len(axial_layers)))
        configurations.extend((n,space_time_step_s) for n in axial_layers)
        config['space_time_control_index']=len(configurations)
        configurations.append((axial_layers[-1],space_control_time_step_s))
    config.update(space_refinement_indices=space_indices,time_refinement_indices=time_indices)
    errors=[];energy=[];secant_energy=[];linear=[]
    samples=[]
    for i,(n,dt) in enumerate(configurations):
        result=solve_transient_fem(*thermal,density=p.density,specific_heat=p.specific_heat,times=times_s,
            time_step=dt,history=PowerHistory(),disk_refinements=1,axial_layers=n,quadrature_order=quadrature_order,
            solver='direct',rtol=1e-10,field_degree=degree,time_integrator=time_integrator,mirror_reduction=False)
        frame=result.frames[-1]
        config['fem_configurations'].append(dict(axial_layers=n,time_step_s=dt,field_degree=frame.field_degree,
            time_integrator=result.diagnostics['time_integrator'],degrees_of_freedom=int(frame.temperature_K.size),
            axial_dofs=int(frame.temperature_K.shape[0]),xy_dofs=int(frame.temperature_K.shape[1])))
        sampled=np.stack([f.temperature_at(0*z,0*z,z) for f in result.frames])
        samples.append(sampled)
        fields[f'configuration_{i}_T_K']=sampled
        errors.append(float(np.max(np.max(abs(sampled[1:]-T[1:]),axis=1)/np.max(abs(T[1:]),axis=1))))
        energy.append(max(v[balance_diagnostic] for v in result.steps)/result.diagnostics['full_input_W'])
        secant_energy.append(max(v['balance_absolute_error_W'] for v in result.steps)/result.diagnostics['full_input_W'])
        for diagnostic in ('balance_absolute_error_W',balance_diagnostic):
            fields[f'configuration_{i}_{diagnostic}']=np.array([v[diagnostic] for v in result.steps])
        linear.append(max(v['relative_linear_residual'] for v in result.steps))
        check(f'initial_condition_{i}',np.max(abs(result.frames[0].temperature_K)),0.,'All prism FEM nodes')
    fields['configuration_errors']=np.array(errors)
    fields['configurations']=np.array(configurations)
    fields['configuration_method_balance_errors']=np.array(energy)
    fields['configuration_secant_balance_errors']=np.array(secant_energy)
    fields['configuration_linear_residuals']=np.array(linear)
    check('transient_known_1d_reference',errors[time_indices[-1]],local_fem_tolerance,'Finest time-group local benchmark only; not final Gaussian mesh/time certificate')
    if degree==2:
        legacy_error=max(0.,float(np.max(np.diff(errors[:len(axial_layers)]))))
        config['legacy_coupled_space_refinement']=dict(indices=list(range(len(axial_layers))),
            status='passed' if legacy_error<=1e-12 else 'failed',max_error=legacy_error,threshold=1e-12,
            scope='Diagnostic retained at the original coarse time step; not an isolated spatial refinement check')
        fields['legacy_coupled_space_refinement']=np.array([legacy_error])
        check('spatial_known_1d_reference',errors[space_indices[-1]],local_fem_tolerance,
            'Finest space-group local benchmark at fixed time step; not final Gaussian mesh/time certificate')
        difference=np.max(abs(samples[space_indices[-1]][1:]-samples[config['space_time_control_index']][1:]),axis=1)/np.max(abs(T[1:]),axis=1)
        fields['space_time_step_relative_differences']=difference
        check('space_time_step_stability',np.max(difference),reference_tolerance,
            'Finest spatial mesh at fixed dt and dt/2; same independent-reference normalization')
    for name,indices in (('space_refinement',space_indices),('time_refinement',time_indices)):
        values=np.array([errors[i] for i in indices])
        check(name,max(0.,np.max(np.diff(values))),1e-12,'Error should decrease under refinement against independently refined 1D reference')
    check('energy_balance',max(energy),1e-9,f'Per-step {time_integrator} storage plus surface losses equals applied input; diagnostic={balance_diagnostic}')
    check('linear_equation_residual',max(linear),1e-9,'Discrete weak-form equations, including natural boundary loads')
    for name in ('gaussian_case_mesh_convergence','gaussian_case_time_convergence','pointwise_boundary_flux','cross_method_comparison'):
        checks.append(validation_check(name,'not_run',scope='Separate general-case verification; weak-form/energy checks do not prove pointwise flux'))
    report=validation_report(checks,config=config,run_directory=directory)
    np.savez_compressed(data_directory(directory)/'verification_fields.npz',**fields)
    (directory/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_summary(directory, dict(kind='scientific_verification', status='COMPLETE', verification_type='fem', config=config, physical_parameters_SI=asdict(p), verification_status=report['status'], provenance=provenance(repository_root, __file__)), defaults=script_defaults(globals()))
    print(f"COMPLETE: {directory}; scientific status={report['status']}; counts={report['counts']}")


if __name__=='__main__':
    main()
