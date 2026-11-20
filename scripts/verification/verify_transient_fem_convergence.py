"""Review saved transient FEM refinement cases without rerunning their solvers."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thermal_opd.run_io import (create_run,data_directory,figure_directory,read_json,write_json,
    write_summary,provenance,script_defaults)
from thermal_opd.transient_validation import transient_fem_convergence_report,validate_saved_transient_run

repository_root=Path(__file__).resolve().parents[2]
observation_times_s=(0.,.001,.0025,.01,.05,.1,.625,2.5,20.,50.,75.)
angles_deg=(0.,30.,60.)
temperature_xy_points,opd_xy_points=201,101
space_tolerance,joint_tolerance,time_tolerance=5e-3,5e-3,1e-3
sanity_tolerance=1e-7
diagnostic_only_times_s=(.001,)


def save_comparison_artifacts(directory,report):
    rows=[];diagnostic_rows=[]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for check in report['checks']:
        if 'errors_by_time' not in check['metrics']:
            if check['name'].endswith('_diagnostic_recording') and check['status']=='not_run':
                for role,times in check['metrics']['missing_times_by_role_s'].items():
                    diagnostic_rows.extend(dict(check=check['name'],role=role,time_s=t,
                        status='not_run',reason='Required diagnostic frame not saved',used_for_gate=False) for t in times)
            continue
        metrics=check['metrics'];values=[]
        used=metrics.get('used_for_gate_by_time',[True]*len(metrics['times_s']))
        for t,norm,errors,in_gate in zip(metrics['times_s'],metrics['temperature_normalization_by_time'],metrics['errors_by_time'],used):
            maximum=max(float(v) for value in errors.values() for v in np.atleast_1d(value))
            values.append(100*maximum)
            rows.append([check['name'],t,norm,maximum,*[max(np.atleast_1d(errors[k])) for k in errors],
                metrics['isolated_parameter_change'],maximum<=check['threshold'],check['threshold'],in_gate,
                'acceptance' if in_gate else 'diagnostic_only'])
            if not in_gate:diagnostic_rows.append(dict(check=check['name'],time_s=t,
                status='recorded',temperature_normalization=norm,errors=errors,maximum_change_fraction=maximum,
                values_within_target=maximum<=check['threshold'],used_for_gate=False))
        selected=np.asarray(used,dtype=bool);t=np.asarray(metrics['times_s']);v=np.asarray(values)
        ax.loglog(t[selected],v[selected],'o-',label=check['name'])
        if np.any(~selected):ax.loglog(t[~selected],v[~selected],'o',markerfacecolor='none',
            label=check['name']+' (diagnostic only)')
    with (data_directory(directory)/'convergence_by_time.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['check','time_s','temperature_normalization','maximum_change_fraction',
            'temperature_max_relative_l2','temperature_max_relative_linf','opd_rms_over_reference_pv',
            'opd_linf_over_reference_pv','opd_relative_pv','isolated_parameter_change','field_values_within_target','threshold_fraction',
            'used_for_gate','time_role'])
        writer.writerows(rows)
    if report['config'].get('diagnostic_only_times_s') and not diagnostic_rows:
        diagnostic_rows.append(dict(time_s=.001,status='not_run',used_for_gate=False,
            reason='No saved comparison is available at the diagnostic time'))
    write_json(data_directory(directory)/'diagnostic_0p001s.json',diagnostic_rows)
    targets={c['threshold'] for c in report['checks'] if c['name'].endswith('_convergence')}
    for target in sorted(targets):
        ax.axhline(100*target,color='black',linestyle='--',label=f'{100*target:g}% target')
    ax.set(xlabel='Time (s)',ylabel='Maximum checked change (%)',title='Saved transient FEM refinement checks')
    ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(figure_directory(directory)/'convergence_by_time.png',dpi=180);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path,help='JSON listing candidate and XY/Z/time/joint reference runs')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--require-scientific-pass',action='store_true')
    args=parser.parse_args(argv)
    manifest=read_json(args.manifest)
    # An explicit legacy single tolerance is replayed rather than silently
    # changed. Otherwise new runs use the user's approved split policy.
    legacy=manifest.get('tolerance')
    manifest.setdefault('space_tolerance',legacy if legacy is not None else space_tolerance)
    manifest.setdefault('joint_tolerance',manifest['space_tolerance'])
    manifest.setdefault('time_tolerance',legacy if legacy is not None else time_tolerance)
    for key,value in dict(angles_deg=list(angles_deg),times_s=list(observation_times_s),
        temperature_xy_points=temperature_xy_points,opd_xy_points=opd_xy_points,
        sanity_tolerance=sanity_tolerance,diagnostic_only_times_s=list(diagnostic_only_times_s)).items():
        manifest.setdefault(key,value)
    for case in manifest['cases']:
        for role in ('candidate','xy_reference','z_reference','time_reference','joint_reference'):
            if case.get(role) is not None:
                path=Path(case[role]['source_run'])
                case[role]['source_run']=str((path if path.is_absolute() else args.manifest.resolve().parent/path).resolve())
    directory=create_run(repository_root,'fem_convergence_validation',args.output_dir)
    summary=dict(kind='transient_fem_convergence_validation',status='RUNNING',config=manifest,
        fem_convergence_established=False,provenance=provenance(repository_root,__file__))
    write_summary(directory,summary,defaults=script_defaults(globals()))
    print(f'Run: {directory}',flush=True)
    try:
        report,fields=transient_fem_convergence_report(directory,manifest)
        np.savez_compressed(data_directory(directory)/'comparison_fields.npz',**fields)
        write_json(directory/'verification.json',report)
        save_comparison_artifacts(directory,report)
        summary.update(status='COMPLETE',fem_convergence_established=report['fem_convergence_established'],
            verification_status=report['status'])
        write_summary(directory,summary,defaults=script_defaults(globals()))
        validated=validate_saved_transient_run(directory)
        write_json(directory/'check_report.json',validated)
    except Exception as error:
        summary.update(status='FAILED',failure_reason=str(error),fem_convergence_established=False)
        write_summary(directory,summary,defaults=script_defaults(globals()))
        raise
    for check in report['checks']:
        print(f"{check['name']}: {check['status']}",flush=True)
    print(f"COMPLETE: {directory}; scientific status={report['status']}; FEM convergence={report['fem_convergence_established']}",flush=True)
    if args.require_scientific_pass and report['status']!='passed':
        raise SystemExit('Scientific checks failed or are incomplete; inspect verification.json')
    return directory


if __name__=='__main__':
    main()
