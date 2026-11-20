"""Compare new model interfaces with a captured pre-refactor numerical baseline."""
import argparse
from pathlib import Path
from thermal_opd.run_io import create_run, provenance, write_summary, relative_metadata_paths, data_directory
import json
import numpy as np
from thermal_opd.model import ThermalModel,ModalResolution
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.transient import PowerHistory



def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    directory=args.baseline.resolve()
    output=create_run(root,'refactor_regression',args.output_dir)
    config=json.loads((directory/'config.json').read_text())
    p,s=PhysicalParameters(**config['physical_parameters_SI']),ModalResolution(**config['resolution'])
    rows=[]
    with np.load(directory/'baseline.npz') as b:
        def check(key,value):
            absolute=float(np.max(np.abs(value-b[key])))
            scale=float(np.max(np.abs(b[key])))
            relative=absolute/scale if scale else absolute
            rows.append(dict(array=key,max_absolute_difference=absolute,max_difference_over_reference_peak=relative))
            if not np.allclose(value,b[key],rtol=1e-12,atol=1e-15):
                raise AssertionError((key,absolute,relative))
        r,phi,z,times=(b[k] for k in ('r_m','phi_rad','z_m','times_s'))
        rx,px,zx=(b[k] for k in ('gradient_r_m','gradient_phi_rad','gradient_z_m'))
        for angle in config['angles_deg']:
            label=f'theta{angle:g}'
            prepared=ThermalModel(p,angle,s).prepare()
            steady=prepared.solve_steady()
            x,y=b[label+'_x_m'],b[label+'_y_m']
            kw=dict(point_chunk_size=64)
            check(label+'_steady_T',np.stack([steady.temperature(r[:,None],phi[None,:],zz) for zz in z]))
            check(label+'_steady_gradient',np.array(steady.temperature_gradient(rx,px,zx)))
            check(label+'_steady_opd',steady.opd(x,y,**kw))
            for m in prepared.coefficients:
                check(f'{label}_c_m{m}',prepared.coefficients[m]);check(f'{label}_roots_m{m}',prepared.roots[m])
            for kind in config['histories']:
                model=prepared.solve_transient(num_axial_modes=config['num_axial_modes'],history=PowerHistory(kind,config['tau_s']))
                check(f'{label}_{kind}_T',np.stack([np.stack([model.temperature(r[:,None],phi[None,:],zz,t) for zz in z]) for t in times]))
                check(f'{label}_{kind}_gradient',np.stack([model.temperature_gradient(rx,px,zx,t) for t in times]))
                check(f'{label}_{kind}_opd',model.opd_series(x,y,times,**kw))
    report=dict(status='passed',baseline_directory=str(directory),comparisons=rows,
                worst_absolute_difference=max(v['max_absolute_difference'] for v in rows),
                worst_normalized_difference=max(v['max_difference_over_reference_peak'] for v in rows))
    (data_directory(output)/'regression.json').write_text(json.dumps(relative_metadata_paths(report,output),indent=2)+'\n')
    write_summary(output,dict(kind='refactor_regression',status='COMPLETE',baseline_directory=str(directory),comparisons=rows,
        physical_parameters_SI=config['physical_parameters_SI'],numerical_settings=config['resolution'],
        worst_absolute_difference=report['worst_absolute_difference'],worst_normalized_difference=report['worst_normalized_difference'],provenance=provenance(root,__file__)))
    print(f"PASS: {len(rows)} arrays, max normalized difference {report['worst_normalized_difference']:.3g}; {output}")


if __name__=='__main__':
    main()
