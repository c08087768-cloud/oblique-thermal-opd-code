"""Add/recompute Robin checks for a saved transient run, then validate the run.

Reuses saved coefficients and roots. Does not rerun projection, OPD or FEM,
and does not replace the original fields/summary/provenance.
"""
from thermal_opd.run_io import data_directory
import argparse
from pathlib import Path
import numpy as np
from thermal_opd.run_io import provenance,read_json,write_json
from thermal_opd.transient_validation import validate_saved_transient_run
from thermal_opd.transient_validation import solution_from_saved_transient,write_transient_boundary_check

side_z_points = 65
relative_tolerance,absolute_tolerance_W_per_m2 = 1e-10,1e-9
repository_root = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    args = parser.parse_args()
    directory = args.results.resolve()
    summary = read_json(directory/'summary.json')
    if summary['status']!='COMPLETE':
        raise ValueError('Need a completed transient run')
    print(f'Checking saved transient boundaries: {directory}',flush=True)
    with np.load(data_directory(directory)/'fields.npz',allow_pickle=False) as saved:
        model = solution_from_saved_transient(summary,saved)
        keys = ('temperature_r_m','temperature_phi_rad') if summary['kind']=='transient_fine_fields' else ('r_m','phi_rad')
        times=saved['times_s']
        if summary.get('boundary_history_required',False):
            with np.load(data_directory(directory)/'history.npz',allow_pickle=False) as history:
                times=history['times_s'].copy()
        write_transient_boundary_check(directory,model,times,saved[keys[0]],saved[keys[1]],
            np.linspace(0,model.prepared.parameters.thickness,side_z_points),
            relative_tolerance=relative_tolerance,absolute_tolerance=absolute_tolerance_W_per_m2,
            provenance=provenance(repository_root,__file__))
    try:
        report = validate_saved_transient_run(directory)
    except (ValueError,OSError,KeyError,IndexError,TypeError) as error:
        report = dict(status='FAIL',error=str(error),run_directory='.')
        write_json(directory/'check_report.json',report)
        raise SystemExit(f'FAIL: {error}') from error
    write_json(directory/'check_report.json',report)
    print(f'PASS: {directory}\nBoundary table: {data_directory(directory)/"boundary_residuals.csv"}',flush=True)


if __name__=='__main__':
    main()
