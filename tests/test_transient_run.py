import json
from pathlib import Path
import os
import subprocess
import sys
import shutil
import numpy as np
import pytest
from thermal_opd.transient_validation import validate_saved_transient_run
from thermal_opd.run_io import data_directory


def test_transient_script_saves_consistent_modes_and_checker_catches_corruption(tmp_path):
    # Real entry point: a lost angular order or corrupted saved nodal field
    # must not be accepted solely because the summary claims COMPLETE.
    root = Path(__file__).resolve().parents[1]
    # Run the unchanged script in an isolated repository layout.
    scripts=tmp_path/'scripts/cases'
    scripts.mkdir(parents=True)
    entry=Path(shutil.copy2(root/'scripts/cases/run_transient_case.py',scripts))
    output = tmp_path/'outputs/fem_comparison/transient'
    env = {**os.environ,'PYTHONPATH':str(root/'src'),'MPLBACKEND':'Agg'}
    run = subprocess.run([sys.executable,str(entry),
                          '--output-dir',str(output)],cwd=root,env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    report = validate_saved_transient_run(output)
    assert report['status']=='PASS'
    assert report['transient_boundary_check']['status']=='PASS'
    summary = json.loads((output/'summary.json').read_text())
    assert summary['transient_truncation_established'] is False
    assert summary['boundary_validation_required'] is True
    assert (data_directory(output)/'boundary_residuals.csv').is_file()
    with np.load(data_directory(output)/'fields.npz') as saved:
        fields = {key:saved[key] for key in saved.files}
    fields['order_temperature_K'][0,1,0,0,0] += 1.
    np.savez_compressed(data_directory(output)/'fields.npz',**fields)
    with pytest.raises(ValueError,match='mode|Mode'):
        validate_saved_transient_run(output)
