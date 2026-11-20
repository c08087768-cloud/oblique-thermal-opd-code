from dataclasses import replace
import numpy as np
import pytest
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.run_io import validation_check,validation_report
from thermal_opd.transient_validation import transient_pde_residual


def test_reports_do_not_promote_missing_scientific_evidence_to_pass():
    passed=validation_check('boundary','passed',metrics={'max':1e-12},threshold=1e-8)
    missing=validation_check('mesh','not_run',scope='Requires independent mesh refinement')
    assert validation_report([passed,missing],config={},run_directory='run')['status']=='not_run'
    assert validation_report([passed],config={},run_directory='run')['status']=='passed'
    failed=validation_check('pde','failed',metrics={'max':1.})
    assert validation_report([passed,missing,failed],config={},run_directory='run')['status']=='failed'
    with pytest.raises(ValueError):
        validation_check('mesh','PASS')


def test_independent_pde_check_detects_wrong_time_response():
    p=replace(PhysicalParameters(),radius=1.,thickness=1.,conductivity=1.,density=2.,specific_heat=3.)
    r,phi,z=np.array([.2,.5]),np.array([.1,1.2]),np.array([.3,.6])
    correct=lambda r,phi,z,t:r*r+z*z+t
    wrong=lambda r,phi,z,t:r*r+z*z+2*t
    args=(p,r,phi,z,1.)
    residual=transient_pde_residual(correct,*args,space_step=1e-3,time_step=1e-3)
    np.testing.assert_allclose(residual,0.,atol=2e-8)
    np.testing.assert_allclose(transient_pde_residual(wrong,*args,space_step=1e-3,time_step=1e-3),6.,atol=2e-8)


def test_independent_one_dimensional_reference_has_correct_steady_and_initial_limits():
    from thermal_opd.transient_validation import transient_uniform_flux_1d_reference
    z,T=transient_uniform_flux_1d_reference(1.,2.,3.,4.,5.,6.,[0.,1e5],num_intervals=40)
    np.testing.assert_array_equal(T[0],np.zeros_like(z))
    denominator=2.*(5.+6.)+5.*6.*1.
    expected=4.*(2.+6.*(1.-z))/denominator
    np.testing.assert_allclose(T[-1],expected,rtol=2e-12)


def test_scientific_artifact_validator_recomputes_evidence_and_preserves_failed_status(tmp_path):
    import json
    from thermal_opd.validation_common import validate_saved_evidence_run
    field=np.array([.01,.02])
    report=validation_report([validation_check('pde','failed',metrics={'max_error':.02},threshold=.001,evidence='pde')],
                               config={},run_directory=tmp_path)
    summary=dict(kind='scientific_verification',status='COMPLETE',verification_status='failed')
    np.savez(tmp_path/'verification_fields.npz',pde=field)
    (tmp_path/'verification.json').write_text(json.dumps(report))
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    result=validate_saved_evidence_run(tmp_path)
    assert result['status']=='PASS'  # File integrity passes, scientific test has failed.
    assert result['verification']['status']=='failed'
    report['checks'][0]['status']='passed'
    (tmp_path/'verification.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Scientific|scientific'):
        validate_saved_evidence_run(tmp_path)


@pytest.mark.parametrize('state',['passed','failed','not_run'])
def test_unified_cli_keeps_strict_scientific_acceptance_for_legacy_reports(tmp_path,state):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    value=.02 if state=='failed' else .0001
    report=validation_report([validation_check('pde',state,metrics={'max_error':value},threshold=.001,evidence='pde')],
                             config={},run_directory=tmp_path)
    (tmp_path/'verification.json').write_text(json.dumps(report))
    # The original report format need not carry verification_type.
    (tmp_path/'summary.json').write_text(json.dumps(dict(kind='scientific_verification',status='COMPLETE',verification_status=state)))
    np.savez(tmp_path/'verification_fields.npz',pde=np.array([value]))
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/verification/validate_run.py'),'--results',str(tmp_path),'--require-scientific-pass'],
                          env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
    assert (result.returncode==0)==(state=='passed'),result.stdout+result.stderr
    saved=json.loads((tmp_path/'check_report.json').read_text())
    assert saved['status']=='PASS'
    assert saved['verification']['status']==state
