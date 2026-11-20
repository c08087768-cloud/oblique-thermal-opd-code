"""Validate a saved truncation/FEM run; writes PASS only after reloading fields."""

import argparse
from pathlib import Path
from thermal_opd.run_io import read_json,write_json
from thermal_opd.steady_validation import STEADY_RUN_KINDS,validate_saved_steady_run
from thermal_opd.transient_validation import TRANSIENT_RUN_KINDS,validate_saved_transient_run
from thermal_opd.validation_common import require,validate_saved_evidence_run


def validate_saved_run(directory):
    """Single CLI dispatch: model review or model-agnostic stored evidence."""
    path=Path(directory)
    summary=read_json(path/'summary.json')
    kind=summary.get('kind')
    if kind in STEADY_RUN_KINDS:
        return validate_saved_steady_run(path,summary=summary)
    if kind in TRANSIENT_RUN_KINDS:
        return validate_saved_transient_run(path,summary=summary)
    if kind=='scientific_verification':
        return validate_saved_evidence_run(path,summary=summary)
    require(summary.get('status')=='COMPLETE','Run is not complete')
    raise ValueError(f'Unsupported run kind: {kind}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--require-scientific-pass',action='store_true',help='Exit nonzero if any scientific check failed or was not run')
    args = parser.parse_args()
    try:
        report = validate_saved_run(args.results)
    except (ValueError, OSError, KeyError, StopIteration, IndexError, TypeError) as error:
        report = {'status': 'FAIL', 'error': str(error), 'run_directory': '.'}
        if args.results.is_dir():
            write_json(args.results/'check_report.json',report)
        raise SystemExit(f'FAIL: {error}') from error
    write_json(args.results/'check_report.json',report)
    write_json(args.results/'verification.json',report['verification'])
    science=report['verification']['status']
    print(f"PASS (saved-file integrity): {args.results}; scientific status={science}")
    if args.require_scientific_pass and science!='passed':
        raise SystemExit('Scientific verification is failed or incomplete; inspect verification.json')


if __name__ == '__main__':
    main()
