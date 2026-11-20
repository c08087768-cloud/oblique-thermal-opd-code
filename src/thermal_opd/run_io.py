"""Unique run directories and reproducibility metadata for command-line studies."""

from datetime import datetime, timezone
from dataclasses import asdict,is_dataclass
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import re
from uuid import uuid4


def create_run(repository_root, prefix, output_dir=None):
    root = Path(repository_root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',prefix):
        raise ValueError('Run label must be a simple name; decimal angle labels are allowed')
    experiment_type=prefix
    for start,category in (('steady_refracted','steady_case'),('transient_fields','transient_case'),
                           ('transient_smoke','fem_comparison'),('transient_full','fem_comparison'),
                           ('fem_smoke','fem_comparison'),('fem_full','fem_comparison'),
                           ('transient_truncation','convergence'),('truncation','convergence')):
        if prefix.startswith(start):
            experiment_type=category
            break
    if experiment_type in ('legacy','figures'):
        raise ValueError('Reserved outputs directory is not an experiment type')
    now=datetime.now(timezone.utc)
    run_id=f'{now:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}'
    path = Path(output_dir).resolve() if output_dir else root/'outputs'/experiment_type/run_id
    try:
        relative=path.relative_to((root/'outputs').resolve())
    except ValueError as error:
        raise ValueError('Output directory must be outputs/<experiment_type>/<run_id> in this repository') from error
    if len(relative.parts)!=2 or relative.parts[0]!=experiment_type:
        raise ValueError(f'Output directory must be outputs/{experiment_type}/<run_id>')
    run_id=path.name
    path.mkdir(parents=True, exist_ok=False)
    (path/'data').mkdir();(path/'figures').mkdir()
    (path/'run.json').write_text(json.dumps(dict(schema_version=3,experiment_type=experiment_type,
        metadata_paths='relative_to_run_directory',
        run_label=prefix,created_at_utc=now.isoformat(),run_id=run_id),indent=2)+'\n',encoding='utf-8')
    write_summary(path,dict(kind=experiment_type,status='RUNNING',run_id=run_id,run_directory='.'))
    return path


def resolve_run_path(directory, value):
    """Resolve a stored reference relative to its run, without legacy redirects."""
    path=Path(value)
    return (path if path.is_absolute() else Path(directory)/path).resolve()


def relative_metadata_paths(value, directory):
    """Normalize path fields only; leave formulas, units and live settings intact."""
    path_keys={'run_directory','figure_directory','selection_directory','selection_run',
               'source_run','baseline_directory','verification_references','artifacts','evidence','data_file','fem_convergence_directory'}
    def convert(item, path_value=False):
        if isinstance(item,dict):
            return {key:convert(child,path_value or key in path_keys) for key,child in item.items()}
        if isinstance(item,(list,tuple)):
            return [convert(child,path_value) for child in item]
        if path_value and isinstance(item,(str,Path)):
            path=Path(item)
            if path.is_absolute():
                try:
                    return Path(os.path.relpath(path,Path(directory).resolve())).as_posix()
                except ValueError:
                    # Separate Windows drives have no relative representation.
                    return path.as_posix()
            return path.as_posix()
        return item
    return convert(value)


def data_directory(directory):
    """Numeric artifacts for new runs; preserve reading legacy flat runs."""
    path=Path(directory)
    return path/'data' if (path/'run.json').is_file() else path


def figure_directory(directory):
    path=Path(directory)
    return path/'figures' if (path/'run.json').is_file() else path


def parse_output_directory(description=None):
    """Small shared option for inspection scripts; no second execution path."""
    import argparse
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--output-dir',type=Path)
    return parser.parse_args().output_dir


def script_defaults(namespace):
    """Snapshot public scalar/dataclass settings; no configuration framework."""
    def plain(value):
        if is_dataclass(value) and not isinstance(value,type):
            return asdict(value)
        if isinstance(value,(str,int,float,bool,type(None))):
            return value
        if isinstance(value,(tuple,list)):
            return [plain(v) for v in value]
        if isinstance(value,dict):
            return {str(k):plain(v) for k,v in value.items()}
        raise TypeError
    settings={}
    for key,value in namespace.items():
        if key.startswith('_') or key in ('parameters','repository_root'):
            continue
        try:
            settings[key]=plain(value)
        except TypeError:
            continue
    return settings


def provenance(repository_root, script):
    root = Path(repository_root)
    files = [Path(script), *sorted((root/'src/thermal_opd').glob('*.py'))]
    versions = {}
    for name in ('numpy', 'scipy', 'scikit-fem', 'matplotlib'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=False).stdout.strip()
    changed = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                             capture_output=True, text=True, check=False).stdout.strip()
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'git_revision': revision or None,
            'working_tree_dirty': bool(changed),
            'source_sha256': {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}


def write_summary(directory, summary, *, defaults=None):
    # allow_nan=False prevents apparently complete JSON containing NaN/Inf.
    if defaults is not None:
        summary={**summary,'script_defaults':defaults}
    summary=relative_metadata_paths(summary,directory)
    path = Path(directory)/'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if not (Path(directory)/'run.json').is_file():
        return
    configuration_keys={'physical_parameters_SI','numerical_settings','modal_settings','reference_settings','reference',
        'enlarged_reference','history','angle_deg','angles_deg','incidence_angle_deg','incidence_angles_deg','axial_counts',
        'selected_axial_modes','times_s','precision_scope','aperture_radius_ratio','tolerance','reference_stability_tolerance',
        'provenance','config','kind','mode','script_defaults','selection_directory','selection_run',
        'source_run','baseline_directory','verification_references'}
    config={key:value for key,value in summary.items() if key in configuration_keys}
    (Path(directory)/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    metrics={key:value for key,value in summary.items() if key not in configuration_keys and key not in ('source_sha256','artifacts','limits')}
    (Path(directory)/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    verification_path=Path(directory)/'verification.json'
    placeholder=not verification_path.exists()
    if not placeholder:
        previous=json.loads(verification_path.read_text(encoding='utf-8'))
        checks=previous.get('checks',[])
        placeholder=(previous.get('status')=='not_run' and len(checks)==1
                     and checks[0].get('name')=='scientific_verification'
                     and checks[0].get('status')=='not_run')
    if placeholder:
        report=validation_report([validation_check('scientific_verification','not_run',
            scope='Execute or attach scoped independent scientific checks separately')],config=config,run_directory=directory)
        verification_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


def validation_check(name,status,*,metrics=None,threshold=None,scope=None,evidence=None):
    """Build a state/evidence record; callers determine the scientific status."""
    if status not in ('passed','failed','not_run'):
        raise ValueError('Scientific status must be passed, failed or not_run')
    return dict(name=name,status=status,metrics=metrics or {},threshold=threshold,scope=scope,evidence=evidence)


def validation_report(checks,*,config,run_directory):
    """Serialize recorded states/counts without evaluating any scientific criterion."""
    names=[row['name'] for row in checks]
    if len(set(names))!=len(names):
        raise ValueError('Duplicate verification check names')
    statuses=[row['status'] for row in checks]
    status='failed' if 'failed' in statuses else ('not_run' if not checks or 'not_run' in statuses else 'passed')
    return relative_metadata_paths(dict(schema_version=1,status=status,checked_at_utc=datetime.now(timezone.utc).isoformat(),
                run_directory='.',config=config,checks=checks,
                counts={key:statuses.count(key) for key in ('passed','failed','not_run')}),run_directory)


def read_json(path):
    """Read saved JSON without changing the scientific meaning of its fields."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_numeric_data(path):
    """Open numeric NPZ data safely; caller owns and closes the returned archive."""
    import numpy as np
    return np.load(Path(path),allow_pickle=False)


def file_sha256(path):
    """Hash large saved artifacts without allocating their entire file contents."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path,value):
    """Save JSON metadata, rejecting nonfinite numbers instead of hiding them."""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
