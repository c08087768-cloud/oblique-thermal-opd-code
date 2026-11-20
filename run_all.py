"""Run existing scientific experiments in dependency order, with honest evidence states.

This collects current experiments; COMPLETE means execution and saved-file checks
finished. It does not certify missing Gaussian FEM convergence or failed early-time
checks. Full runs return 2 when scientific evidence remains failed or incomplete.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from time import perf_counter

from thermal_opd.experiments import EXPERIMENTS, experiment_configuration
from thermal_opd.run_io import create_run, provenance, read_json, resolve_run_path, write_summary,file_sha256

ROOT=Path(__file__).resolve().parent


def build_plan(mode, angles):
    if mode not in ('smoke','full') or not angles or len(set(angles))!=len(angles) or not set(angles)<={0.,30.,60.}:
        raise ValueError('Choose smoke/full and distinct angles from the current selection scope: 0, 30, 60')
    stages=[]
    def add(name,experiment=None,args=(),category=None,needs=(),validate=True,target=None):
        stages.append(dict(name=name,experiment=experiment,args=list(args),category=category,
                           needs=list(needs),validate=validate,target=target))
    add('software_tests',validate=False)
    add('fem_reference','fem-validation',category='fem_verification')
    if mode=='full':
        add('steady_selection','convergence',category='convergence')
        add('transient_selection','transient-convergence',category='convergence')
        for angle in angles:
            add(f'semi_reference_theta{angle:g}','semi-validation',('--angle',str(angle),'--selection','@transient_selection'),
                category='semi_analytical_verification',needs=('transient_selection',))
        for angle in angles:
            add(f'steady_theta{angle:g}','steady',('--angle',str(angle)),category='steady_case',needs=('steady_selection',))
            add(f'transient_fields_theta{angle:g}','transient',('--mode','fields','--angle',str(angle),
                '--selection','@transient_selection'),category='transient_case',needs=('transient_selection',))
        add('steady_fem','fem-comparison',('--mode','full','--selection','@steady_selection'),
            category='fem_comparison',needs=('steady_selection',))
        add('transient_fem_convergence','transient-fem-convergence',
            ('--mode','full','--angles',*[str(a) for a in angles]),category='fem_convergence_validation')
    for angle in (angles if mode=='full' else [max(angles)]):
        args=['--mode',mode,'--angle',str(angle),'--history','step']
        needs=[]
        if mode=='full':
            args+=['--selection','@transient_selection','--fem-convergence','@transient_fem_convergence']
            needs=['transient_selection','transient_fem_convergence']
        add(f'transient_fem_theta{angle:g}','transient',args,category='fem_comparison',needs=needs)
    if mode=='full':
        for name,category in (('azimuthal','azimuthal_inspection'),('source','heat_source_inspection'),
                              ('temperature','temperature_inspection'),('opd','opd_inspection'),
                              ('sweeps','convergence_diagnostic')):
            add(name,name,category=category,validate=False)
    fields=[s['name'] for s in stages if s['experiment']=='transient']
    for name in fields:
        add('boundaries_'+name,'boundaries',('--results','@'+name),needs=(name,),target=name)
    if mode=='full':
        for name in [s['name'] for s in stages if s['name'].startswith(('steady_theta','transient_fields_'))]:
            add('replot_'+name,'replot',('--results','@'+name),category='replot',needs=(name,),validate=False)
    return stages


def _file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=digest.hexdigest())


def _code_hashes():
    files=[ROOT/'run_all.py',ROOT/'run.py',ROOT/'pyproject.toml']
    for folder in ('src','scripts','tests'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend((ROOT/'tests/fixtures').glob('*.json'))
    return {p.relative_to(ROOT).as_posix():_file_hash(p) for p in sorted(set(files)) if p.is_file()}


def _artifacts(directory):
    # Boundary stages refresh sidecars/review timestamps on their own new runs.
    # Preserve hashes of all stable numerical data, configuration and figures.
    return {p.relative_to(directory).as_posix():_file_hash(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in ('verification.json','check_report.json')
            and not p.name.startswith('boundary_')}


def _run_command(command, log_file, *, append=False):
    with log_file.open('a' if append else 'x',encoding='utf-8') as stream:
        stream.write(json.dumps(command,ensure_ascii=False)+'\n');stream.flush()
        proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg'},start_new_session=os.name=='posix')
        try:
            return proc.wait()
        except KeyboardInterrupt:
            if os.name=='posix':os.killpg(proc.pid,signal.SIGTERM)
            else:proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name=='posix':os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait()
            raise


def _check_steady_selection(directory):
    selected=read_json(directory/'summary.json')
    defaults=experiment_configuration('steady')['defaults']
    settings=dict(M=defaults['max_azimuthal_order'],N=defaults['num_radial_modes'],
        projection_r=defaults['num_projection_r'],projection_phi=defaults['num_projection_phi'],
        ray_points=defaults['num_opd_ray_points'])
    if selected['selected_settings']!=settings or selected['selected_figure_xy']>defaults['num_opd_xy']:
        raise ValueError('New steady selection differs from steady case settings; update the case settings before running it')


def _save(directory, report):
    write_summary(directory,report)
    with (directory/'data/stages.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['name','status','scientific_status','run_directory','log_file','error'])
        writer.writeheader()
        for row in report['stages']:
            writer.writerow({key:row.get(key,'') for key in writer.fieldnames})


def _continuation_parent(parent,evidence_path,mode,angles,plan,fingerprint):
    """Verify an explicit same-formula code transition before allocating a batch."""
    parent=Path(parent).resolve();evidence_path=Path(evidence_path).resolve()
    if parent.parent!=(ROOT/'outputs/experiment_suite').resolve():
        raise ValueError('Continuation parent must be outputs/experiment_suite/<ID>')
    old=read_json(parent/'summary.json');evidence=read_json(evidence_path)
    if old.get('status') not in ('FAILED','ABORTED','COMPLETE'):
        raise ValueError('Continuation parent status must be FAILED, ABORTED or COMPLETE; RUNNING cannot be continued')
    if (evidence.get('kind')!='experiment_suite_code_continuation' or evidence.get('version')!=1
        or evidence.get('purpose') not in ('same_formula_residual_accumulation_precision',
                                           'same_formula_modal_sum_grouping')):
        raise ValueError('Unsupported continuation evidence kind/version/purpose')
    if evidence.get('numerical_policy')!=dict(residual_formula_unchanged=True,
        residual_tolerance_unchanged=True,saved_temperature_dtype='float64'):
        raise ValueError('Continuation must preserve residual formula/tolerance and float64 saved temperatures')
    parent_hash=file_sha256(parent/'summary.json')
    if evidence.get('parent_suite_sha256')!=parent_hash:raise ValueError('Continuation parent summary hash differs')
    before=old.get('code_sha256',{})
    if not before or any(not isinstance(v,dict) or set(v)!={'bytes','sha256'} or
        not isinstance(v['bytes'],int) or v['bytes']<0 or not isinstance(v['sha256'],str) or len(v['sha256'])!=64
        or Path(k).is_absolute() or '..' in Path(k).parts for k,v in before.items()):
        raise ValueError('Invalid parent code manifest')
    manifest_hash=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if evidence.get('parent_code_manifest_sha256')!=manifest_hash:raise ValueError('Continuation parent code manifest hash differs')
    changes=evidence.get('changes',[])
    changed={k for k in before.keys()|fingerprint.keys() if before.get(k)!=fingerprint.get(k)}
    if (not changed or len(changes)!=len(changed) or {c.get('path') for c in changes}!=changed
        or any(c.get('before_sha256')!=before.get(c['path'],{}).get('sha256')
            or c.get('after_sha256')!=fingerprint.get(c['path'],{}).get('sha256')
            or not isinstance(c.get('reason'),str) or not c['reason'].strip() for c in changes)):
        raise ValueError('Continuation changes must exactly match every before/after code hash')
    snapshot=resolve_run_path(evidence_path.parent,evidence['parent_source_snapshot'])
    source=snapshot/'run_all.py'
    if file_sha256(source)!=before['run_all.py']['sha256']:raise ValueError('Parent snapshot run_all.py hash differs')
    # Execute the hash-verified frozen module without __main__ or bytecode writes.
    namespace={'__name__':'_continuation_parent_plan','__file__':str(source)}
    exec(compile(source.read_bytes(),str(source),'exec'),namespace)
    if (old.get('kind')!='experiment_suite' or old.get('mode')!=mode or old.get('angles_deg')!=list(angles)
        or namespace['build_plan'](mode,angles)!=plan
        or [r['name'] for r in old['stages']]!=[s['name'] for s in plan]
        or (old.get('plan') is not None and old['plan']!=plan)):
        raise ValueError('Continuation parent mode/angles/plan differ')
    reports=[]
    for record in evidence.get('independent_test_reports',[]):
        path=resolve_run_path(evidence_path.parent,record['data_file'])
        if record.get('status')!='passed' or file_sha256(path)!=record.get('sha256'):
            raise ValueError('Continuation regression report hash/status differs')
        try:
            tree=ET.parse(path).getroot()
            suites=[s for s in tree.iter('testsuite') if s.find('testsuite') is None]
            counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
        except (ET.ParseError,ValueError) as error:
            raise ValueError('Invalid continuation JUnit regression report') from error
        if counts['tests']<=counts['skipped'] or counts['failures'] or counts['errors']:
            raise ValueError('Continuation regression report has no passing tests or has failures/errors')
        reports.append(dict(data_file=str(path),sha256=record['sha256'],status='passed',counts=counts))
    if not reports:raise ValueError('Continuation requires hashed passing JUnit regression reports')
    for row in old['stages']:
        if row['status']=='COMPLETE':
            if row.get('run_directory'):
                path=resolve_run_path(parent,row['run_directory'])
                if not path.is_dir() or _artifacts(path)!=row.get('artifacts'):
                    raise ValueError(f"Completed parent data changed or is corrupt: {row['name']}")
            if not row.get('log_file') or not resolve_run_path(parent,row['log_file']).is_file():
                raise ValueError(f"Completed parent stage log is missing: {row['name']}")
    if file_sha256(parent/'summary.json')!=parent_hash or file_sha256(source)!=before['run_all.py']['sha256']:
        raise ValueError('Continuation parent/snapshot changed during verification')
    return old,dict(parent_suite=str(parent),parent_summary_sha256=parent_hash,
        parent_code_sha256=before,parent_code_manifest_sha256=manifest_hash,parent_source_snapshot=str(snapshot),
        evidence=dict(data_file=str(evidence_path),sha256=file_sha256(evidence_path)),
        independent_test_reports=reports,evidence_record=evidence,
        parent_stage_states=[dict(name=r['name'],status=r['status'],scientific_status=r.get('scientific_status','not_run')) for r in old['stages']])


def run_suite(mode, angles, *, output_dir=None, resume=None, runner=None,fem_reuse_index=None,
              continue_from=None,continuation_evidence=None):
    runner=_run_command if runner is None else runner
    plan=build_plan(mode,angles)
    fingerprint=_code_hashes()
    if resume is not None and (continue_from is not None or continuation_evidence is not None):
        raise ValueError('Use strict resume or explicit continuation, not both')
    if (continue_from is None)!=(continuation_evidence is None):
        raise ValueError('Continuation requires both --continue-from and --continuation-evidence')
    parent_report=continuation=None
    if continue_from is not None:
        parent_report,continuation=_continuation_parent(continue_from,continuation_evidence,mode,angles,plan,fingerprint)
    reuse_record=None if fem_reuse_index is None else dict(data_file=str(Path(fem_reuse_index).resolve()),
        sha256=file_sha256(Path(fem_reuse_index)))
    if resume is not None:
        if output_dir is not None:raise ValueError('Use --resume or --output-dir, not both')
        directory=Path(resume).resolve()
        if directory.parent!=(ROOT/'outputs/experiment_suite').resolve():
            raise ValueError('Resume must name an outputs/experiment_suite/<ID> directory')
        report=read_json(directory/'summary.json')
        previous_reuse=report.get('fem_reuse_index')
        if previous_reuse is not None:
            previous_reuse={**previous_reuse,'data_file':str(resolve_run_path(directory,previous_reuse['data_file']))}
        if previous_reuse!=reuse_record:raise ValueError('Resume FEM reuse index changed')
        if report.get('mode')!=mode or report.get('angles_deg')!=list(angles):
            raise ValueError('Resume mode/angles differ from the recorded batch')
        if report.get('code_sha256')!=fingerprint:raise ValueError('Resume code changed; use a new batch for a new code version')
        if [r['name'] for r in report['stages']]!=[r['name'] for r in plan]:raise ValueError('Resume plan changed')
        for row in report['stages']:
            if row['status']=='COMPLETE' and row.get('run_directory'):
                path=resolve_run_path(directory,row['run_directory'])
                if not path.is_dir() or _artifacts(path)!=row['artifacts']:
                    raise ValueError(f"Completed data changed or is corrupt: {row['name']}")
            if row.get('inherited_log_sha256') and file_sha256(resolve_run_path(directory,row['log_file']))!=row['inherited_log_sha256']:
                raise ValueError(f"Inherited stage log changed: {row['name']}")
        if report.get('continuation'):
            previous=report['continuation']
            for record in [previous['evidence'],*previous['independent_test_reports']]:
                if file_sha256(resolve_run_path(directory,record['data_file']))!=record['sha256']:
                    raise ValueError('Continuation evidence/report changed on resume')
    else:
        directory=create_run(ROOT,'experiment_suite',output_dir)
        (directory/'data/logs').mkdir()
        report=dict(kind='experiment_suite',status='RUNNING',mode=mode,angles_deg=list(angles),
            started_at_utc=datetime.now(timezone.utc).isoformat(),code_sha256=fingerprint,
            provenance=provenance(ROOT,ROOT/'run_all.py'),paper_ready=False,scientific_status='not_run',
            fem_reuse_index=reuse_record,
            stages=[dict(name=s['name'],status='PENDING',attempts=[]) for s in plan],
            plan=plan,
            limits=['Existing experimental settings/thresholds are unchanged.',
                    'Formal transient cases require an independently reread FEM space/time/joint certificate.',
                    'Independent semi-analytical checks use this batch selection; unresolved checks retain failed/not_run states.',
                    'Intermediate transient history times are not individually precision-certified.',
                    'Development baselines are not needed or available on a fresh scientific checkout.'])
        if continuation is not None:
            parent=Path(continue_from).resolve()
            continuation['parent_suite']=os.path.relpath(parent,directory)
            continuation['parent_source_snapshot']=os.path.relpath(continuation['parent_source_snapshot'],directory)
            report['continuation']=continuation
            for index,oldrow in enumerate(parent_report['stages']):
                if oldrow['status']!='COMPLETE':continue
                row=json.loads(json.dumps(oldrow))
                for item in [row,*row.get('attempts',[])]:
                    for key in ('run_directory','log_file','fem_control_directory'):
                        if item.get(key):item[key]=os.path.relpath(resolve_run_path(parent,item[key]),directory)
                row.update(inherited_from_stage=oldrow['name'],inherited_from_suite=os.path.relpath(parent,directory),
                    inherited_parent_summary_sha256=continuation['parent_summary_sha256'],
                    inherited_code_manifest_sha256=continuation['parent_code_manifest_sha256'],
                    inherited_log_sha256=file_sha256(resolve_run_path(parent,oldrow['log_file'])))
                report['stages'][index]=row
    print(f'Batch: {directory}',flush=True)
    report['status']='RUNNING';_save(directory,report)
    by_name={row['name']:row for row in report['stages']}
    for index,stage in enumerate(plan,1):
        row=by_name[stage['name']]
        if row['status']=='COMPLETE':
            print(f'[{index}/{len(plan)}] already complete: {stage["name"]}',flush=True);continue
        attempt=len(row['attempts'])+1
        previous_fem_control=next((path for path in
            [row.get('fem_control_directory'),*[a.get('fem_control_directory') for a in reversed(row['attempts'])]]
            if path is not None and (resolve_run_path(directory,path)/'summary.json').is_file()),None)
        log=directory/f'data/logs/{stage["name"]}_a{attempt}.log'
        result=None
        if stage['category']:
            result=ROOT/'outputs'/stage['category']/f'{directory.name}_{stage["name"]}_a{attempt}'
        elif stage['target']:
            result=resolve_run_path(directory,by_name[stage['target']]['run_directory'])
        row.update(status='RUNNING',scientific_status='not_run',log_file=str(log.relative_to(directory)),error=None)
        if result is not None:row['run_directory']=os.path.relpath(result,directory)
        row['attempts'].append(dict(number=attempt,started_at_utc=datetime.now(timezone.utc).isoformat(),log_file=row['log_file']))
        _save(directory,report)
        started=perf_counter()
        print(f'[{index}/{len(plan)}] running: {stage["name"]}; log={log}',flush=True)
        try:
            for name in stage['needs']:
                if by_name[name]['status']!='COMPLETE':raise ValueError(f'Prerequisite not complete: {name}')
            if stage['experiment']=='steady':
                _check_steady_selection(resolve_run_path(directory,by_name['steady_selection']['run_directory']))
            if stage['experiment'] is None:
                command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                         '--basetemp',str(directory/f'data/pytest_a{attempt}')]
            else:
                args=[str(resolve_run_path(directory,by_name[a[1:]]['run_directory'])) if a.startswith('@') else a
                      for a in stage['args']]
                command=[sys.executable,'-u',str(ROOT/'scripts'/EXPERIMENTS[stage['experiment']][1]),*args]
                if stage['experiment']=='transient-fem-convergence' and fem_reuse_index is not None:
                    command+=['--reuse-index',str(Path(fem_reuse_index).resolve())]
                if stage['experiment']=='transient-fem-convergence':
                    control=ROOT/'outputs/fem_convergence'/f'{directory.name}_{stage["name"]}_a{attempt}'
                    row['fem_control_directory']=os.path.relpath(control,directory)
                    row['attempts'][-1]['fem_control_directory']=row['fem_control_directory']
                    command+=['--control-dir',str(control)]
                    if previous_fem_control is not None:
                        previous=resolve_run_path(directory,previous_fem_control)
                        if (previous/'summary.json').is_file():command+=['--resume-control',str(previous)]
                    _save(directory,report)
                if stage['category']:command+=['--output-dir',str(result)]
            row['command']=command
            code=runner(command,log)
            row['returncode']=code
            if code:raise RuntimeError(f'Command exited {code}; inspect {log}')
            if result is not None:
                summary=read_json(result/'summary.json')
                if summary.get('status')!='COMPLETE':raise ValueError('Experiment did not record COMPLETE')
                if stage['validate']:
                    command=[sys.executable,'-u',str(ROOT/'scripts'/EXPERIMENTS['validate'][1]),'--results',str(result)]
                    code=runner(command,log,append=True)
                    if code:raise RuntimeError(f'Saved-result validation exited {code}; inspect {log}')
                    check=read_json(result/'check_report.json')
                    if check.get('status')!='PASS':raise ValueError('Saved-result check did not pass')
                    row['scientific_status']=check['verification']['status']
                    row['integrity_status']='PASS'
                else:row['integrity_status']='metadata_only'
                row['artifacts']=_artifacts(result)
                row['mesh_convergence_established']=summary.get('mesh_convergence_established')
                row['fem_convergence_established']=summary.get('fem_convergence_established')
            row['status']='COMPLETE'
        except KeyboardInterrupt:
            row['status']='ABORTED';report['status']='ABORTED';_save(directory,report)
            return 130,directory
        except Exception as error:
            row.update(status='FAILED',error=str(error));report['status']='FAILED'
            _save(directory,report)
            print(f'FAILED: {stage["name"]}: {error}',flush=True)
            return 1,directory
        finally:
            row['attempts'][-1].update(elapsed_seconds=perf_counter()-started,status=row['status'])
            _save(directory,report)
    science=[r['scientific_status'] for s,r in zip(plan,report['stages']) if s['validate'] and s['experiment']]
    convergence=(mode=='full' and by_name['steady_fem'].get('mesh_convergence_established') is True
                 and by_name['transient_fem_convergence'].get('fem_convergence_established') is True
                 and all(by_name[f'transient_fem_theta{a:g}'].get('fem_convergence_established') is True for a in angles))
    report['scientific_status']='failed' if 'failed' in science else ('passed' if science and all(v=='passed' for v in science) and convergence else 'not_run')
    report.update(status='COMPLETE',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  paper_ready=mode=='full' and report['scientific_status']=='passed')
    _save(directory,report)
    print(f'COMPLETE execution: {directory}; scientific={report["scientific_status"]}; paper_ready={report["paper_ready"]}',flush=True)
    return (2 if mode=='full' and not report['paper_ready'] else 0),directory


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','full'),default='smoke')
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,30.,60.])
    parser.add_argument('--plan',action='store_true',help='Print the command/dependency plan without computing or creating outputs')
    parser.add_argument('--output-dir',type=Path,help='New outputs/experiment_suite/<ID>; child runs stay in their experiment categories')
    parser.add_argument('--resume',type=Path,help='Resume this exact code and mode; completed data are hash-checked, failed attempts get new output IDs')
    parser.add_argument('--continue-from',type=Path,help='Create a new code-version batch inheriting only verified completed parent stages')
    parser.add_argument('--continuation-evidence',type=Path,help='Explicit before/after code hashes, frozen parent source and passing JUnit reports')
    parser.add_argument('--fem-reuse-index',type=Path,help='Explicit verified complete FEM source cases; index bytes fixed when resuming')
    args=parser.parse_args(argv)
    try:
        plan=build_plan(args.mode,args.angles)
        if args.plan:
            print(json.dumps(dict(mode=args.mode,angles_deg=args.angles,stages=plan),ensure_ascii=False,indent=2));return 0
        code,_=run_suite(args.mode,args.angles,output_dir=args.output_dir,resume=args.resume,fem_reuse_index=args.fem_reuse_index,
            continue_from=args.continue_from,continuation_evidence=args.continuation_evidence)
        return code
    except (ValueError,OSError,KeyError) as error:
        print(f'ERROR: {error}',file=sys.stderr);return 1


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(main())
