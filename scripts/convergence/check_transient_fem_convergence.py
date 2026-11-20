"""Run independent quadratic FEM refinements, then reread them through validation.

Every solve is isolated and bounded. Existing complete cases may be referenced
after configuration/source/data checks; they are never overwritten. This entry
uses the production FEM, not the semi-analytical temperature or source modes.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4
import numpy as np
from skfem import MeshTri,Basis,ElementTriP2,LinearForm,asm
from thermal_opd import fem
from thermal_opd.parameters import PhysicalParameters
from thermal_opd.heat_source import oblique_gaussian_heat_flux
from thermal_opd.transient import PowerHistory
from thermal_opd.run_io import (create_run,data_directory,write_json,read_json,write_summary,
    file_sha256,resolve_run_path,provenance)

repository_root=Path(__file__).resolve().parents[2]
observation_times_s=(0.,.001,.0025,.01,.05,.1,.625,2.5,20.,50.,75.)
interval_counts=(128,128,128,128,128,1024,128,128,128,128)
roles=('candidate','xy_reference','z_reference','time_reference','joint_reference')
field_degree=2
candidate_xy,candidate_z_resolution=4,64
quadrature_order,check_quadrature_order=10,14
case_budget_seconds=8100
startup_budget_seconds=180
space_tolerance,joint_tolerance,time_tolerance=.005,.005,.001
diagnostic_only_times_s=(.001,)
# Audited disposable predecessors used this same Gaussian/step/BDF2 problem.
# A changed/unknown implementation is never silently accepted as equivalent.
known_probe_signatures={
    ('c555a72d3e2444439de99ff91090ac354b071dbb39a688b87b882fef6d3e53e9',
     '8f9c82684c481f54b9d9de3a061f36a612f7830bca88f26057279c6d953e716e'),
    ('c555a72d3e2444439de99ff91090ac354b071dbb39a688b87b882fef6d3e53e9',
     'a2a7d7b05e03238ae680de6e2cd4c5735396813c1d3a61d4115fad0bea4d1ea3')}
# Audited production_fem_20261016T195737541880Z.manifest.json birth identity.
# Later true-residual accumulation changes do not relabel these saved fields.
known_production_fem_hashes={'b4f806342b2f1dbc2639436d073e75c5c38725492fd1de02f2447d14c62f292e'}


def solver_hashes():
    source=Path(fem.__file__).resolve().parent
    return {'src/thermal_opd/'+name:file_sha256(source/name) for name in
        ('fem.py','parameters.py','heat_source.py','transient.py')}


def make_plan(mode,angles):
    if mode not in ('smoke','full') or not angles or len(set(angles))!=len(angles) or not set(angles)<={0.,30.,60.}:
        raise ValueError('Choose smoke/full and distinct angles from 0,30,60')
    xy,z=(2,8) if mode=='smoke' else (candidate_xy,candidate_z_resolution)
    times=list((0.,.001,.0025) if mode=='smoke' else observation_times_s)
    counts=[4,4] if mode=='smoke' else list(interval_counts)
    return [dict(angle_deg=float(angle),role=role,xy=xy+(role in ('xy_reference','joint_reference')),
        z_resolution=z*(2 if role in ('z_reference','joint_reference') else 1),
        interval_counts=[n*(2 if role in ('time_reference','joint_reference') else 1) for n in counts],
        times_s=times,field_degree=2,mirror_reduction=True,time_integrator='bdf2',
        quadrature_order=quadrature_order,check_quadrature_order=check_quadrature_order)
        for angle in angles for role in roles]


def axial_nodes(thickness,resolution):
    """The already tested two-face graded element-vertex profile."""
    bulk=thickness/resolution;front=[0.]
    while front[-1]<thickness/2:
        front.append(min(thickness/2,front[-1]+min(bulk,bulk/20+bulk/4e-4*front[-1])))
    z=np.r_[front,thickness-np.asarray(front)[-2::-1]];z[0]=0.;z[-1]=thickness
    return z


def save_observation(output,frame):
    """Write real observations, explicitly insufficient for a BDF2 restart."""
    folder=data_directory(output)/'partial_observations';folder.mkdir(exist_ok=True)
    path=data_directory(output)/'observation_progress.json'
    record=read_json(path) if path.exists() else dict(status='PARTIAL',fem_convergence_established=False,
        can_resume_time_integration=False,completed_observation_times_s=[],observations=[])
    t=float(frame.diagnostics['time_s']);i=len(record['observations'])
    if record['completed_observation_times_s'] and t<=record['completed_observation_times_s'][-1]:
        raise ValueError('Observation times must increase')
    file=folder/f'temperature_{i:02d}.npy'
    with file.open('xb') as stream:np.save(stream,frame.temperature_K,allow_pickle=False)
    if i==0:
        geometry=folder/'geometry.npz'
        with geometry.open('xb') as stream:
            np.savez_compressed(stream,disk_nodes_xy_m=frame.disk_mesh.p,disk_triangles=frame.disk_mesh.t,
                axial_nodes_z_m=frame.z_m,xy_dof_coordinates_m=Basis(frame.disk_mesh,ElementTriP2()).doflocs,
                field_degree=frame.field_degree)
        record['geometry_sha256']=file_sha256(geometry)
    record['completed_observation_times_s'].append(t)
    record['observations'].append(dict(time_s=t,data_file=str(file.relative_to(output)),sha256=file_sha256(file)))
    pending=path.with_suffix('.pending.json');write_json(pending,record);pending.replace(path)
    print('FEM_OBSERVATION',t,output,flush=True)


def solve_case(root,spec,output):
    p=PhysicalParameters();started=perf_counter();output=Path(output)
    summary=dict(kind='transient_fem_joint_case',status='RUNNING',config=spec,
        physical_parameters_SI=asdict(p),angle_deg=spec['angle_deg'],history=asdict(PowerHistory()),
        times_s=spec['times_s'],solver_source_sha256=solver_hashes(),cases=[],
        fem_convergence_established=False,provenance=provenance(repository_root,__file__))
    write_summary(output,summary)
    try:
        z=axial_nodes(p.thickness,spec['z_resolution'])
        times=np.asarray(spec['times_s']);caps=np.nextafter(np.diff(times)/spec['interval_counts'],np.inf)
        flux=lambda x,y:oblique_gaussian_heat_flux(x,y,p.beam_radius,np.deg2rad(spec['angle_deg']),p.absorbed_power)
        @LinearForm
        def source(v,w):return flux(w.x[0],w.x[1])*np.asarray(v)
        control_mesh=MeshTri.init_circle(nrefs=spec['xy']).scaled(p.radius)
        loads=[asm(source,Basis(control_mesh,ElementTriP2(),intorder=order)) for order in
            (spec['quadrature_order'],spec['check_quadrature_order'])]
        qerror=float(np.linalg.norm(loads[0]-loads[1])/np.linalg.norm(loads[1]))
        if qerror>1e-5:raise ValueError('Gaussian front-load quadrature control failed')
        result=fem.solve_transient_fem(p.radius,p.thickness,p.conductivity,p.h_eff,p.h_eff,p.h_eff,flux,
            density=p.density,specific_heat=p.specific_heat,times=times,time_step=float(max(caps)),
            time_step_by_interval_s=caps,history=PowerHistory(),disk_refinements=spec['xy'],axial_layers=len(z)-1,
            axial_nodes_m=z,quadrature_order=spec['quadrature_order'],solver='cg',rtol=1e-12,
            linear_preconditioner='axial_lines',time_integrator=spec['time_integrator'],field_degree=2,
            mirror_reduction=spec['mirror_reduction'],observation_callback=lambda f:save_observation(output,f))
        mesh=result.frames[0].disk_mesh
        file=data_directory(output)/'fem_run0.npz';keys=list(result.steps[0])
        np.savez_compressed(file,disk_nodes_xy_m=mesh.p,disk_triangles=mesh.t,
            axial_nodes_z_m=result.frames[0].z_m,xy_dof_coordinates_m=Basis(mesh,ElementTriP2()).doflocs,
            field_degree=2,times_s=result.times_s,temperature_K=np.array([f.temperature_K for f in result.frames]),
            step_keys=np.array(keys),steps=np.array([[s[k] for k in keys] for s in result.steps]),
            front_load_W=loads[0],higher_quadrature_front_load_W=loads[1])
        scale=abs(result.diagnostics['full_input_W'])
        energy=max(s['bdf_balance_absolute_error_W'] for s in result.steps)/scale
        residual=max(s['relative_linear_residual'] for s in result.steps)
        if np.max(abs(result.frames[0].temperature_K))!=0 or energy>=1e-7 or residual>=1e-7:
            raise ValueError('FEM initial/energy/weak residual gate failed')
        summary.update(status='COMPLETE',elapsed_seconds=perf_counter()-started,
            cases=[dict(index=0,diagnostics=result.diagnostics,data_sha256=file_sha256(file))],
            gaussian_load_quadrature_relative_l2=qerror,gaussian_load_quadrature_threshold=1e-5,
            maximum_relative_energy_error=energy,maximum_relative_linear_residual=residual)
        write_summary(output,summary);print('FEM_CASE_COMPLETE',output,summary['elapsed_seconds'],flush=True)
    except BaseException as error:
        summary.update(status='FAILED',failure_reason=str(error),elapsed_seconds=perf_counter()-started)
        write_summary(output,summary);raise
    return summary


def reuse_case(directory,spec):
    directory=Path(directory);header=read_json(directory/'summary.json')
    if header['status']!='COMPLETE' or header['config']!=spec:raise ValueError('Reused FEM configuration/status differs')
    if header['physical_parameters_SI']!=asdict(PhysicalParameters()) or header['history']!=asdict(PowerHistory()):
        raise ValueError('Reused FEM physics/history differs')
    if header.get('solver_source_sha256')!=solver_hashes():raise ValueError('Reused FEM solver source differs')
    row=header['cases'][0]
    file=resolve_run_path(directory,row['data_file']) if 'data_file' in row else data_directory(directory)/'fem_run0.npz'
    if file_sha256(file)!=row['data_sha256']:raise ValueError('Reused FEM data hash differs')
    return dict(source_run=str(directory),case_index=0)


def recovery_references(control,plan,reuse_hash):
    """Recover completed cases only; 2026-10-17: never rerun them after a later failure.

    The old control remains immutable. Startup/partial runs are not certificates
    or recovery sources; every returned source is independently reread again.
    """
    from thermal_opd.transient_validation import transient_fem_convergence_report
    control=Path(control).resolve();state=read_json(control/'summary.json')
    if state.get('kind')!='transient_fem_workflow' or state.get('plan')!=plan:
        raise ValueError('Recovery control plan/configuration differs')
    if state.get('status') not in ('FAILED','COMPLETE','ABORTED'):
        raise ValueError('Recovery control must be stopped; a RUNNING queue cannot be resumed concurrently')
    if state.get('solver_source_sha256')!=solver_hashes():raise ValueError('Recovery solver source differs')
    if state.get('reuse_index_sha256')!=reuse_hash:raise ValueError('Recovery frozen reuse index differs')
    planned={(s['angle_deg'],s['role']):s for s in plan};recovered={}
    cases=list(state.get('cases',[]))
    recorded={(c['angle_deg'],role) for c in cases for role in roles if role in c}
    # Retain a complete full case if interruption occurred after worker return
    # but before the case reference was appended. Timing specs never match plan.
    for row in state.get('runs',[]):
        spec=row.get('config',{});key=(spec.get('angle_deg'),spec.get('role'))
        if row.get('status')!='COMPLETE' or key in recorded or planned.get(key)!=spec:continue
        cases.append(dict(angle_deg=key[0],**{key[1]:dict(source_run=row['source_run'],case_index=0)}))
        recorded.add(key)
    for case in cases:
        for role in roles:
            if role not in case:continue
            key=(case['angle_deg'],role)
            if key in recovered:raise ValueError('Duplicate recovery role')
            if key not in planned:raise ValueError('Recovery role is outside plan')
            spec=planned[key];reference=case[role]
            if reference.get('case_index')!=0:raise ValueError('Recovery case index differs')
            source=resolve_run_path(control,reference['source_run']);header=read_json(source/'summary.json')
            if 'applicable_solver_source_sha256' in header:
                # Historical birth is explicit; never pretend it has current birth hashes.
                signature=(header.get('solver_source_sha256',{}).get('src/thermal_opd/fem.py'),
                    header.get('source_spike_sha256'))
                production=(header.get('source_data_format')=='thermal_opd_production_nodes_v1'
                    and signature[0] in known_production_fem_hashes)
                if ((signature not in known_probe_signatures and not production) or not header.get('birth_evidence_files')
                        or any(header.get('solver_source_sha256',{}).get(name)!=sha
                            for name,sha in solver_hashes().items() if not name.endswith('/fem.py'))
                        or header['applicable_solver_source_sha256']!=solver_hashes()):
                    raise ValueError('Recovery historical source applicability differs')
                if (header.get('status')!='COMPLETE' or header.get('config')!=spec
                        or header.get('physical_parameters_SI')!=asdict(PhysicalParameters())
                        or header.get('history')!=asdict(PowerHistory())):
                    raise ValueError('Recovery historical configuration/status differs')
                recovered[key]=dict(source_run=str(source),case_index=0)
            else:recovered[key]=reuse_case(source,spec)
            # Hashes alone do not establish physical validity of saved nodes.
            manifest=dict(angles_deg=[spec['angle_deg']],times_s=spec['times_s'],
                temperature_xy_points=5,opd_xy_points=5,diagnostic_only_times_s=[.001],
                cases=[dict(angle_deg=spec['angle_deg'],candidate=recovered[key])])
            report,_=transient_fem_convergence_report(control,manifest)
            required={'source_completion','initial_condition','energy_balance','weak_equation_residual',
                'robin_loss_diagnostics','prescribed_load'}
            checks={c['name']:c['status'] for c in report['checks']}
            if not all(checks.get(f'theta{spec["angle_deg"]:g}_'+name)=='passed' for name in required):
                raise ValueError('Recovery saved physical checks failed')
    return recovered


def bridge_probe_case(root,entry,spec,index_directory):
    """Explicit metadata bridge to audited archived P2 nodes; no data copying.

    The old solver birth hashes are retained. The current solver's applicability
    is recorded separately, after exact-refactor regression and nodal reread.
    Ordinary same-source reuse checks above are deliberately not weakened.
    """
    from thermal_opd.transient_validation import _transient_fem_case_header
    source=resolve_run_path(index_directory,entry['source_run'])
    snapshot=resolve_run_path(index_directory,entry['source_snapshot'])
    signature=(file_sha256(snapshot/'src/thermal_opd/fem.py'),file_sha256(snapshot/'tmp/p2_fem_spike.py'))
    if signature not in known_probe_signatures:raise ValueError('Unknown historical probe solver source')
    current=solver_hashes()
    birth={name:file_sha256(snapshot/name) for name in current}
    if any(birth[name]!=sha for name,sha in current.items() if not name.endswith('/fem.py')):
        raise ValueError('Historical probe physics/history source differs')
    summary=read_json(source/'summary.json');config=summary['config'];p=PhysicalParameters()
    if summary.get('kind')!='fem_order_probe_case' or summary['status']!='COMPLETE' or not summary['physical_passed']:
        raise ValueError('Historical probe is incomplete or failed')
    expected=dict(degree=2,xy=spec['xy'],zres=spec['z_resolution'],angle_deg=spec['angle_deg'])
    if any(config.get(k)!=v for k,v in expected.items()) or summary['physical_parameters_SI']!=asdict(p):
        raise ValueError('Historical probe configuration/physics differs')
    if config.get('timing_steps') or summary.get('gaussian_load_quadrature_orders')!=[10,14]:
        raise ValueError('Historical probe is timing-only or has unreviewed quadrature')
    if summary['gaussian_load_quadrature_relative_l2']>1e-5:raise ValueError('Historical probe load control failed')
    file=data_directory(source)/'nodes.npz'
    if file_sha256(file)!=summary['nodes_sha256']:raise ValueError('Historical probe node hash differs')
    with np.load(file,allow_pickle=False) as raw:
        times=raw['times_s'];z=raw['z_dofs_m'];keys=raw['step_keys'].tolist();steps=raw['steps']
        if not np.array_equal(times,spec['times_s']):raise ValueError('Historical probe observation times differ')
        if not np.array_equal(z[::2],axial_nodes(p.thickness,spec['z_resolution'])):
            raise ValueError('Historical probe axial element vertices differ')
        mesh=MeshTri.init_circle(nrefs=spec['xy']).scaled(p.radius)
        if not np.array_equal(raw['xy_vertices_m'],mesh.p) or not np.array_equal(raw['triangles'],mesh.t):
            raise ValueError('Historical probe XY mesh differs')
        counts=np.asarray(spec['interval_counts']);dt=np.diff(times)/counts
        expected_dt=np.repeat(dt,counts)
        expected_t=np.concatenate([a+np.arange(1,n+1)*h for a,n,h in zip(times[:-1],counts,dt)])
        expected_t[np.cumsum(counts)-1]=times[1:]
        if steps.shape[0]!=sum(counts) or not np.allclose(steps[:,keys.index('dt_s')],expected_dt,rtol=32*np.finfo(float).eps,atol=0.) or not np.allclose(steps[:,keys.index('time_s')],expected_t,rtol=32*np.finfo(float).eps,atol=0.):
            raise ValueError('Historical probe actual time integration differs')
        diagnostics=dict(degrees_of_freedom=int(summary['dofs']),disk_refinements=spec['xy'],
            axial_layers=(len(z)-1)//2,quadrature_order=10,field_degree_xy=2,field_degree_z=2,geometry_degree=1,
            time_integrator='bdf2',bdf2_startup='backward_euler',bdf2_max_step_ratio=1.5,
            full_input_W=float(steps[0,keys.index('input_W')]),solver='cg',solver_rtol=1e-12)
    canonical=create_run(root,'fem_comparison')
    archived={}
    for name in (*birth,'tmp/p2_fem_spike.py'):
        target=data_directory(canonical)/'source_birth'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot/name,target)
        archived[target.relative_to(canonical).as_posix()]=file_sha256(target)
    original_header=data_directory(canonical)/'source_birth/original_summary.json'
    shutil.copy2(source/'summary.json',original_header)
    archived[original_header.relative_to(canonical).as_posix()]=file_sha256(original_header)
    header=dict(kind='transient_fem_joint_case',status='COMPLETE',config=spec,
        angle_deg=spec['angle_deg'],physical_parameters_SI=asdict(p),history=asdict(PowerHistory()),
        times_s=spec['times_s'],solver_source_sha256=birth,applicable_solver_source_sha256=current,
        source_snapshot=str(snapshot),source_spike_sha256=signature[1],source_run=str(source),
        original_summary_sha256=file_sha256(source/'summary.json'),fem_convergence_established=False,
        birth_evidence_files=archived,
        source_equivalence_scope='Audited P2 assembly/BDF2 and full-node regression; current saved review remains required',
        cases=[dict(index=0,diagnostics=diagnostics,data_file=str(file),data_format='thermal_opd_probe_nodes_v1',
            data_sha256=summary['nodes_sha256'])])
    write_summary(canonical,header)
    _transient_fem_case_header(canonical,dict(source_run='.',case_index=0))
    return dict(source_run=str(canonical),case_index=0)


def bridge_production_case(root,entry,spec,index_directory):
    """2026-10-17: explicitly reuse complete audited production births.

    Only source applicability changes: original float64 nodes, diagnostics and
    all four birth hashes remain evidence. A bridge is never a convergence pass.
    """
    from thermal_opd.transient_validation import _transient_fem_case_header,transient_fem_convergence_report
    source=resolve_run_path(index_directory,entry['source_run'])
    snapshot=resolve_run_path(index_directory,entry['source_snapshot'])
    current=solver_hashes();birth={name:file_sha256(snapshot/name) for name in current}
    if birth['src/thermal_opd/fem.py'] not in known_production_fem_hashes:
        raise ValueError('Unknown historical production FEM source')
    if any(birth[name]!=sha for name,sha in current.items() if not name.endswith('/fem.py')):
        raise ValueError('Historical production physics/history source differs')
    summary=read_json(source/'summary.json');p=PhysicalParameters()
    if summary.get('kind')!='transient_fem_joint_case' or summary.get('status')!='COMPLETE':
        raise ValueError('Historical production case is incomplete or failed')
    if (summary.get('config')!=spec or summary.get('angle_deg')!=spec['angle_deg']
            or summary.get('physical_parameters_SI')!=asdict(p)
            or summary.get('history')!=asdict(PowerHistory())
            or summary.get('times_s')!=spec['times_s']):
        raise ValueError('Historical production configuration/physics/history differs')
    if summary.get('solver_source_sha256')!=birth:
        raise ValueError('Historical production summary birth source differs from snapshot')
    if (summary.get('gaussian_load_quadrature_threshold')!=1e-5
            or not 0<=summary.get('gaussian_load_quadrature_relative_l2',np.inf)<=1e-5):
        raise ValueError('Historical production load control failed or missing')
    rows=summary.get('cases',[])
    if len(rows)!=1 or rows[0].get('index')!=0:raise ValueError('Historical production case index differs')
    row=rows[0];diagnostics=row['diagnostics'];file=data_directory(source)/'fem_run0.npz'
    if ('data_file' in row and resolve_run_path(source,row['data_file'])!=file.resolve()
            or row.get('data_format','thermal_opd_fem')!='thermal_opd_fem'):
        raise ValueError('Historical production case is not an original nodal archive')
    if file_sha256(file)!=row['data_sha256']:raise ValueError('Historical production node hash differs')
    if any(diagnostics.get(k)!=v for k,v in dict(disk_refinements=spec['xy'],
            quadrature_order=spec['quadrature_order'],field_degree_xy=2,field_degree_z=2,
            time_integrator=spec['time_integrator'],solver='cg',solver_rtol=1e-12).items()):
        raise ValueError('Historical production diagnostics differ')
    with np.load(file,allow_pickle=False) as raw:
        times=raw['times_s'];z=raw['axial_nodes_z_m'];keys=raw['step_keys'].tolist();steps=raw['steps']
        if not np.array_equal(times,spec['times_s']):raise ValueError('Historical production observation times differ')
        if not np.array_equal(z[::2],axial_nodes(p.thickness,spec['z_resolution'])):
            raise ValueError('Historical production axial element vertices differ')
        mesh=MeshTri.init_circle(nrefs=spec['xy']).scaled(p.radius)
        if not np.array_equal(raw['disk_nodes_xy_m'],mesh.p) or not np.array_equal(raw['disk_triangles'],mesh.t):
            raise ValueError('Historical production XY mesh differs')
        counts=np.asarray(spec['interval_counts']);dt=np.diff(times)/counts
        expected_dt=np.repeat(dt,counts)
        expected_t=np.concatenate([a+np.arange(1,n+1)*h for a,n,h in zip(times[:-1],counts,dt)])
        expected_t[np.cumsum(counts)-1]=times[1:]
        if (steps.shape[0]!=sum(counts)
                or not np.allclose(steps[:,keys.index('dt_s')],expected_dt,rtol=32*np.finfo(float).eps,atol=0.)
                or not np.allclose(steps[:,keys.index('time_s')],expected_t,rtol=32*np.finfo(float).eps,atol=0.)):
            raise ValueError('Historical production actual time integration differs')
        if raw['temperature_K'].dtype!=np.dtype('float64'):
            raise ValueError('Historical production nodes must retain float64 precision')
    canonical=create_run(root,'fem_comparison');archived={}
    for name in birth:
        target=data_directory(canonical)/'source_birth'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot/name,target)
        archived[target.relative_to(canonical).as_posix()]=file_sha256(target)
    original_header=data_directory(canonical)/'source_birth/original_summary.json'
    shutil.copy2(source/'summary.json',original_header)
    archived[original_header.relative_to(canonical).as_posix()]=file_sha256(original_header)
    header=dict(kind='transient_fem_joint_case',status='COMPLETE',config=spec,
        angle_deg=spec['angle_deg'],physical_parameters_SI=asdict(p),history=asdict(PowerHistory()),
        times_s=spec['times_s'],solver_source_sha256=birth,applicable_solver_source_sha256=current,
        source_snapshot=str(snapshot),source_run=str(source),source_data_format='thermal_opd_production_nodes_v1',
        original_summary_sha256=file_sha256(source/'summary.json'),fem_convergence_established=False,
        birth_evidence_files=archived,
        source_equivalence_scope='Audited unchanged FEM equations and float64 nodes; only true-residual accumulation refined; independent saved review remains required',
        cases=[dict(index=0,diagnostics=diagnostics,data_file=str(file),data_format='thermal_opd_fem',
            data_sha256=row['data_sha256'])])
    write_summary(canonical,header)
    reference=dict(source_run=str(canonical),case_index=0)
    try:
        _transient_fem_case_header(canonical,dict(source_run='.',case_index=0))
        manifest=dict(angles_deg=[spec['angle_deg']],times_s=spec['times_s'],
            temperature_xy_points=5,opd_xy_points=5,diagnostic_only_times_s=[.001],
            cases=[dict(angle_deg=spec['angle_deg'],candidate=reference)])
        report,_=transient_fem_convergence_report(canonical,manifest)
        required={'source_completion','initial_condition','energy_balance','weak_equation_residual',
            'robin_loss_diagnostics','prescribed_load'}
        checks={c['name']:c['status'] for c in report['checks']}
        if not all(checks.get(f'theta{spec["angle_deg"]:g}_'+name)=='passed' for name in required):
            raise ValueError('Historical production saved physical checks failed')
    except Exception as error:
        header.update(status='FAILED',failure_reason=str(error));write_summary(canonical,header);raise
    return reference


def launch_case(root,control,state,spec,budget):
    out=create_run(root,'fem_comparison');file=data_directory(out)/'worker_spec.json';write_json(file,spec)
    row=dict(config=spec,source_run=str(out),status='RUNNING');state['runs'].append(row);write_summary(control,state)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker-spec',str(file),'--output-dir',str(out)]
    try:
        with (data_directory(out)/'solver.log').open('xb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=budget,
                env={**os.environ,'PYTHONUNBUFFERED':'1'})
        result=read_json(out/'summary.json');row.update(status=result['status'],elapsed_seconds=result['elapsed_seconds'])
    except BaseException as error:
        row.update(status='FAILED',failure_reason=str(error));write_summary(control,state)
        summary=read_json(out/'summary.json');summary.update(status='FAILED',failure_reason=str(error));write_summary(out,summary)
        raise
    write_summary(control,state);return out,result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','full'),default='smoke')
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,30.,60.])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--reuse-index',type=Path,help='Explicit complete same-source cases by angle and role')
    parser.add_argument('--control-dir',type=Path,help='New outputs/fem_convergence/<ID> control directory')
    parser.add_argument('--resume-control',type=Path,help='Read completed cases from a previous exact-plan control; never resume partial time history')
    parser.add_argument('--worker-spec',type=Path)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args(argv)
    if args.worker_spec:
        solve_case(repository_root,read_json(args.worker_spec),args.output_dir);return 0
    plan=make_plan(args.mode,args.angles)
    if args.plan:print(json.dumps(plan,indent=2));return 0
    reused=read_json(args.reuse_index)['cases'] if args.reuse_index else []
    keys=[(r['angle_deg'],r['role']) for r in reused]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate reuse role')
    reuse_hash=file_sha256(args.reuse_index) if args.reuse_index else None
    recovered=recovery_references(args.resume_control,plan,reuse_hash) if args.resume_control else {}
    by_angle={angle:dict(angle_deg=angle) for angle in args.angles}
    for (angle,role),reference in recovered.items():by_angle[angle][role]=reference
    control=create_run(repository_root,'fem_convergence',args.control_dir)
    state=dict(kind='transient_fem_workflow',status='RUNNING',runs=[],cases=list(by_angle.values()),
        solver_source_sha256=solver_hashes(),fem_convergence_established=False,mode=args.mode,
        resource_limits=dict(case_window_seconds=case_budget_seconds,full_case_timeout_seconds=case_budget_seconds-180,
            startup_timeout_seconds=startup_budget_seconds,startup_admission_fraction=.8),
        plan=plan,reuse_index_sha256=reuse_hash,resume_control=str(args.resume_control) if args.resume_control else None)
    write_summary(control,state);print('FEM_CONTROL',control,flush=True)
    try:
        manifest=dict(angles_deg=args.angles,times_s=plan[0]['times_s'],temperature_xy_points=15 if args.mode=='smoke' else 201,
            opd_xy_points=13 if args.mode=='smoke' else 101,space_tolerance=.005,joint_tolerance=.005,time_tolerance=.001,
            diagnostic_only_times_s=[.001],sanity_tolerance=1e-7,solver_source_sha256=solver_hashes(),cases=[])
        for spec in plan:
            key=(spec['angle_deg'],spec['role'])
            if key in recovered:
                reference=recovered[key]
                by_angle[spec['angle_deg']][spec['role']]=reference
                state['cases']=list(by_angle.values());write_summary(control,state)
                print('FEM_CASE_RECOVERED',*key,reference['source_run'],flush=True)
                continue
            existing=next((r for r in reused if r['angle_deg']==spec['angle_deg'] and r['role']==spec['role']),None)
            if existing is not None:
                source=resolve_run_path(args.reuse_index.parent,existing['source_run'])
                if existing.get('data_format')=='thermal_opd_probe_nodes_v1':
                    reference=bridge_probe_case(repository_root,existing,spec,args.reuse_index.parent)
                    manifest.setdefault('reused_source_lineage',[]).append(dict(original_source_run=str(source),
                        canonical_source_run=reference['source_run'],source_snapshot=existing['source_snapshot']))
                elif existing.get('data_format')=='thermal_opd_production_nodes_v1':
                    reference=bridge_production_case(repository_root,existing,spec,args.reuse_index.parent)
                    manifest.setdefault('reused_source_lineage',[]).append(dict(original_source_run=str(source),
                        canonical_source_run=reference['source_run'],source_snapshot=existing['source_snapshot'],
                        source_data_format='thermal_opd_production_nodes_v1'))
                else:reference=reuse_case(source,spec)
                state['runs'].append(dict(config=spec,status='COMPLETE',source_run=str(source),reused=True))
                print('FEM_CASE_REUSED',spec['angle_deg'],spec['role'],source,flush=True)
            else:
                if args.mode=='full':
                    samples=[]
                    for n in (16,32):
                        pilot={**spec,'times_s':[0.,float(n*np.nextafter((spec['times_s'][1]/spec['interval_counts'][0]),np.inf))],
                            'interval_counts':[n]}
                        _,result=launch_case(repository_root,control,state,pilot,startup_budget_seconds)
                        samples.append(result['elapsed_seconds'])
                    rate=(samples[1]-samples[0])/16;projection=rate*sum(spec['interval_counts'])
                    if rate<=0 or projection>.8*(case_budget_seconds-180):
                        raise RuntimeError(f'Full FEM cost requires replanning: startup projection {projection:.1f}s')
                source,_=launch_case(repository_root,control,state,spec,case_budget_seconds-180)
                reference=reuse_case(source,spec)
            by_angle[spec['angle_deg']][spec['role']]=reference
            state['cases']=list(by_angle.values());write_summary(control,state)
        manifest['cases']=list(by_angle.values())
        file=data_directory(control)/'manifest.json';write_json(file,manifest)
        path=repository_root/'scripts/verification/verify_transient_fem_convergence.py'
        loader=importlib.util.spec_from_file_location('_fem_saved_review',path)
        module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        final=module.main(['--manifest',str(file),*(['--output-dir',str(args.output_dir)] if args.output_dir else [])])
        state.update(status='COMPLETE',verification_directory=str(final),
            fem_convergence_established=read_json(final/'summary.json')['fem_convergence_established'])
        write_summary(control,state)
        return 0
    except BaseException as error:
        state.update(status='FAILED',failure_reason=str(error));write_summary(control,state);raise


if __name__=='__main__':raise SystemExit(main())
