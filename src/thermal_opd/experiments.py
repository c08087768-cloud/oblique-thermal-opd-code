"""Small CLI catalogue; interactive and direct dispatch use the same scripts."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from thermal_opd.run_io import script_defaults

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS={
    'steady':('稳态温度与折射OPD','cases/run_steady_case.py'),
    'transient':('瞬态精细场或FEM诊断','cases/run_transient_case.py'),
    'convergence':('稳态联合截断选择','convergence/select_steady_truncation.py'),
    'transient-convergence':('瞬态联合截断选择','convergence/select_transient_truncation.py'),
    'semi-validation':('半解析独立科学验证','verification/verify_semi_analytical.py'),
    'fem-validation':('FEM独立一维基准','verification/verify_fem.py'),
    'fem-comparison':('稳态FEM与半解析对照','verification/compare_steady_fem.py'),
    'transient-fem-validation':('已保存瞬态FEM网格与时间步验收','verification/verify_transient_fem_convergence.py'),
    'transient-fem-convergence':('瞬态P2 FEM空间时间联合实验','convergence/check_transient_fem_convergence.py'),
    'validate':('重读结果与验证证据','verification/validate_run.py'),
    'boundaries':('既有瞬态三面边界检查','verification/check_transient_boundaries.py'),
    'source':('热源投影检查','diagnostics/inspect_heat_source.py'),
    'azimuthal':('单圆周方位角分解','diagnostics/inspect_azimuthal_modes.py'),
    'temperature':('稳态温度与能量检查','diagnostics/inspect_steady_temperature.py'),
    'opd':('折射OPD检查','diagnostics/inspect_opd.py'),
    'sweeps':('逐参数收敛诊断','convergence/check_steady_convergence.py'),
    'numerical-settings':('Steady numerical settings audit','convergence/check_steady_numerical_settings.py'),
    'replot':('从已保存场重绘','tools/replot_case.py'),
    'regression':('重构数值基线对照','tools/compare_refactor_baseline.py'),
}


def experiment_configuration(name):
    script=ROOT/'scripts'/EXPERIMENTS[name][1]
    spec=importlib.util.spec_from_file_location(f'_preview_{name.replace("-","_")}',script)
    module=importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules while importing.
    sys.modules[spec.name]=module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name,None)
    defaults=script_defaults(vars(module))
    p=getattr(module,'parameters',None)
    return dict(experiment=name,script=str(script),physical_parameters_SI=asdict(p) if p is not None else None,defaults=defaults)


def main(argv=None,*,input_fn=input,print_fn=print):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--experiment',choices=EXPERIMENTS)
    parser.add_argument('--yes',action='store_true',help='Skip interactive confirmation')
    parser.add_argument('--show-config',action='store_true')
    args,extra=parser.parse_known_args(argv)
    if extra and extra[0]=='--':
        extra=extra[1:]
    if args.list:
        for key,(label,script) in EXPERIMENTS.items():
            print_fn(f'{key}: {label} ({script})')
        return 0
    name=args.experiment
    if name is None:
        for i,(key,(label,_)) in enumerate(EXPERIMENTS.items(),1):
            print_fn(f'{i}. {key}: {label}')
        selection=input_fn('选择实验编号或名称（q退出）：').strip()
        if selection.lower()=='q':
            return 0
        if selection.isdigit() and 1<=int(selection)<=len(EXPERIMENTS):
            name=list(EXPERIMENTS)[int(selection)-1]
        elif selection in EXPERIMENTS:
            name=selection
        else:
            print_fn('无效选择；未运行。')
            return 2
    config=experiment_configuration(name)
    config['command_line_arguments']=extra
    if args.show_config:
        print_fn(json.dumps(config,ensure_ascii=False,indent=2))
        return 0
    if not extra and name in ('validate','boundaries','replot'):
        extra=['--results',input_fn('已有结果目录：').strip()]
    if not extra and name=='fem-comparison':
        extra=['--selection',input_fn('完成的稳态截断选择目录：').strip()]
    if not extra and name=='regression':
        extra=['--baseline',input_fn('重构前基线目录：').strip()]
    config['command_line_arguments']=extra
    print_fn(json.dumps(config,ensure_ascii=False,indent=2))
    if not args.yes and input_fn('确认运行？[y/N] ').strip().lower() not in ('y','yes'):
        print_fn('已取消；未启动计算。')
        return 0
    sys.stdout.flush()
    return subprocess.run([sys.executable,config['script'],*extra],cwd=ROOT).returncode
