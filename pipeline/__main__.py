"""One source-checkout CLI; SDK imports occur only in the requested worker."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('command',choices=['smoke','assemble','inspect','capture','m2','m3-sample',
        'm3-capture','m3-accept','m4','render','materials-audit','materials-repair','materials-refine',
        'material-pair-ue','material-pair-isaac','desk-normal','materials-seven'])
    if len(sys.argv)==1 or sys.argv[1] in ('--help','-h'):
        parser.print_help();return
    args,tail=parser.parse_known_args()
    if args.command=='smoke':
        from .smoke import main as smoke
        smoke(tail);return
    if args.command in ('assemble','inspect'):
        sub=argparse.ArgumentParser(prog='pipeline '+args.command)
        sub.add_argument('scene');sub.add_argument('--out');sub.add_argument('--manifest');sub.add_argument('--registry')
        sub.add_argument('--allow-provisional',action='store_true');opts=sub.parse_args(tail)
        from .ue_isaac_pipeline.usd_pipeline import prepare,inventory
        from .ue_isaac_pipeline.io import read_json
        if args.command=='assemble':
            if not opts.out:sub.error('--out required')
            if Path(opts.out).exists():raise FileExistsError('Fresh output required')
            result=prepare(opts.scene,opts.out,manifest=opts.manifest,registry=opts.registry,allow_provisional=opts.allow_provisional)
        else:result=inventory(opts.scene,read_json(opts.manifest) if opts.manifest else None)
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    modules={'capture':'pipeline.simforge.sensors.export','m2':'pipeline.simforge.physics.sweep',
        'm3-sample':'pipeline.simforge.randomize.sampler','m3-capture':'pipeline.native.capture_m3',
        'm3-accept':'pipeline.m3_accept','m4':'pipeline.simforge.cli',
        'render':'pipeline.native.render_apartment_physics_movie',
        'materials-audit':'pipeline.native.audit_apartment_materials',
        'materials-repair':'pipeline.native.apply_apartment_material_review',
        'materials-refine':'pipeline.native.refine_apartment_finishes',
        'material-pair-ue':'pipeline.native.capture_material_ue',
        'material-pair-isaac':'pipeline.native.capture_material_isaac',
        'desk-normal':'pipeline.native.author_v2_desk',
        'materials-seven':'pipeline.native.material_profile_apartment'}
    sys.argv=[args.command,*tail]
    runpy.run_module(modules[args.command],run_name='__main__')


if __name__=='__main__':main()
