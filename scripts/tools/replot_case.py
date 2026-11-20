"""Replot saved representative steady/fine transient fields without solving."""
import argparse
from pathlib import Path
from thermal_opd.plotting import replot_saved_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(f'Rendered: {replot_saved_case(args.results,output_dir=args.output_dir)}')


if __name__=='__main__':
    main()
