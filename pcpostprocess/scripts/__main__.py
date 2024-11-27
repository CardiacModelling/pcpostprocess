import argparse
import sys

from . import export_staircase_data, run_herg_qc, summarise_herg_export


def main():
    parser = argparse.ArgumentParser(
        usage="pcpostprocess (run_herg_qc | summarise_herg_export) [<args>]",
    )
    parser.add_argument(
        "subcommand",
        choices=["run_herg_qc", "summarise_herg_export", "export_staircase_data"],
    )
    args = parser.parse_args(sys.argv[1:2])

    sys.argv[0] = f"pcpostprocess {args.subcommand}"
    sys.argv.pop(1)  # Subcommand's argparser shouldn't see this

    if args.subcommand == "run_herg_qc":
        run_herg_qc.main()

    elif args.subcommand == "summarise_herg_export":
        summarise_herg_export.main()

    elif args.subcommand == 'export_staircase_data':
        export_staircase_data.main()


if __name__ == "__main__":
    main()
