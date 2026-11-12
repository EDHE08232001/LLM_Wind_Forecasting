"""Build the tables and figures (including the comparison with the paper's Table 5) from results/."""

from _common import base_arg_parser, load_config

from windfc.report import make_report


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    make_report(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
