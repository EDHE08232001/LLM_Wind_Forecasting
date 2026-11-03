"""Phase 1: load, impute (kNN), resample, merge and flag -> data/processed/hourly_clean.parquet."""

from _common import base_arg_parser, load_config

from windfc.pipeline import stage_prepare


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    stage_prepare(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
