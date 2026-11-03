"""Phase 5a: Chronos-2 zero-shot (univariate / past covariates / known-future covariates)."""

from _common import base_arg_parser, load_config

from windfc.pipeline import ensure_processed, stage_chronos


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=["univariate", "past_covariates", "future_covariates"])
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    ensure_processed(cfg)
    stage_chronos(cfg, args.variants)


if __name__ == "__main__":
    main()
