"""Phase 3: persistence, seasonal naive, linear regression and XGBoost (protocols A and B)."""

from _common import base_arg_parser, load_config

from windfc.pipeline import ensure_processed, stage_baselines


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--protocols", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of: persistence seasonal_naive linear xgboost")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    ensure_processed(cfg)
    stage_baselines(cfg, args.protocols, args.models)


if __name__ == "__main__":
    main()
