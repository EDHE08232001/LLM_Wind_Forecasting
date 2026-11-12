"""Chronos-2 zero-shot on the validation and test months, plus the two no-training references.

    python scripts/02_zero_shot.py
    python scripts/02_zero_shot.py --variants paper_inputs
"""

from _common import base_arg_parser, load_config

from windfc.chronos import VARIANTS
from windfc.pipeline import ensure_processed, stage_references, stage_zeroshot


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--variants", nargs="+", default=None, choices=VARIANTS,
                        help="input variants to run (default: chronos.variants in the config)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    ensure_processed(cfg)
    stage_references(cfg)
    stage_zeroshot(cfg, args.variants)


if __name__ == "__main__":
    main()
