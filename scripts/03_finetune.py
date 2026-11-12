"""Fine-tune Chronos-2 with LoRA (default) or QLoRA, then score the best adapter on the test months.

For each input variant, a grid over context length x learning rate is trained on Jan-Aug and scored
on the Sep-Oct validation months. Adapters are saved under checkpoints/chronos2/<method>/<variant>/.

    python scripts/03_finetune.py                          # LoRA, full grid, every variant in the config
    python scripts/03_finetune.py --method qlora           # QLoRA (NVIDIA GPU + bitsandbytes)
    python scripts/03_finetune.py --variants paper_inputs --no-grid
    python scripts/03_finetune.py --set finetune.num_steps=300
    python scripts/03_finetune.py --eval-only              # re-score the saved adapters, no training
"""

from _common import base_arg_parser, load_config

from windfc.chronos import METHODS, VARIANTS
from windfc.pipeline import ensure_processed, stage_evaluate_saved, stage_finetune


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--method", choices=METHODS, default=None, help="default: finetune.method in the config")
    parser.add_argument("--variants", nargs="+", default=None, choices=VARIANTS,
                        help="default: finetune.variants in the config")
    parser.add_argument("--no-grid", action="store_true",
                        help="train only the first context length and learning rate of the grid")
    parser.add_argument("--eval-only", nargs="*", default=None, metavar="ADAPTER_DIR",
                        help="skip training; score the given adapter directories (or the best saved ones)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    ensure_processed(cfg)
    if args.eval_only is not None:
        stage_evaluate_saved(cfg, args.eval_only)
    else:
        stage_finetune(cfg, method=args.method, variants=args.variants, grid=not args.no_grid)


if __name__ == "__main__":
    main()
