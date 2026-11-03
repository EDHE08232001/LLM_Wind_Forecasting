"""Phase 5b: fine-tune Chronos-2 with LoRA (default), QLoRA (4-bit, NVIDIA only) or full fine-tuning.

Grid-searches context length x learning rate on the Sep-Oct validation period
(config `finetune`), saves adapters under checkpoints/chronos2/<method>/ and
evaluates the best one on the Nov-Dec test period for every horizon.

    python scripts/05_finetune_chronos.py                       # LoRA, full grid
    python scripts/05_finetune_chronos.py --method qlora        # QLoRA on CUDA
    python scripts/05_finetune_chronos.py --no-grid --set finetune.num_steps=300
"""

from _common import base_arg_parser, load_config

from windfc.pipeline import ensure_processed, stage_finetune


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--method", choices=["lora", "qlora", "full"], default=None)
    parser.add_argument("--variant", choices=["univariate", "past_covariates", "future_covariates"], default=None)
    parser.add_argument("--no-grid", action="store_true", help="only use the first context length / learning rate")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.variant:
        cfg["finetune"]["variant"] = args.variant
    ensure_processed(cfg)
    stage_finetune(cfg, method=args.method, grid=not args.no_grid)


if __name__ == "__main__":
    main()
