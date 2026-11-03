"""Run the whole pipeline: download -> prepare -> baselines -> paper nets -> Chronos-2 -> fine-tune -> report.

    python scripts/run_all.py                              # everything (real data)
    python scripts/run_all.py --config configs/smoke.yaml  # fast end-to-end check on synthetic data
    python scripts/run_all.py --skip finetune chronos      # skip stages
    python scripts/run_all.py --only baselines report
"""

import time

from _common import base_arg_parser, load_config

from windfc.config import resolve_path
from windfc.device import describe_environment, free_device_memory
from windfc.download import download_scada, download_weather_nasa_power
from windfc.pipeline import stage_baselines, stage_chronos, stage_finetune, stage_paper, stage_prepare
from windfc.report import make_report
from windfc.synthetic import make_synthetic

STAGES = ["download", "prepare", "baselines", "paper", "chronos", "finetune", "report"]


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    parser.add_argument("--only", nargs="*", default=None, choices=STAGES)
    parser.add_argument("--synthetic", action="store_true",
                        help="generate synthetic raw files instead of downloading (implied by smoke config)")
    parser.add_argument("--finetune-method", choices=["lora", "qlora", "full"], default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    synthetic = args.synthetic or cfg.get("name") == "smoke"
    stages = args.only if args.only else [s for s in STAGES if s not in args.skip]
    print(describe_environment())
    timings = {}
    for stage in stages:
        t0 = time.time()
        if stage == "download":
            scada, weather = resolve_path(cfg["paths"]["scada_csv"]), resolve_path(cfg["paths"]["weather_csv"])
            if synthetic:
                if not (scada.exists() and weather.exists()):
                    make_synthetic(scada.parent, year=int(cfg["data"]["year"]), seed=int(cfg["seed"]))
            else:
                download_scada(scada)
                download_weather_nasa_power(weather, cfg["site"]["latitude"], cfg["site"]["longitude"],
                                            year=int(cfg["data"]["year"]))
        elif stage == "prepare":
            stage_prepare(cfg)
        elif stage == "baselines":
            stage_baselines(cfg)
        elif stage == "paper":
            stage_paper(cfg)
        elif stage == "chronos":
            stage_chronos(cfg)
        elif stage == "finetune":
            stage_finetune(cfg, method=args.finetune_method, grid=True)
        elif stage == "report":
            make_report(cfg)
        free_device_memory()
        timings[stage] = time.time() - t0
    print("\nStage timings: " + ", ".join(f"{k}={v:.0f}s" for k, v in timings.items()))


if __name__ == "__main__":
    main()
