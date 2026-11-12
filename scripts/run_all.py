"""Run the whole pipeline: download -> prepare -> zero-shot -> fine-tune -> report.

    python scripts/run_all.py                              # everything, real data, LoRA
    python scripts/run_all.py --method qlora               # QLoRA instead (NVIDIA GPU)
    python scripts/run_all.py --config configs/smoke.yaml  # fast end-to-end check on synthetic data
    python scripts/run_all.py --skip download finetune     # leave stages out
    python scripts/run_all.py --only report                # run just some stages
"""

import time

from _common import base_arg_parser, load_config

from windfc.chronos import METHODS
from windfc.config import resolve_path
from windfc.device import describe_environment, free_device_memory
from windfc.download import download_scada, download_weather_nasa_power
from windfc.pipeline import stage_finetune, stage_prepare, stage_references, stage_zeroshot
from windfc.report import make_report
from windfc.synthetic import make_synthetic

STAGES = ["download", "prepare", "zeroshot", "finetune", "report"]


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    parser.add_argument("--only", nargs="*", default=None, choices=STAGES)
    parser.add_argument("--method", choices=METHODS, default=None, help="fine-tuning method (default: config)")
    parser.add_argument("--no-grid", action="store_true", help="fine-tune a single grid point per variant")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    synthetic = cfg.get("name") == "smoke"
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
                    print(f"[synthetic] wrote fake raw files to {scada.parent} (NOT real data)")
            else:
                download_scada(scada)
                download_weather_nasa_power(weather, cfg["site"]["latitude"], cfg["site"]["longitude"],
                                            year=int(cfg["data"]["year"]))
        elif stage == "prepare":
            stage_prepare(cfg)
        elif stage == "zeroshot":
            stage_references(cfg)
            stage_zeroshot(cfg)
        elif stage == "finetune":
            stage_finetune(cfg, method=args.method, grid=not args.no_grid)
        elif stage == "report":
            make_report(cfg)
        free_device_memory()
        timings[stage] = time.time() - t0
    print("\nStage timings: " + ", ".join(f"{k}={v:.0f}s" for k, v in timings.items()))


if __name__ == "__main__":
    main()
