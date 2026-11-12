"""Download the SCADA (Kaggle) and MERRA-2 weather (NASA POWER) files into data/raw/.

    python scripts/00_download_data.py                 # real data
    python scripts/00_download_data.py --synthetic     # fake data in the same format (smoke config)
"""

from _common import base_arg_parser, load_config

from windfc.config import resolve_path
from windfc.download import download_scada, download_weather_nasa_power
from windfc.synthetic import make_synthetic


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--synthetic", action="store_true", help="write synthetic files instead of downloading")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--only", choices=["scada", "weather"], help="download only one of the two files")
    args = parser.parse_args()
    if args.synthetic and args.config.endswith("default.yaml"):
        args.config = "configs/smoke.yaml"
    cfg = load_config(args.config, args.overrides)
    scada_csv = resolve_path(cfg["paths"]["scada_csv"])
    weather_csv = resolve_path(cfg["paths"]["weather_csv"])

    if args.synthetic:
        s, w = make_synthetic(scada_csv.parent, year=int(cfg["data"]["year"]), seed=int(cfg["seed"]))
        print(f"[synthetic] wrote {s} and {w} (NOT real data)")
        return

    failures = []
    if args.only in (None, "scada"):
        try:
            download_scada(scada_csv, force=args.force)
        except Exception as exc:
            failures.append(f"SCADA: {exc}")
    if args.only in (None, "weather"):
        try:
            site = cfg["site"]
            download_weather_nasa_power(weather_csv, site["latitude"], site["longitude"],
                                        year=int(cfg["data"]["year"]), force=args.force)
        except Exception as exc:
            failures.append(f"weather: {exc}")
    if failures:
        print("\nSome downloads failed:\n  " + "\n  ".join(failures))
        print("See README.md -> 'Dataset' for the manual download steps.")
        raise SystemExit(1)
    print("\nAll raw files present. Next: python scripts/01_prepare_data.py")


if __name__ == "__main__":
    main()
