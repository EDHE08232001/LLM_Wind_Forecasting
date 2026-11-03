"""Phase 4: ANN / CNN / RNN / LSTM from Karaman (2023).

Protocol A = paper replication (random 75/25, seed 42, 100 epochs, same-hour inputs).
Protocol B = chronological split, lagged inputs, early stopping, several seeds.
"""

from _common import base_arg_parser, load_config

from windfc.pipeline import ensure_processed, stage_paper


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--protocols", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--models", nargs="+", default=None, choices=["ann", "cnn", "rnn", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="protocol-B seeds (default from config)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    ensure_processed(cfg)
    stage_paper(cfg, args.protocols, args.models, args.seeds)


if __name__ == "__main__":
    main()
