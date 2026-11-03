"""Chronos-2 (amazon/chronos-2): zero-shot forecasting and LoRA / QLoRA fine-tuning.

Variants (``chronos.variants`` in the config):

* ``univariate``        - context = past active power only.
* ``past_covariates``   - + past wind speed / direction / theoretical power / weather.
* ``future_covariates`` - + weather over the forecast window treated as *known*.
  Observed MERRA-2 values stand in for NWP forecasts, which is optimistic;
  the report flags this variant.

Evaluation is rolling-origin: for every forecast origin ``t`` the model sees
``context_length`` hours ending at ``t`` and predicts ``t+1 ... t+H``
(H = max horizon). The prediction for target ``T`` at horizon ``h`` is step
``h`` of the forecast issued at ``T - h``, so origins/targets are identical to
every other model in the harness (paired comparison).

Fine-tuning uses a small custom loop (rather than ``Chronos2Pipeline.fit``)
so that the same code runs on CUDA, MPS and CPU, supports QLoRA (4-bit base
weights via bitsandbytes, CUDA only) and selects checkpoints by rolling-origin
validation RMSE in kW on Sep-Oct.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..device import free_device_memory, get_device, set_seed
from ..evaluate import Forecaster
from ..splits import ForecastTask, chronological_labels

VARIANTS = ("univariate", "past_covariates", "future_covariates")


def _require_chronos():
    try:
        from chronos import Chronos2Pipeline  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "chronos-forecasting is not installed. Run `pip install -e '.[chronos]'` "
            "(or re-run the setup_env script)."
        ) from exc


def load_pipeline(model_id: str = "amazon/chronos-2", device: str | torch.device = "auto"):
    """Load a Chronos-2 pipeline (base model, local dir, or a saved LoRA adapter dir)."""
    _require_chronos()
    from chronos import Chronos2Pipeline

    dev = get_device(str(device)) if not isinstance(device, torch.device) else device
    adapter_meta = Path(model_id) / "windfc_finetune.json"
    if adapter_meta.exists():
        return load_finetuned(Path(model_id), device=dev)[0]
    pipe = Chronos2Pipeline.from_pretrained(model_id)
    pipe.model.to(dev)
    pipe.model.eval()
    return pipe


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def variant_columns(variant: str, cfg: dict) -> tuple[list[str], list[str]]:
    """Return (past covariate columns, known-future covariate columns) for a variant."""
    ccfg = cfg["chronos"]
    if variant == "univariate":
        return [], []
    if variant == "past_covariates":
        return list(ccfg["covariates"]), []
    if variant == "future_covariates":
        fut = list(ccfg["future_covariates"])
        past = list(dict.fromkeys(list(ccfg["covariates"]) + fut))
        return past, fut
    raise ValueError(f"Unknown Chronos variant {variant!r}; choose from {VARIANTS}")


def build_inputs(
    df: pd.DataFrame,
    origins: np.ndarray,
    variant: str,
    cfg: dict,
    context_length: int,
    prediction_length: int,
) -> list[dict]:
    """One input dict per forecast origin (position in ``df``), context ends at the origin."""
    past_cols, fut_cols = variant_columns(variant, cfg)
    y = df["active_power"].to_numpy(dtype=np.float32)
    cov = {c: df[c].to_numpy(dtype=np.float32) for c in past_cols}
    n = len(df)
    inputs = []
    for p in np.asarray(origins, dtype=int):
        lo = max(0, p - context_length + 1)
        item: dict = {"target": y[lo:p + 1]}
        if past_cols:
            item["past_covariates"] = {c: cov[c][lo:p + 1] for c in past_cols}
        if fut_cols:
            idx = np.clip(np.arange(p + 1, p + 1 + prediction_length), 0, n - 1)  # edge-pad past the end
            item["future_covariates"] = {c: cov[c][idx] for c in fut_cols}
        inputs.append(item)
    return inputs


def rolling_forecast(
    pipe,
    df: pd.DataFrame,
    origins: np.ndarray,
    variant: str,
    cfg: dict,
    context_length: int,
    prediction_length: int,
    quantile_levels: list[float] = (0.1, 0.5, 0.9),
    batch_size: int = 256,
    chunk: int = 512,
    verbose: bool = True,
) -> np.ndarray:
    """Forecast from each origin. Returns array (n_origins, prediction_length, n_quantiles) in kW."""
    out = []
    t0 = time.time()
    for start in range(0, len(origins), chunk):
        part = origins[start:start + chunk]
        inputs = build_inputs(df, part, variant, cfg, context_length, prediction_length)
        q, _ = pipe.predict_quantiles(
            inputs,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels),
            batch_size=batch_size,
            context_length=context_length,
        )
        out.append(torch.stack([qi[0] for qi in q]).float().cpu().numpy())  # (n, H, Q) target variate 0
        if verbose:
            done = min(start + chunk, len(origins))
            print(f"    chronos-2 [{variant}] {done}/{len(origins)} origins  [{time.time() - t0:.0f}s]", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, prediction_length, len(quantile_levels)))


class ChronosForecaster(Forecaster):
    """Rolling-origin Chronos-2 forecaster with a per-origin cache shared across horizons."""

    family = "chronos2"

    def __init__(self, pipe, cfg: dict, variant: str = "univariate", context_length: int | None = None,
                 tag: str = "zeroshot", prediction_length: int | None = None, verbose: bool = True):
        self.pipe = pipe
        self.cfg = cfg
        self.variant = variant
        self.context_length = int(context_length or cfg["chronos"]["context_length"])
        self.prediction_length = int(prediction_length or max(cfg["protocol_b"]["horizons"]))
        self.quantile_levels = list(cfg["chronos"].get("quantile_levels", [0.1, 0.5, 0.9]))
        self.batch_size = int(cfg["chronos"].get("batch_size", 256))
        self.name = f"chronos2_{tag}_{variant}"
        self.verbose = verbose
        self._cache: dict[int, np.ndarray] = {}
        self.notes = f"context={self.context_length}" + (
            "; OPTIMISTIC: observed weather used as known future covariates" if variant == "future_covariates" else ""
        )
        self.last_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, task: ForecastTask) -> ChronosForecaster:
        if task.protocol != "B":
            raise ValueError("Chronos-2 is a forecaster; use protocol B")
        return self

    def _ensure(self, task: ForecastTask, origins: np.ndarray) -> None:
        todo = np.array(sorted(set(int(o) for o in origins) - set(self._cache)), dtype=int)
        if len(todo) == 0:
            return
        preds = rolling_forecast(self.pipe, task.df, todo, self.variant, self.cfg, self.context_length,
                                 self.prediction_length, self.quantile_levels, self.batch_size,
                                 verbose=self.verbose)
        for o, p in zip(todo, preds):
            self._cache[int(o)] = p

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:
        h = task.horizon
        if h > self.prediction_length:
            raise ValueError(f"horizon {h} > prediction_length {self.prediction_length}")
        origins = task.positions(split) - h
        self._ensure(task, origins)
        q = np.stack([self._cache[int(o)][h - 1] for o in origins])  # (n, Q)
        levels = self.quantile_levels
        med = q[:, levels.index(0.5)] if 0.5 in levels else q.mean(axis=1)
        if 0.1 in levels and 0.9 in levels:
            lo, hi = q[:, levels.index(0.1)], q[:, levels.index(0.9)]
            self.last_intervals[f"{task.name}_{split}"] = (lo, hi)
        # Power cannot be negative; clip like the other models' physical range.
        return np.clip(med, task.y_scaler.lo[0], None)

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Fine-tuning (LoRA / QLoRA / full)
# ---------------------------------------------------------------------------
def _training_inputs(df: pd.DataFrame, mask: np.ndarray, variant: str, cfg: dict, prediction_length: int):
    from chronos.chronos2 import preprocess

    past_cols, fut_cols = variant_columns(variant, cfg)
    sub = df.loc[mask]
    item: dict = {"target": sub["active_power"].to_numpy(dtype=np.float32)}
    if past_cols:
        item["past_covariates"] = {c: sub[c].to_numpy(dtype=np.float32) for c in past_cols}
    return preprocess.from_list_of_dicts(
        [item], prediction_length=prediction_length, known_covariates_names=fut_cols or None
    )


def _load_base_model(model_id: str, method: str, device: torch.device):
    from chronos.chronos2 import Chronos2Model

    if method == "qlora":
        if device.type != "cuda":
            raise RuntimeError("QLoRA (4-bit) needs an NVIDIA GPU with bitsandbytes; use method=lora on MPS/CPU.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers with bitsandbytes support is required for QLoRA") from exc
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            # keep the small patch-embedding MLPs in full precision
            llm_int8_skip_modules=["input_patch_embedding", "output_patch_embedding"],
        )
        model = Chronos2Model.from_pretrained(model_id, quantization_config=bnb, device_map={"": device.index or 0})
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        return model
    model = Chronos2Model.from_pretrained(model_id)
    return model.to(device)


def _wrap_pipeline(model):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline(model=model)


def _val_score(model, df: pd.DataFrame, val_positions: np.ndarray, variant: str, cfg: dict, context_length: int,
               horizons: list[int], max_origins: int) -> float:
    """Mean rolling-origin RMSE (kW) over horizons on (a subsample of) validation targets."""
    model.eval()
    pipe = _wrap_pipeline(model)
    H = max(horizons)
    targets = val_positions
    if max_origins and len(targets) > max_origins:
        targets = targets[np.linspace(0, len(targets) - 1, max_origins).astype(int)]
    origins = np.unique(np.concatenate([targets - h for h in horizons]))
    preds = rolling_forecast(pipe, df, origins, variant, cfg, context_length, H, [0.5],
                             batch_size=int(cfg["chronos"].get("batch_size", 256)), verbose=False)
    lookup = {int(o): p[:, 0] for o, p in zip(origins, preds)}
    y = df["active_power"].to_numpy()
    scores = []
    for h in horizons:
        yhat = np.array([lookup[int(t - h)][h - 1] for t in targets])
        scores.append(float(np.sqrt(np.mean((yhat - y[targets]) ** 2))))
    model.train()
    return float(np.mean(scores))


def _trainable_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def finetune_chronos(
    df: pd.DataFrame,
    cfg: dict,
    context_length: int,
    learning_rate: float,
    method: str | None = None,
    variant: str | None = None,
    num_steps: int | None = None,
    device: str = "auto",
    seed: int = 0,
    out_dir: Path | None = None,
    verbose: bool = True,
):
    """Fine-tune Chronos-2 on Jan-Aug, select the best step on Sep-Oct. Returns (pipeline, info)."""
    _require_chronos()
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    fcfg = cfg["finetune"]
    method = (method or fcfg.get("method", "lora")).lower()
    variant = variant or fcfg.get("variant", "past_covariates")
    num_steps = int(num_steps or fcfg["num_steps"])
    dev = get_device(device)
    if method == "qlora" and dev.type != "cuda":
        print("[finetune] QLoRA requires CUDA + bitsandbytes; falling back to LoRA on", dev)
        method = "lora"
    set_seed(seed)

    pb = cfg["protocol_b"]
    labels = chronological_labels(df.index, pb["train_end"], pb["val_end"])
    horizons = [int(h) for h in pb["horizons"]]
    H = int(fcfg.get("prediction_length", max(horizons)))
    need = max(context_length, max(horizons))
    val_positions = np.flatnonzero(labels == "val")
    val_positions = val_positions[val_positions >= need + max(horizons)]

    model = _load_base_model(cfg["chronos"]["model_id"], method, dev)
    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model

        lcfg = fcfg["lora"]
        lora = LoraConfig(
            r=int(lcfg["r"]),
            lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg.get("dropout", 0.0)),
            target_modules=list(lcfg["target_modules"]),
        )
        model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[finetune] method={method} variant={variant} ctx={context_length} lr={learning_rate} "
              f"steps={num_steps} device={dev} trainable={n_train:,}/{n_total:,}", flush=True)

    patch = model.chronos_config.output_patch_size
    prepared = _training_inputs(df, labels == "train", variant, cfg, H)
    ds = Chronos2Dataset(prepared, context_length=context_length, prediction_length=H,
                         batch_size=int(fcfg["batch_size"]), output_patch_size=patch,
                         min_past=min(context_length, 64), mode=DatasetMode.TRAIN)
    batches = iter(ds)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=float(fcfg.get("weight_decay", 0.0)))
    warmup = int(float(fcfg.get("warmup_ratio", 0.0)) * num_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (num_steps - step) / max(1, num_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    grad_accum = int(fcfg.get("grad_accum", 1))
    eval_every = int(fcfg.get("eval_every", 200))
    max_norm = float(fcfg.get("max_grad_norm", 1.0))
    select_best = bool(fcfg.get("select_best", True))  # False = keep the final step
    val_max = int(fcfg.get("val_max_origins", 400))

    history: list[dict] = []
    best_score = _val_score(model, df, val_positions, variant, cfg, context_length, horizons, val_max)
    best_state, best_step = _trainable_state(model), 0
    history.append({"step": 0, "val_rmse_kw": best_score})
    if verbose:
        print(f"[finetune] step 0 (zero-shot) val RMSE={best_score:.1f} kW", flush=True)

    model.train()
    t0, running = time.time(), 0.0
    for step in range(1, num_steps + 1):
        for _ in range(grad_accum):
            batch = next(batches)
            batch.pop("target_idx_ranges", None)
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss / grad_accum
            loss.backward()
            running += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % eval_every == 0 or step == num_steps:
            score = _val_score(model, df, val_positions, variant, cfg, context_length, horizons, val_max)
            history.append({"step": step, "train_loss": running / eval_every, "val_rmse_kw": score})
            if verbose:
                print(f"[finetune] step {step}/{num_steps} loss={running / eval_every:.4f} "
                      f"val RMSE={score:.1f} kW (best {min(best_score, score):.1f})  [{time.time() - t0:.0f}s]",
                      flush=True)
            running = 0.0
            if score < best_score or not select_best:
                best_score, best_step, best_state = score, step, _trainable_state(model)

    # restore best trainable weights
    with torch.no_grad():
        named = dict(model.named_parameters())
        for k, v in best_state.items():
            named[k].copy_(v.to(named[k].device, named[k].dtype))
    model.eval()
    # Chronos-2 caps context to its config; make sure longer fine-tuning contexts are honoured.
    model.chronos_config.context_length = max(model.chronos_config.context_length, context_length)
    info = {
        "method": method,
        "variant": variant,
        "context_length": context_length,
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "best_step": best_step,
        "best_val_rmse_kw": best_score,
        "zero_shot_val_rmse_kw": history[0]["val_rmse_kw"],
        "select_best": select_best,
        "trainable_params": n_train,
        "history": history,
        "model_id": cfg["chronos"]["model_id"],
    }
    if out_dir is not None:
        save_finetuned(model, info, Path(out_dir))
    return _wrap_pipeline(model), info


def save_finetuned(model, info: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))  # PEFT -> adapter only; full -> whole model
    (out_dir / "windfc_finetune.json").write_text(json.dumps(info, indent=2, default=float))
    return out_dir


def load_finetuned(path: Path, device: str | torch.device = "auto"):
    """Load a directory written by :func:`save_finetuned`. Returns (pipeline, info)."""
    _require_chronos()
    from chronos import Chronos2Pipeline

    dev = get_device(str(device)) if not isinstance(device, torch.device) else device
    info = json.loads((Path(path) / "windfc_finetune.json").read_text())
    method = info["method"]
    if method == "full":
        pipe = Chronos2Pipeline.from_pretrained(str(path))
        pipe.model.to(dev).eval()
        return pipe, info
    from peft import PeftModel

    if method == "qlora" and dev.type == "cuda":
        base = _load_base_model(info["model_id"], "qlora", dev)
        model = PeftModel.from_pretrained(base, str(path))
    else:  # LoRA (or QLoRA adapter loaded on a non-CUDA machine): merge into fp32 weights
        base = _load_base_model(info["model_id"], "lora", dev)
        model = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
    model.chronos_config.context_length = max(model.chronos_config.context_length, int(info["context_length"]))
    model.eval()
    return _wrap_pipeline(model), info


def grid_search_finetune(df: pd.DataFrame, cfg: dict, device: str = "auto", seed: int = 0,
                         out_root: Path | None = None, verbose: bool = True):
    """Grid over context length x learning rate; keep the best on validation."""
    fcfg = cfg["finetune"]
    results = []
    best = None
    for ctx in fcfg["context_lengths"]:
        for lr in fcfg["learning_rates"]:
            tag = f"ctx{ctx}_lr{lr:g}"
            out_dir = (out_root / tag) if out_root is not None else None
            pipe, info = finetune_chronos(df, cfg, int(ctx), float(lr), device=device, seed=seed,
                                          out_dir=out_dir, verbose=verbose)
            info["tag"] = tag
            info["dir"] = str(out_dir) if out_dir else None
            results.append({k: v for k, v in info.items() if k != "history"})
            if best is None or info["best_val_rmse_kw"] < best[1]["best_val_rmse_kw"]:
                if best is not None:
                    del best[0]
                    free_device_memory()
                best = (pipe, info)
            else:
                del pipe
                free_device_memory()
    return best[0], best[1], results

