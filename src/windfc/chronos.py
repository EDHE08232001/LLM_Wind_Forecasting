"""Chronos-2 (``amazon/chronos-2``): zero-shot forecasting and LoRA / QLoRA fine-tuning.

Input variants (``chronos.variants`` in the config):

* ``univariate``      - context = past active power only.
* ``past_covariates`` - + past wind speed / direction / theoretical power / weather.
  A true forecast: nothing about the target hour is known.
* ``paper_inputs``    - + the same covariates at the target hours, given as *known
  future covariates*. This is the information the paper's networks had (same-hour
  wind speed, theoretical power and weather), so it is the variant to compare with
  the paper's Table 5. It is an estimation task, not a forecast, unless real wind
  and weather forecasts replace the observed values.

Evaluation is rolling-origin: for every origin ``t`` the model sees
``context_length`` hours ending at ``t`` and predicts ``t+1 ... t+H``. The
prediction for target ``T`` at horizon ``h`` is step ``h`` of the forecast issued
at ``T - h``.

Fine-tuning uses a small training loop of its own (rather than
``Chronos2Pipeline.fit``) so that one code path runs on CUDA, MPS and CPU,
supports QLoRA (frozen base weights in 4-bit NF4 via bitsandbytes, NVIDIA only),
and can pick the best step by rolling-origin validation RMSE in kW. Step 0 (the
zero-shot weights) takes part in that selection, so fine-tuning can never make
the selected model worse on validation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .device import free_device_memory, get_device, set_seed
from .splits import ForecastTask, split_labels

VARIANTS = ("univariate", "past_covariates", "paper_inputs")
METHODS = ("lora", "qlora")
META_FILE = "windfc_finetune.json"


def _require_chronos() -> None:
    try:
        from chronos import Chronos2Pipeline  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "chronos-forecasting is not installed. Re-run the setup_env script (or `pip install -e .`)."
        ) from exc


def load_pipeline(model_id: str = "amazon/chronos-2", device: str | torch.device = "auto"):
    """Load a Chronos-2 pipeline: the base model, or a directory written by :func:`save_finetuned`."""
    _require_chronos()
    from chronos import Chronos2Pipeline

    dev = get_device(str(device)) if not isinstance(device, torch.device) else device
    if (Path(model_id) / META_FILE).exists():
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
    covariates = list(cfg["chronos"]["covariates"])
    if variant == "univariate":
        return [], []
    if variant == "past_covariates":
        return covariates, []
    if variant == "paper_inputs":
        return covariates, covariates
    raise ValueError(f"Unknown Chronos variant {variant!r}; choose from {VARIANTS}")


def build_inputs(df: pd.DataFrame, origins: np.ndarray, variant: str, cfg: dict, context_length: int,
                 prediction_length: int) -> list[dict]:
    """One input dict per forecast origin (a row position in ``df``); the context ends at the origin."""
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
            # Past the end of the data the last value is repeated; those steps are never scored.
            idx = np.clip(np.arange(p + 1, p + 1 + prediction_length), 0, n - 1)
            item["future_covariates"] = {c: cov[c][idx] for c in fut_cols}
        inputs.append(item)
    return inputs


def rolling_forecast(pipe, df: pd.DataFrame, origins: np.ndarray, variant: str, cfg: dict, context_length: int,
                     prediction_length: int, quantile_levels: list[float] = (0.1, 0.5, 0.9),
                     batch_size: int = 256, chunk: int = 512, verbose: bool = True) -> np.ndarray:
    """Forecast from each origin. Returns an array (n_origins, prediction_length, n_quantiles) in kW."""
    out = []
    t0 = time.time()
    for start in range(0, len(origins), chunk):
        part = origins[start:start + chunk]
        inputs = build_inputs(df, part, variant, cfg, context_length, prediction_length)
        q, _ = pipe.predict_quantiles(inputs, prediction_length=prediction_length,
                                      quantile_levels=list(quantile_levels), batch_size=batch_size,
                                      context_length=context_length)
        out.append(torch.stack([qi[0] for qi in q]).float().cpu().numpy())  # variate 0 = active power
        if verbose:
            done = min(start + chunk, len(origins))
            print(f"    chronos-2 [{variant}] {done}/{len(origins)} origins  [{time.time() - t0:.0f}s]", flush=True)
    return np.concatenate(out, axis=0) if out else np.zeros((0, prediction_length, len(quantile_levels)))


class ChronosForecaster:
    """Rolling-origin Chronos-2 forecaster. Forecasts are cached per origin and shared across horizons."""

    family = "chronos2"

    def __init__(self, pipe, cfg: dict, variant: str = "univariate", context_length: int | None = None,
                 tag: str = "zeroshot", verbose: bool = True):
        self.pipe = pipe
        self.cfg = cfg
        self.variant = variant
        self.context_length = int(context_length or cfg["chronos"]["context_length"])
        self.prediction_length = int(max(cfg["evaluation"]["horizons"]))
        self.quantile_levels = [0.1, 0.5, 0.9]
        self.batch_size = int(cfg["chronos"].get("batch_size", 256))
        self.name = f"chronos2_{tag}_{variant}"
        self.verbose = verbose
        self.notes = f"context={self.context_length}h"
        self._cache: dict[int, np.ndarray] = {}

    def predict(self, task: ForecastTask, split: str) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Return (median forecast, (q10, q90)) in kW for every target of ``split``."""
        h = task.horizon
        if h > self.prediction_length:
            raise ValueError(f"horizon {h} > prediction_length {self.prediction_length}")
        origins = task.positions(split) - h
        todo = np.array(sorted({int(o) for o in origins} - set(self._cache)), dtype=int)
        if len(todo):
            preds = rolling_forecast(self.pipe, task.df, todo, self.variant, self.cfg, self.context_length,
                                     self.prediction_length, self.quantile_levels, self.batch_size,
                                     verbose=self.verbose)
            self._cache.update({int(o): p for o, p in zip(todo, preds)})
        q = np.stack([self._cache[int(o)][h - 1] for o in origins])  # (n, 3)
        q = np.sort(q, axis=1)  # quantile heads are independent and can cross slightly; restore the order
        # A turbine cannot output less than the lowest power seen in training (about zero).
        lo, med, hi = (np.clip(q[:, i], task.y_min, None) for i in range(3))
        return med, (lo, hi)

    def clear_cache(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Fine-tuning (LoRA / QLoRA)
# ---------------------------------------------------------------------------
def _training_inputs(df: pd.DataFrame, mask: np.ndarray, variant: str, cfg: dict, prediction_length: int):
    from chronos.chronos2 import preprocess

    past_cols, fut_cols = variant_columns(variant, cfg)
    sub = df.loc[mask]
    item: dict = {"target": sub["active_power"].to_numpy(dtype=np.float32)}
    if past_cols:
        item["past_covariates"] = {c: sub[c].to_numpy(dtype=np.float32) for c in past_cols}
    return preprocess.from_list_of_dicts([item], prediction_length=prediction_length,
                                         known_covariates_names=fut_cols or None)


def _load_base_model(model_id: str, method: str, device: torch.device):
    """Base model in fp32 (LoRA) or with frozen 4-bit NF4 weights (QLoRA, NVIDIA only)."""
    from chronos.chronos2 import Chronos2Model

    if method != "qlora":
        return Chronos2Model.from_pretrained(model_id).to(device)
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        # The small patch-embedding MLPs stay in full precision; only the transformer is quantised.
        llm_int8_skip_modules=["input_patch_embedding", "output_patch_embedding"],
    )
    model = Chronos2Model.from_pretrained(model_id, quantization_config=bnb, device_map={"": device.index or 0})
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


def qlora_unavailable_reason(device: torch.device) -> str | None:
    """Why QLoRA cannot run here, or None if it can."""
    if device.type != "cuda":
        return f"QLoRA needs an NVIDIA GPU (4-bit bitsandbytes kernels are CUDA-only); this machine uses {device}"
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # ImportError, or a broken CUDA setup
        return f"bitsandbytes is not usable ({type(exc).__name__}: {exc}); run the setup script with the qlora option"
    return None


def _wrap_pipeline(model):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline(model=model)


def _val_score(model, df: pd.DataFrame, val_positions: np.ndarray, variant: str, cfg: dict, context_length: int,
               horizons: list[int], max_origins: int) -> float:
    """Mean rolling-origin RMSE (kW) over the horizons on (a subsample of) the validation targets."""
    model.eval()
    targets = val_positions
    if max_origins and len(targets) > max_origins:
        targets = targets[np.linspace(0, len(targets) - 1, max_origins).astype(int)]
    origins = np.unique(np.concatenate([targets - h for h in horizons]))
    preds = rolling_forecast(_wrap_pipeline(model), df, origins, variant, cfg, context_length, max(horizons),
                             [0.5], batch_size=int(cfg["chronos"].get("batch_size", 256)), verbose=False)
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


def finetune_chronos(df: pd.DataFrame, cfg: dict, context_length: int, learning_rate: float,
                     method: str | None = None, variant: str | None = None, num_steps: int | None = None,
                     device: str = "auto", seed: int = 0, out_dir: Path | None = None, verbose: bool = True):
    """Fine-tune on the training months, select the best step on validation. Returns (pipeline, info)."""
    _require_chronos()
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    from peft import LoraConfig, get_peft_model

    fcfg = cfg["finetune"]
    method = (method or fcfg.get("method", "lora")).lower()
    if method not in METHODS:
        raise ValueError(f"Unknown fine-tuning method {method!r}; choose from {METHODS}")
    variant = variant or fcfg["variants"][0]
    num_steps = int(num_steps or fcfg["num_steps"])
    dev = get_device(device)
    requested = method
    if method == "qlora":
        reason = qlora_unavailable_reason(dev)
        if reason:
            print(f"[finetune] {reason}.\n[finetune] Falling back to LoRA (full-precision base weights).")
            method = "lora"
    set_seed(seed)

    labels = split_labels(df, cfg)
    horizons = [int(h) for h in cfg["evaluation"]["horizons"]]
    pred_len = int(fcfg.get("prediction_length", max(horizons)))
    val_positions = np.flatnonzero(labels == "val")
    val_positions = val_positions[val_positions >= context_length + max(horizons)]

    model = _load_base_model(cfg["chronos"]["model_id"], method, dev)
    lcfg = fcfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=int(lcfg["r"]), lora_alpha=int(lcfg["alpha"]), lora_dropout=float(lcfg.get("dropout", 0.0)),
        target_modules=list(lcfg["target_modules"]),
    ))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[finetune] method={method} variant={variant} context={context_length}h lr={learning_rate:g} "
              f"steps={num_steps} device={dev} trainable={n_train:,} of {n_total:,} parameters "
              f"({100 * n_train / n_total:.2f}%)", flush=True)

    prepared = _training_inputs(df, labels == "train", variant, cfg, pred_len)
    ds = Chronos2Dataset(prepared, context_length=context_length, prediction_length=pred_len,
                         batch_size=int(fcfg["batch_size"]), output_patch_size=model.chronos_config.output_patch_size,
                         min_past=min(context_length, 64), mode=DatasetMode.TRAIN)
    batches = iter(ds)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=learning_rate, weight_decay=float(fcfg.get("weight_decay", 0.0)))
    warmup = int(float(fcfg.get("warmup_ratio", 0.0)) * num_steps)

    def lr_lambda(step: int) -> float:  # linear warm-up, then linear decay to zero
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (num_steps - step) / max(1, num_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    eval_every = int(fcfg.get("eval_every", 200))
    max_norm = float(fcfg.get("max_grad_norm", 1.0))
    select_best = bool(fcfg.get("select_best", True))
    val_max = int(fcfg.get("val_max_origins", 400))

    best_score = _val_score(model, df, val_positions, variant, cfg, context_length, horizons, val_max)
    zero_shot_score = best_score
    best_state, best_step = _trainable_state(model), 0
    history: list[dict] = [{"step": 0, "val_rmse_kw": best_score}]
    if verbose:
        print(f"[finetune] step 0 (zero-shot) val RMSE={best_score:.1f} kW", flush=True)

    model.train()
    t0, running, since_eval = time.time(), 0.0, 0
    for step in range(1, num_steps + 1):
        batch = next(batches)
        batch.pop("target_idx_ranges", None)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += float(loss.detach())
        since_eval += 1
        if step % eval_every == 0 or step == num_steps:
            score = _val_score(model, df, val_positions, variant, cfg, context_length, horizons, val_max)
            history.append({"step": step, "train_loss": running / since_eval, "val_rmse_kw": score})
            if verbose:
                print(f"[finetune] step {step}/{num_steps} loss={running / since_eval:.4f} val RMSE={score:.1f} kW "
                      f"(best so far {min(best_score, score):.1f})  [{time.time() - t0:.0f}s]", flush=True)
            running, since_eval = 0.0, 0
            if score < best_score or not select_best:
                best_score, best_step, best_state = score, step, _trainable_state(model)

    with torch.no_grad():  # restore the selected step
        named = dict(model.named_parameters())
        for k, v in best_state.items():
            named[k].copy_(v.to(named[k].device, named[k].dtype))
    model.eval()
    info = {
        "method": method,
        "requested_method": requested,
        "variant": variant,
        "context_length": int(context_length),
        "learning_rate": float(learning_rate),
        "num_steps": num_steps,
        "best_step": best_step,
        "best_val_rmse_kw": best_score,
        "zero_shot_val_rmse_kw": zero_shot_score,
        "select_best": select_best,
        "trainable_params": n_train,
        "total_params": n_total,
        "train_seconds": round(time.time() - t0, 1),
        "device": str(dev),
        "seed": seed,
        "lora": dict(lcfg),
        "history": history,
        "model_id": cfg["chronos"]["model_id"],
    }
    if out_dir is not None:
        save_finetuned(model, info, Path(out_dir))
    return _wrap_pipeline(model), info


def save_finetuned(model, info: dict, out_dir: Path) -> Path:
    """Write the LoRA adapter (a few MB; the base weights are not copied) plus run metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    (out_dir / META_FILE).write_text(json.dumps(info, indent=2, default=float))
    return out_dir


def load_finetuned(path: Path, device: str | torch.device = "auto"):
    """Load a directory written by :func:`save_finetuned`. Returns (pipeline, info)."""
    _require_chronos()
    from peft import PeftModel

    dev = get_device(str(device)) if not isinstance(device, torch.device) else device
    info = json.loads((Path(path) / META_FILE).read_text())
    if info["method"] == "qlora" and qlora_unavailable_reason(dev) is None:
        # Keep the 4-bit base the adapter was trained against.
        model = PeftModel.from_pretrained(_load_base_model(info["model_id"], "qlora", dev), str(path))
    else:
        if info["method"] == "qlora":
            print("[chronos] QLoRA adapter loaded on full-precision base weights (no 4-bit support here); "
                  "results can differ slightly from the 4-bit run.")
        model = PeftModel.from_pretrained(_load_base_model(info["model_id"], "lora", dev), str(path))
        model = model.merge_and_unload()
    model.eval()
    return _wrap_pipeline(model), info


def grid_search_finetune(df: pd.DataFrame, cfg: dict, variant: str, method: str | None = None,
                         device: str = "auto", seed: int = 0, out_root: Path | None = None, grid: bool = True,
                         verbose: bool = True):
    """Fine-tune over context length x learning rate; keep the best on validation.

    Returns (best pipeline, best info, one summary row per grid point).
    """
    fcfg = cfg["finetune"]
    contexts = [int(c) for c in fcfg["context_lengths"]]
    rates = [float(r) for r in fcfg["learning_rates"]]
    if not grid:
        contexts, rates = contexts[:1], rates[:1]
    results, best = [], None
    for ctx in contexts:
        for lr in rates:
            tag = f"ctx{ctx}_lr{lr:g}"
            pipe, info = finetune_chronos(df, cfg, ctx, lr, method=method, variant=variant, device=device,
                                          seed=seed, out_dir=(out_root / tag) if out_root else None,
                                          verbose=verbose)
            info["tag"] = tag
            info["dir"] = str(out_root / tag) if out_root else None
            results.append({k: v for k, v in info.items() if k not in ("history", "lora")})
            if best is None or info["best_val_rmse_kw"] < best[1]["best_val_rmse_kw"]:
                best = (pipe, info)
            del pipe
            free_device_memory()
    return best[0], best[1], results
