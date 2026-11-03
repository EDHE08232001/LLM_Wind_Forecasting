"""Training loop for the paper networks (works on CUDA, MPS and CPU)."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .device import free_device_memory, get_device, set_seed
from .evaluate import Forecaster
from .models.paper_nets import build_net, count_params
from .splits import ForecastTask


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                       torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None,
                      num_workers=num_workers, drop_last=False)


@torch.no_grad()
def predict_array(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch_size], dtype=np.float32)).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_net(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int | None = 10,
    device: torch.device | str = "cpu",
    seed: int = 0,
    num_workers: int = 0,
    verbose: bool = True,
    log_every: int = 10,
) -> tuple[nn.Module, dict]:
    """Adam + MSE (as in the paper). Early stopping on validation if given."""
    device = torch.device(device)
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_dl = _loader(x_train, y_train, batch_size, True, seed, num_workers)
    has_val = x_val is not None and len(x_val) > 0
    history = {"train_loss": [], "val_loss": []}
    best_state, best_val, bad_epochs, best_epoch = None, np.inf, 0, -1
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        history["train_loss"].append(total / max(count, 1))
        if has_val:
            pv = predict_array(model, x_val, device)
            val_loss = float(np.mean((pv - y_val) ** 2))
            history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-7:
                best_val, bad_epochs, best_epoch = val_loss, 0, epoch
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
        if verbose and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            msg = f"    epoch {epoch:3d}/{epochs} train_mse={history['train_loss'][-1]:.5f}"
            if has_val:
                msg += f" val_mse={history['val_loss'][-1]:.5f} (best {best_val:.5f} @ {best_epoch})"
            print(msg + f"  [{time.time() - t0:.0f}s]", flush=True)
        if has_val and patience and bad_epochs >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch if has_val else epochs
    history["seconds"] = time.time() - t0
    return model, history


class NetForecaster(Forecaster):
    """Wraps a paper net so it fits the common :class:`Forecaster` interface."""

    family = "paper_net"

    def __init__(self, arch: str, train_cfg: dict, device: str = "auto", seed: int = 0,
                 use_early_stopping: bool = True, checkpoint_dir: Path | None = None, verbose: bool = True):
        self.arch = arch
        self.name = arch
        self.cfg = train_cfg
        self.device = get_device(device)
        self.seed = seed
        self.use_early_stopping = use_early_stopping
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

    def fit(self, task: ForecastTask) -> NetForecaster:
        x_tr, y_tr = task.windows("train")
        x_va, y_va = task.windows("val") if (task.has_val and self.use_early_stopping) else (None, None)
        set_seed(self.seed)
        self.model = build_net(self.arch, seq_len=x_tr.shape[1], channels=x_tr.shape[2])
        if self.verbose:
            print(f"  [{self.arch}] params={count_params(self.model):,} input={tuple(x_tr.shape[1:])} "
                  f"device={self.device} seed={self.seed}", flush=True)
        self.model, self.history = train_net(
            self.model, x_tr, y_tr, x_va, y_va,
            epochs=int(self.cfg["epochs"]),
            batch_size=int(self.cfg["batch_size"]),
            lr=float(self.cfg["lr"]),
            patience=int(self.cfg.get("patience", 10)) if x_va is not None else None,
            device=self.device,
            seed=self.seed,
            num_workers=int(self.cfg.get("num_workers", 0)),
            verbose=self.verbose,
        )
        self.notes = f"best_epoch={self.history['best_epoch']}"
        if self.checkpoint_dir is not None and self.cfg.get("save_checkpoints", True):
            self.save(Path(self.checkpoint_dir) / f"{task.name}__{self.arch}__seed{self.seed}.pt", task)
        return self

    def predict(self, task: ForecastTask, split: str) -> np.ndarray:
        x, _ = task.windows(split)
        return task.unscale_y(predict_array(self.model, x, self.device))

    def save(self, path: Path, task: ForecastTask) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": self.arch,
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "seq_len": task.windows("train")[0].shape[1],
                "channels": len(task.window_columns),
                "window_columns": task.window_columns,
                "protocol": task.protocol,
                "horizon": task.horizon,
                "x_lo": task.x_scaler.lo, "x_hi": task.x_scaler.hi,
                "y_lo": task.y_scaler.lo, "y_hi": task.y_scaler.hi,
                "history": self.history,
            },
            path,
        )
        return path

    def release(self) -> None:
        self.model = self.model.cpu()
        free_device_memory()


def load_net_checkpoint(path: Path, device: str = "auto") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_net(ckpt["arch"], ckpt["seq_len"], ckpt["channels"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(get_device(device)).eval(), ckpt
