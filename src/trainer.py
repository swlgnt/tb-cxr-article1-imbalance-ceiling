"""Pelatihan & evaluasi satu run (satu kombinasi desain-rasio-strategi-model-seed)."""
from __future__ import annotations

import datetime as dt
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import BalancedOversampleSampler, CXRDataset, build_train_set, build_transforms
from .losses import build_loss
from .metrics import bootstrap_ci, compute_metrics
from .models import build_model, count_params, set_backbone_trainable
from .utils import git_info, make_run_id, save_json, seed_worker, set_seed


# ----------------------------------------------------------------------------- helper
def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def predict(model, loader, device, amp_dtype=None, criterion=None):
    model.eval()
    probs, labels, idxs, losses = [], [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            losses.append(criterion(logits, y).item() * len(y))
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        labels.append(y.cpu().numpy())
        idxs.append(i.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    loss = float(np.sum(losses) / len(labels)) if losses else float("nan")
    return probs, labels, np.concatenate(idxs), loss


def predictions_frame(df: pd.DataFrame, probs, labels, idxs, class_names) -> pd.DataFrame:
    sub = df.reset_index(drop=True).iloc[idxs]
    out = pd.DataFrame({"image_id": sub["image_id"].values, "path": sub["path"].values, "label": labels})
    for j, c in enumerate(class_names):
        out[f"prob_{c}"] = probs[:, j]
    out["pred"] = probs.argmax(1)
    out["correct"] = (out["pred"] == out["label"]).astype(int)
    return out.sort_values("image_id").reset_index(drop=True)


def _make_loader(ds, bs, shuffle, sampler, workers, seed, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=(shuffle and sampler is None), sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      worker_init_fn=seed_worker, generator=g, persistent_workers=workers > 0)


# ----------------------------------------------------------------------------- run
def run_one(cfg: dict, design: str, ratio: str, strategy: str, model_key: str, seed: int,
            train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
            runs_dir: str | Path, logger, overwrite: bool = False) -> dict:
    class_names = cfg["data"]["class_names"]
    key_prefixes = cfg["data"]["short_names"]
    tcfg, scfg = cfg["train"], cfg["strategies"][strategy]
    mcfg = cfg["models"][model_key]
    run_id = make_run_id(design, ratio, strategy, model_key, seed)
    out = Path(runs_dir) / run_id
    if (out / "metrics.json").exists() and not overwrite:
        logger.info(f"[SKIP] {run_id} sudah selesai")
        return {"run_id": run_id, "status": "skipped"}
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed, tcfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data latih (asli + tambahan klasik/GAN sesuai strategi)
    train_df = build_train_set(cfg, strategy, ratio, train_df, cfg["paths"]["work_dir"])
    comp = train_df.groupby("source").size().to_dict()
    n_syn = int(comp.get("gan", 0))
    n_classic = int(comp.get("classic", 0))
    if n_syn or n_classic:
        logger.info(f"Data latih: asli {comp.get('real', 0)} + salinan klasik {n_classic} + sintetis GAN {n_syn}")
    if scfg.get("synthetic_csv") and "dose" not in scfg and scfg.get("synthetic_mode", "balance") == "balance":
        cnt = train_df["label"].value_counts()
        short = {class_names[c]: int(cnt.max() - cnt.get(c, 0)) for c in
                 sorted(set(train_df.loc[train_df["source"] == "gan", "label"])) if cnt.get(c, 0) < cnt.max()}
        if short:
            logger.warning(f"Stok citra sintetis kurang; data latih belum seimbang. Kekurangan per kelas: {short}")

    # ---------------- model
    model, data_cfg = build_model(mcfg["timm_name"], len(class_names), tcfg.get("pretrained", True),
                                  mcfg.get("drop_rate", tcfg.get("drop_rate", 0.0)))
    img_size = mcfg.get("img_size") or data_cfg["input_size"][-1]
    mean, std = data_cfg["mean"], data_cfg["std"]
    model = model.to(device)
    if tcfg.get("channels_last") and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    path_col = "path_cached" if ("path_cached" in train_df and tcfg.get("use_cache", True)) else "path"
    if path_col == "path_cached" and train_df["path_cached"].isna().any():  # sintetis tak ter-cache
        train_df["path_cached"] = train_df["path_cached"].fillna(train_df["path"])
    tf_train = build_transforms(img_size, True, scfg.get("augment", False), cfg["augmentation"], mean, std,
                                cfg["data"].get("clahe"), cfg["data"])
    tf_eval = build_transforms(img_size, False, False, cfg["augmentation"], mean, std, cfg["data"].get("clahe"), cfg["data"])
    tf_copy = None
    if train_df["aug_copy"].any():  # salinan klasik: augmentasi hanya pada salinan tambahan
        tf_copy = build_transforms(img_size, True, True, cfg["augmentation"], mean, std, cfg["data"].get("clahe"), cfg["data"])
    ds_tr = CXRDataset(train_df, tf_train, path_col, copy_transform=tf_copy,
                       copy_mode=scfg.get("classic_mode", "fixed"),
                       copy_seed=int(cfg.get("dose_design", {}).get("selection_seed", cfg["split"]["seed"])))
    ds_va = CXRDataset(val_df, tf_eval, path_col if path_col in val_df else "path")
    ds_te = CXRDataset(test_df, tf_eval, path_col if path_col in test_df else "path")

    bs, nw = mcfg.get("batch_size", tcfg["batch_size"]), tcfg.get("num_workers", 4)
    sampler = BalancedOversampleSampler(ds_tr.labels, seed) if scfg.get("balance") == "oversample" else None
    dl_tr = _make_loader(ds_tr, bs, True, sampler, nw, seed, drop_last=True)
    dl_va = _make_loader(ds_va, bs * 2, False, None, nw, seed)
    dl_te = _make_loader(ds_te, bs * 2, False, None, nw, seed)

    criterion = build_loss(scfg.get("loss", "ce"), ds_tr.labels, len(class_names), tcfg).to(device)
    eval_criterion = torch.nn.CrossEntropyLoss()  # loss val selalu CE biasa agar sebanding antar strategi

    lr = mcfg.get("lr", tcfg["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg.get("weight_decay", 1e-4))
    epochs = tcfg["epochs"]
    spe = len(dl_tr)
    scheduler = _cosine_with_warmup(optimizer, int(tcfg.get("warmup_epochs", 1) * spe), epochs * spe,
                                    tcfg.get("min_lr_ratio", 0.01))
    amp_dtype = None
    if device.type == "cuda" and tcfg.get("amp", True):
        amp_dtype = torch.bfloat16 if (tcfg.get("amp_dtype") == "bf16" and torch.cuda.is_bf16_supported()) \
            else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    monitor, mode = tcfg.get("monitor", "macro_f1"), tcfg.get("monitor_mode", "max")
    patience, min_delta = tcfg.get("patience", 7), tcfg.get("min_delta", 1e-4)
    better = (lambda a, b: a > b + min_delta) if mode == "max" else (lambda a, b: a < b - min_delta)

    # ---------------- resume (jika sesi terputus di tengah run)
    start_epoch, history, best_val, best_epoch, bad_epochs, best_state = 0, [], None, -1, 0, None
    last_ckpt = out / "last.pt"
    if last_ckpt.exists() and not overwrite:
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, history = ck["epoch"] + 1, ck["history"]
        best_val, best_epoch, bad_epochs, best_state = ck["best_val"], ck["best_epoch"], ck["bad_epochs"], ck["best_state"]
        logger.info(f"[RESUME] {run_id} dari epoch {start_epoch}")

    logger.info(f"[RUN] {run_id} | n_train={len(ds_tr)} (per kelas {np.bincount(ds_tr.labels, minlength=len(class_names)).tolist()}) "
                f"| img={img_size} | device={device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    freeze_epochs = tcfg.get("freeze_epochs", 0)

    for epoch in range(start_epoch, epochs):
        set_backbone_trainable(model, epoch >= freeze_epochs)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0, tot_loss, tot_correct, tot_n = time.time(), 0.0, 0, 0
        for x, y, _ in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if tcfg.get("channels_last") and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tot_loss += loss.item() * len(y)
            tot_correct += (logits.argmax(1) == y).sum().item()
            tot_n += len(y)

        pv, yv, _, vloss = predict(model, dl_va, device, amp_dtype, eval_criterion)
        vm = compute_metrics(yv, pv, class_names, key_prefixes)
        vm["loss"] = vloss
        rec = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": tot_loss / tot_n,
               "train_acc": tot_correct / tot_n, "val_loss": vloss, "val_accuracy": vm["accuracy"],
               "val_macro_f1": vm["macro_f1"], "val_balanced_accuracy": vm["balanced_accuracy"],
               "val_macro_auc": vm["macro_auc"], "val_tb_recall": vm.get("tb_recall"),
               "val_tb_f1": vm.get("tb_f1"), "epoch_time_sec": time.time() - t0,
               "frozen_backbone": epoch < freeze_epochs}
        history.append(rec)
        cur = vm[monitor]
        if best_val is None or better(cur, best_val):
            best_val, best_epoch, bad_epochs = cur, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        logger.info(f"  ep {epoch:02d} | tr_loss {rec['train_loss']:.4f} | va_loss {vloss:.4f} | "
                    f"va_mF1 {vm['macro_f1']:.4f} | va_TBrec {vm.get('tb_recall', float('nan')):.4f} | "
                    f"best {monitor}={best_val:.4f}@{best_epoch} | {rec['epoch_time_sec']:.0f}s")
        if tcfg.get("save_last", True):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                        "history": history, "best_val": best_val, "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs, "best_state": best_state}, last_ckpt)
        if bad_epochs >= patience and epoch + 1 >= tcfg.get("min_epochs", 0):
            logger.info(f"  early stopping pada epoch {epoch}")
            break

    train_time = time.time() - t_start
    model.load_state_dict(best_state)

    # ---------------- evaluasi akhir dengan bobot terbaik
    pv, yv, iv, vloss = predict(model, dl_va, device, amp_dtype, eval_criterion)
    t_inf = time.time()
    pt, yt, it, tloss = predict(model, dl_te, device, amp_dtype, eval_criterion)
    inf_time = time.time() - t_inf
    val_m = compute_metrics(yv, pv, class_names, key_prefixes)
    val_m["loss"] = vloss
    test_m = compute_metrics(yt, pt, class_names, key_prefixes)
    test_m["loss"] = tloss
    ci = bootstrap_ci(yt, pt, class_names, cfg["evaluation"]["bootstrap_metrics"],
                      cfg["evaluation"].get("n_bootstrap", 1000), seed=seed, key_prefixes=key_prefixes)

    predictions_frame(val_df, pv, yv, iv, class_names).to_csv(out / "val_predictions.csv", index=False)
    predictions_frame(test_df, pt, yt, it, class_names).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    import timm
    meta = {
        "run_id": run_id, "design": design, "ratio": ratio, "strategy": strategy, "strategy_cfg": scfg,
        "model": model_key, "timm_name": mcfg["timm_name"], "seed": seed, "img_size": img_size,
        "n_train": len(ds_tr), "n_train_per_class": np.bincount(ds_tr.labels, minlength=len(class_names)).tolist(),
        "n_train_synthetic": n_syn, "n_train_classic_copies": n_classic,
        "method": scfg.get("extra", "gan" if scfg.get("synthetic_csv") else "none"), "dose": scfg.get("dose"),
        "n_val": len(ds_va), "n_test": len(ds_te),
        "epochs_run": len(history), "best_epoch": best_epoch, f"best_val_{monitor}": best_val,
        "train_time_sec": round(train_time, 1),
        "mean_epoch_time_sec": round(float(np.mean([h["epoch_time_sec"] for h in history])), 2),
        "inference_img_per_sec": round(len(ds_te) / inf_time, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__, "timm": timm.__version__, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        **count_params(model),
        **{k: v for k, v in git_info().items() if k != "git_dirty_files"},
    }
    save_json({"meta": meta, "val": val_m, "test": test_m, "test_ci95": ci}, out / "metrics.json")

    if tcfg.get("keep_checkpoint", True):
        torch.save({"model": best_state, "timm_name": mcfg["timm_name"], "img_size": img_size,
                    "mean": mean, "std": std, "class_names": class_names, "meta": meta}, out / "best.pt")
    if last_ckpt.exists():
        last_ckpt.unlink()
    logger.info(f"[DONE] {run_id} | test mF1={test_m['macro_f1']:.4f} | TB rec={test_m.get('tb_recall', float('nan')):.4f} "
                f"| AUC={test_m['macro_auc']:.4f} | {train_time/60:.1f} min")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"run_id": run_id, "status": "done", "test_macro_f1": test_m["macro_f1"]}
