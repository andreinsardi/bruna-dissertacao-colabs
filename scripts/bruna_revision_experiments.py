"""Controlled supplementary experiments for Bruna's final dissertation revision.

Nothing in this module downloads data, modifies canonical artifacts or runs on import.
CPU-only helpers are independently testable; training/IG import PyTorch lazily.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

CANONICAL_RUN = "20260809_223452_c0613d3c"
PROTOCOL_VERSION = "bruna-controlled-revision-20261107-v1"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
VARIANTS = ("EmoT_full", "TemporalCNN1D_full", "EmoT_position")
HYPERPARAMETERS = {
    "epochs": 3000, "patience": 50, "batch_size": 256, "learning_rate_base": 1.0,
    "weight_decay": 3e-5, "label_smoothing": 0.10, "warmup_fraction": 0.10,
    "gradient_clip_value": 0.5, "gradient_clip_norm": 1.0,
    "d_model": 192, "n_heads": 3, "n_layers": 6, "dim_ff": 768,
    "dropout": 0.30, "mlp_head_size": 256, "max_len": 100,
}


def hash_file(path, block_size=2**20):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def require_compatible(existing, config):
    """Reject any existing job belonging to a different protocol/configuration."""
    expected = fingerprint(config)
    if existing.get("fingerprint") != expected:
        raise ValueError("Existing result/checkpoint has a different fingerprint; use a new output directory.")
    return expected


def lock_config(directory, config):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        require_compatible(json.loads(path.read_text(encoding="utf-8")), config)
    else:
        atomic_json(path, {"fingerprint": fingerprint(config), "config": config})
    return fingerprint(config)


def temporal_features(x, position_only=False):
    """Exact canonical finite differences; no physical-time derivative claim."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 100 or x.shape[2] != 310:
        raise ValueError(f"Expected [N,100,310], found {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite landmarks.")
    if position_only:
        return x.copy()
    out = np.zeros((len(x), 100, 1240), dtype=np.float32)
    out[:, :, :310] = x
    out[:, 1:, 310:620] = x[:, 1:] - x[:, :-1]
    out[:, 1:-1, 620:930] = x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]
    out[:, 2:-2, 930:] = x[:, 4:] - 2*x[:, 3:-1] + 2*x[:, 1:-3] - x[:, :-4]
    return out


def validate_alignment(manifest, x, y, file_stems, expected_n=7441):
    if len(manifest) != expected_n or x.shape != (expected_n, 100, 310):
        raise ValueError("The full canonical 7,441-video tensor is required.")
    required = {"ActorID", "file_stem", "y", "split", "Race"}
    if required.difference(manifest.columns):
        raise ValueError(f"Manifest missing {sorted(required.difference(manifest.columns))}")
    if not np.array_equal(manifest["file_stem"].astype(str).to_numpy(), np.asarray(file_stems).astype(str)):
        raise ValueError("Manifest/cache file_stems are not aligned.")
    if not np.array_equal(manifest["y"].to_numpy(dtype=int), np.asarray(y, dtype=int)):
        raise ValueError("Manifest/cache labels are not aligned.")
    if manifest["ActorID"].isna().any() or manifest["file_stem"].duplicated().any():
        raise ValueError("Missing actor or duplicate video.")
    if expected_n == 7441 and manifest.ActorID.nunique() != 91:
        raise ValueError("Expected the 91 canonical actors.")
    for col in ("Race","Sex","Ethnicity","Age","split"):
        if col in manifest and (manifest.groupby("ActorID")[col].nunique(dropna=False)>1).any():
            raise ValueError(f"Inconsistent {col} within an actor.")
    if np.any((np.asarray(y) < 0) | (np.asarray(y) > 5)):
        raise ValueError("Invalid six-class labels.")


def validate_fold(manifest, train_idx, val_idx, test_idx):
    """All videos and actors must occur in exactly one split of this fold."""
    arrays = [np.asarray(z) for z in (train_idx, val_idx, test_idx)]
    for z in arrays:
        if z.ndim != 1 or not np.issubdtype(z.dtype, np.integer) or not len(z):
            raise ValueError("Split indices must be nonempty integer vectors.")
        if len(np.unique(z)) != len(z) or z.min() < 0 or z.max() >= len(manifest):
            raise ValueError("Duplicate or invalid index.")
    merged = np.concatenate(arrays)
    if len(merged) != len(manifest) or not np.array_equal(np.sort(merged), np.arange(len(manifest))):
        raise ValueError("Split indices do not partition the entire manifest.")
    actors = [set(manifest.iloc[z]["ActorID"].tolist()) for z in arrays]
    if any(actors[i].intersection(actors[j]) for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Actor leakage between training, validation or test.")
    return {"train_actors": len(actors[0]), "validation_actors": len(actors[1]), "test_actors": len(actors[2])}


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not len(y_true):
        raise ValueError("Incompatible metric arrays.")
    if min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) > 5:
        raise ValueError("Invalid class index.")
    cm = np.bincount(6*y_true + y_pred, minlength=36).reshape(6, 6)
    true, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    f1 = np.divide(2*tp, true+predicted, out=np.zeros(6), where=(true+predicted)>0)
    recall = np.divide(tp, true, out=np.zeros(6), where=true>0)
    return {"accuracy": float(tp.sum()/len(y_true)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[true>0].mean()), "weighted_f1": float((f1*true).sum()/len(y_true))}


def select_sanity_samples(manifest, n_samples=64, seed=42):
    """Fixed holdout only; round-robin race then actor, sampling without replacement."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    candidates = manifest.loc[manifest["split"].eq("test")].copy()
    if not len(candidates):
        raise ValueError("No fixed-test samples.")
    rng = np.random.default_rng(seed)
    queues = {}
    for race, race_df in candidates.groupby("Race", dropna=False, sort=True):
        actor_ids = race_df["ActorID"].drop_duplicates().to_numpy(copy=True)
        rng.shuffle(actor_ids)
        queues[str(race)] = [list(rng.permutation(race_df.index[race_df["ActorID"].eq(a)].to_numpy()).astype(int)) for a in actor_ids]
    selected, cursors = [], {race: 0 for race in queues}
    while len(selected) < min(n_samples, len(candidates)):
        progress = False
        for race, actor_queues in queues.items():
            available = [j for j, q in enumerate(actor_queues) if q]
            if not available:
                continue
            start = cursors[race]
            chosen = next((j for j in range(start, len(actor_queues)) if actor_queues[j]), available[0])
            selected.append(actor_queues[chosen].pop())
            cursors[race] = (chosen+1) % len(actor_queues)
            progress = True
            if len(selected) == min(n_samples, len(candidates)):
                break
        if not progress:
            raise RuntimeError("Could not complete sanity sample selection.")
    return np.asarray(selected, dtype=np.int64)


def _dependencies():
    import torch
    from bruna_revision_models import EmoTransformer, TemporalCNN1D
    return torch, EmoTransformer, TemporalCNN1D


def _seed(seed, torch):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path, torch, device="cpu"):
    # Canonical and revision checkpoints are trusted user-owned experiment files.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _atomic_torch(path, payload, torch):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+"."+uuid.uuid4().hex+".partial")
    try:
        torch.save(payload, tmp); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _environment(torch):
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cudnn": torch.backends.cudnn.version(), "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _runtime_signature(torch):
    env = _environment(torch)
    return {k: env[k] for k in ("python", "numpy", "pandas", "torch", "cuda", "device", "cudnn", "cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms")}


def _load_sources(base_dir):
    base = Path(base_dir)
    run = base/"runs"/CANONICAL_RUN/"outputs"
    cache = base/"cache/cremad_landmarks_T100_semantic.npz"
    manifest_path = run/"tables/manifest_valid_with_splits.csv"
    with np.load(cache, allow_pickle=False) as z:
        x = z["X_base"].astype(np.float32)
        y = z["y"].astype(np.int64)
        stems = z["file_stems"].astype(str)
        landmarks = z["selected_landmark_ids"].astype(int)
    manifest = pd.read_csv(manifest_path)
    validate_alignment(manifest, x, y, stems)
    if len(landmarks) != 155 or len(np.unique(landmarks)) != 155:
        raise ValueError("Expected 155 unique canonical landmarks.")
    print("Hashing canonical inputs once for this execution; no source files are modified.", flush=True)
    identity = {"cache_sha256": hash_file(cache), "manifest_sha256": hash_file(manifest_path),
                "model_source_sha256": hash_file(Path(__file__).with_name("bruna_revision_models.py")),
                "experiment_source_sha256": hash_file(__file__)}
    return run, manifest, x, y, stems, landmarks, identity


def _model(variant, dimensions, EmoTransformer, TemporalCNN1D):
    if variant == "TemporalCNN1D_full":
        return TemporalCNN1D(T=100, D=dimensions, n_classes=6)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant}.")
    return EmoTransformer(D=dimensions, n_classes=6)


def _predict(model, x, torch, device, batch_size=256):
    model.eval(); logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start+batch_size]).to(device)
            logits.append(model(xb).detach().cpu().numpy())
    logits = np.concatenate(logits).astype(np.float32)
    if not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite predictions.")
    exp = np.exp(logits-logits.max(1, keepdims=True)); prob = exp/exp.sum(1, keepdims=True)
    return prob.argmax(1), prob.astype(np.float32), logits


def _rng_state(torch):
    return {"python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(payload, torch):
    random.setstate(payload["python_random_state"]); np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_random_state_all"]])


def _fit_job(directory, config, x_train, y_train, x_val, y_val, mu, sigma, torch, EmoTransformer, TemporalCNN1D):
    directory = Path(directory); fp = lock_config(directory, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed(config["seed"], torch)
    model = _model(config["variant"], config["dimensions"], EmoTransformer, TemporalCNN1D).to(device)
    counts = np.bincount(y_train, minlength=6).astype(np.float32)
    weights = counts.sum()/np.maximum(counts, 1); weights /= weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device), label_smoothing=.10)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=device=="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=3e-5)
    warmup = max(1, int(.10*len(loader)*3000))
    def schedule(step):
        step = max(1, step)
        return 192**-.5 * min(step**-.5, step*warmup**-1.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    latest, best = directory/"latest.pt", directory/"best.pt"
    start_epoch, best_f1, best_epoch, patience, history = 1, -1.0, -1, 50, []
    if latest.exists():
        state = _torch_load(latest, torch)
        require_compatible(state, config)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch, best_f1 = state["epoch"]+1, state["best_f1"]
        best_epoch, patience, history = state["best_epoch"], state["patience_left"], state["history"]
        _restore_rng(state, torch)
        if state["completed"]:
            if not best.exists():
                raise RuntimeError("Completed checkpoint has no best checkpoint.")
            chosen = _torch_load(best, torch); require_compatible(chosen, config)
            model.load_state_dict(chosen["model_state_dict"], strict=True)
            return model, history, best_epoch, best_f1
    for epoch in range(start_epoch, 3001):
        model.train(); loss_sum, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), .5)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = criterion(model(xb), yb); loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), .5)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            scheduler.step(); loss_sum += float(loss.item())*len(yb); seen += len(yb)
        pred, _, _ = _predict(model, x_val, torch, device)
        val = metrics(y_val, pred)
        row = {"epoch": epoch, "train_loss": loss_sum/seen, "learning_rate": optimizer.param_groups[0]["lr"], **{f"validation_{k}": v for k,v in val.items()}}
        history.append(row)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, patience = val["macro_f1"], epoch, 50
            _atomic_torch(best, {"fingerprint": fp, "model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                               "mu": mu, "sigma": sigma, "class_names": CLASSES, "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "config": config}, torch)
        else:
            patience -= 1
        complete = patience <= 0 or epoch == 3000
        payload = {"fingerprint": fp, "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch, "patience_left": patience,
                   "completed": complete, "history": history, "model_state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                   "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                   "scaler_state_dict": scaler.state_dict() if scaler is not None else None, **_rng_state(torch)}
        _atomic_torch(latest, payload, torch)
        atomic_csv(directory/"history.csv", pd.DataFrame(history))
        if epoch == 1 or epoch % 10 == 0 or complete:
            print(f"{config['variant']} seed={config['seed']} fold={config['fold']} epoch={epoch} valF1={val['macro_f1']:.5f} patience={patience}", flush=True)
        if complete:
            break
    if not best.exists():
        raise RuntimeError("No best checkpoint was produced.")
    chosen = _torch_load(best, torch); require_compatible(chosen, config)
    model.load_state_dict(chosen["model_state_dict"], strict=True)
    return model, history, best_epoch, best_f1


def run_revision_training(base_dir, output_dir=None, seeds=(42, 2026, 2027), folds=(1,2,3,4,5)):
    """Run all 45 controlled fits; retain historical fits only as separate evidence."""
    torch, EmoTransformer, TemporalCNN1D = _dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the definitive 45-fit revision; select a Colab GPU.")
    seeds, folds = tuple(map(int,seeds)), tuple(map(int,folds))
    if not seeds or not folds or len(set(seeds)) != len(seeds) or len(set(folds)) != len(folds) or not set(folds).issubset(range(1,6)):
        raise ValueError("Seeds and fold IDs must be unique; folds must be between 1 and 5.")
    root = Path(output_dir or Path(base_dir)/"revision_final_20261107")/"training"
    run, manifest, x_base, y, stems, landmarks, identity = _load_sources(base_dir)
    protocol = {"version": PROTOCOL_VERSION, "canonical_run": CANONICAL_RUN, "seeds": list(seeds), "folds": list(folds),
                "variants": list(VARIANTS), "hyperparameters": HYPERPARAMETERS, "source": identity,
                "runtime": _runtime_signature(torch), "historical_fits_reused": False,
                "seed_policy": "reset RNG independently before each seed/fold/variant fit"}
    lock_config(root, protocol); atomic_json(root/"environment.json", _environment(torch))
    historical = root/"historical_reference"; historical.mkdir(parents=True, exist_ok=True)
    for name in ("cv_fold_metrics.csv", "cv_summary_mean_std.csv"):
        source, dest = run/"tables"/name, historical/name
        if dest.exists() and hash_file(dest) != hash_file(source):
            raise ValueError("Historical reference changed; do not overwrite it.")
        if not dest.exists():shutil.copyfile(source, dest)
    split_specs, test_coverage = {}, np.zeros(len(manifest), dtype=int)
    for fold in folds:
        split_path = run/"cross_validation"/f"fold_{fold:02d}"/"normalization_and_indices.npz"
        with np.load(split_path, allow_pickle=False) as z:
            tr, va, te = [z[k].astype(np.int64) for k in ("train_idx","val_idx","test_idx")]
            saved_mu, saved_sigma = z["mu"].astype(np.float32).reshape(-1), z["sigma"].astype(np.float32).reshape(-1)
        validate_fold(manifest, tr, va, te); test_coverage[te] += 1
        split_specs[fold] = (tr, va, te, saved_mu, saved_sigma, hash_file(split_path))
    if set(folds) == set(range(1,6)) and not np.all(test_coverage == 1):
        raise ValueError("The five external tests do not cover every video exactly once.")
    rows, prediction_frames = [], []
    for fold in folds:
        tr, va, te, saved_mu, saved_sigma, split_sha = split_specs[fold]
        for position_only in (False, True):
            dimensions = 310 if position_only else 1240
            x_train = temporal_features(x_base[tr], position_only)
            mu = x_train.reshape(-1, dimensions).mean(0).astype(np.float32)
            sigma = x_train.reshape(-1, dimensions).std(0).astype(np.float32); sigma[sigma==0] = 1
            expected_mu = saved_mu[:dimensions] if position_only else saved_mu
            expected_sigma = saved_sigma[:dimensions] if position_only else saved_sigma
            if not (np.allclose(mu, expected_mu, rtol=1e-5, atol=1e-6) and np.allclose(sigma, expected_sigma, rtol=1e-5, atol=1e-6)):
                raise ValueError(f"Recomputed training-only normalization differs from canonical fold {fold}.")
            x_train -= mu; x_train /= sigma
            x_val, x_test = temporal_features(x_base[va], position_only), temporal_features(x_base[te], position_only)
            x_val -= mu; x_val /= sigma; x_test -= mu; x_test /= sigma
            variants = ("EmoT_position",) if position_only else ("EmoT_full", "TemporalCNN1D_full")
            for seed in seeds:
                for variant in variants:
                    directory = root/f"seed_{seed}"/f"fold_{fold:02d}"/variant
                    config = {"version": PROTOCOL_VERSION, "variant": variant, "seed": seed, "fold": fold, "dimensions": dimensions,
                              "source": identity, "split_sha256": split_sha, "hyperparameters": HYPERPARAMETERS, "runtime": protocol["runtime"],
                              "normalization_sha256": fingerprint({"mu": array_digest(mu), "sigma": array_digest(sigma)})}
                    fp = lock_config(directory, config); result_path = directory/"result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text(encoding="utf-8")); require_compatible(result, config)
                        for name,digest in result["artifact_sha256"].items():
                            if hash_file(directory/name) != digest:
                                raise RuntimeError(f"Completed artifact failed hash validation: {directory/name}")
                        with np.load(directory/"predictions.npz", allow_pickle=False) as z:
                            if not np.array_equal(z["test_idx"], te) or not np.array_equal(z["y_true"], y[te]):
                                raise RuntimeError("Completed prediction/manifest mismatch.")
                        frame = pd.read_csv(directory/"predictions.csv"); row = result["metrics"]
                    else:
                        started = time.time()
                        model, history, best_epoch, best_f1 = _fit_job(directory, config, x_train, y[tr], x_val, y[va], mu, sigma, torch, EmoTransformer, TemporalCNN1D)
                        pred, prob, logits = _predict(model, x_test, torch, "cuda")
                        frame = pd.DataFrame({"sample_idx": te,"ActorID": manifest.iloc[te]["ActorID"].to_numpy(),"fold": fold,"seed": seed,"variant": variant,"y_true": y[te],"y_pred": pred})
                        for c,name in enumerate(CLASSES):
                            frame[f"prob_{name}"] = prob[:,c]
                        atomic_npz(directory/"predictions.npz", test_idx=te,y_true=y[te],y_pred=pred,y_prob=prob,logits=logits)
                        atomic_csv(directory/"predictions.csv", frame)
                        row = {"fold": fold,"seed": seed,"variant": variant,"dimensions": dimensions,"parameters": sum(p.numel() for p in model.parameters()),
                               "best_epoch": best_epoch,"best_validation_macro_f1": best_f1,"elapsed_seconds": time.time()-started,**metrics(y[te],pred)}
                        hashes = {name:hash_file(directory/name) for name in ("best.pt","latest.pt","predictions.npz","predictions.csv","history.csv")}
                        atomic_json(result_path,{"fingerprint":fp,"status":"completed","metrics":row,"artifact_sha256":hashes})
                        del model; gc.collect(); torch.cuda.empty_cache()
                    rows.append(row); prediction_frames.append(frame)
                    atomic_csv(root/"fold_metrics_progress.csv", pd.DataFrame(rows))
            del x_train,x_val,x_test; gc.collect(); torch.cuda.empty_cache()
    results = pd.DataFrame(rows)
    expected = len(seeds)*len(folds)*3
    if len(results) != expected or results.duplicated(["seed","fold","variant"]).any():
        raise RuntimeError("Supplementary training grid is incomplete or duplicated.")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_csv(root/"fold_metrics.csv",results); atomic_csv(root/"pooled_predictions.csv",predictions)
    seed_summaries = []
    for (seed,variant), group in predictions.groupby(["seed","variant"]):
        if set(folds) == set(range(1,6)) and (len(group)!=len(manifest) or group.sample_idx.nunique()!=len(manifest)):
            raise RuntimeError("Supplementary OOF coverage is incomplete.")
        seed_summaries.append({"seed":int(seed),"variant":variant,"n_videos":len(group),**metrics(group.y_true,group.y_pred)})
    seed_metrics = pd.DataFrame(seed_summaries)
    atomic_csv(root/"pooled_metrics_by_seed.csv",seed_metrics)
    by_seed = results.groupby(["seed","variant"])[['macro_f1','accuracy']].agg(['mean','std']).reset_index()
    by_seed.columns = ['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in by_seed.columns]
    atomic_csv(root/"fold_summary_by_seed.csv",by_seed)
    across_seeds = seed_metrics.groupby("variant")[['macro_f1','accuracy']].agg(['mean','std']).reset_index()
    across_seeds.columns = ['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in across_seeds.columns]
    atomic_csv(root/"seed_summary_pooled_metrics.csv",across_seeds)
    summary = results.groupby("variant")[['macro_f1','accuracy']].agg(['mean','std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    atomic_csv(root/"summary_descriptive.csv",summary)
    paired = results.pivot(index=["seed","fold"],columns="variant",values="macro_f1").reset_index()
    paired["delta_EmoT_minus_CNN"] = paired.EmoT_full-paired.TemporalCNN1D_full
    paired["delta_full_minus_position"] = paired.EmoT_full-paired.EmoT_position
    atomic_csv(root/"paired_fold_seed_differences.csv",paired)
    final_grid = seeds==(42,2026,2027) and set(folds)==set(range(1,6))
    completion = {"status":"completed" if final_grid else "partial_grid_only", "fits":len(results),"definitive_grid_complete":final_grid,
                  "protocol_fingerprint":fingerprint(protocol),"historical_fits_reused":False,
                  "limitations":["Fold training sets overlap; 15 seed/fold metrics are not independent observations.","Three seeds describe limited training variability; no significance claim is generated.","Architecture parameter count changes when input dimensionality changes.","CUDA operators can retain nondeterminism; environment flags are recorded."]}
    atomic_json(root/"completion.json",completion)
    return completion


def _integrated_gradients(model, x, target, torch, device, steps=16, batch_size=4):
    """Canonical right-Riemann 16-step IG on logits, preserving signed values."""
    model.eval(); output=[]
    alphas=torch.linspace(0.0,1.0,steps+1,device=device)[1:]
    for start in range(0,len(x),batch_size):
        xb=torch.tensor(x[start:start+batch_size],dtype=torch.float32,device=device)
        labels=torch.tensor(target[start:start+batch_size],dtype=torch.long,device=device)
        total=torch.zeros_like(xb)
        for alpha in alphas:
            point=(alpha*xb).detach().requires_grad_(True)
            logits=model(point); selected=logits[torch.arange(len(point),device=device),labels].sum()
            grad=torch.autograd.grad(selected,point,retain_graph=False,create_graph=False)[0]
            total += grad.detach()
        output.append((xb*total/steps).detach().cpu().numpy())
    return np.concatenate(output)


def _average_ranks(values):
    values=np.asarray(values); order=np.argsort(values,kind="mergesort"); sorted_values=values[order]
    starts=np.r_[0,np.flatnonzero(sorted_values[1:]!=sorted_values[:-1])+1]
    ends=np.r_[starts[1:],len(values)]; ranks=np.empty(len(values),dtype=float)
    ranks[order]=np.repeat((starts+ends-1)/2,ends-starts)
    return ranks


def compare_maps(original, changed, top_k):
    a,b=np.asarray(original,dtype=float).reshape(-1),np.asarray(changed,dtype=float).reshape(-1)
    if a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid attribution maps.")
    norms=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(np.dot(a,b)/norms) if norms else None
    ra,rb=_average_ranks(a),_average_ranks(b);ra-=ra.mean();rb-=rb.mean()
    rnorm=np.linalg.norm(ra)*np.linalg.norm(rb)
    spearman=float(np.dot(ra,rb)/rnorm) if rnorm else None
    k=min(int(top_k),len(a))
    if k<1:raise ValueError("top_k must be positive.")
    # Stable tie-breaking is deterministic and explicitly recorded.
    ia=set(np.argsort(-a,kind="stable")[:k]);ib=set(np.argsort(-b,kind="stable")[:k])
    return {"cosine":cosine,"spearman":spearman,"top_k":k,"jaccard_top_k":len(ia&ib)/len(ia|ib),
            "original_nonzero":int(np.count_nonzero(a)),"randomized_nonzero":int(np.count_nonzero(b))}


def _attribution_views(attrs):
    # Canonical plots aggregate absolute magnitudes, so primary sanity comparisons do too.
    maps=np.abs(attrs).reshape(len(attrs),100,4,155,2).sum(axis=(2,4))
    return {"landmark_time_absolute":maps,"landmark_absolute":maps.sum(1),"temporal_absolute":maps.sum(2)}


def run_parameter_sanity(base_dir, output_dir=None, n_samples=64, random_seeds=(11,29,47)):
    """Cascading parameter randomization; fixed original targets and holdout inputs."""
    random_seeds=tuple(map(int,random_seeds))
    if not random_seeds or len(set(random_seeds))!=len(random_seeds):
        raise ValueError("Randomization seeds must be nonempty and unique.")
    torch,EmoTransformer,_=_dependencies(); device="cuda" if torch.cuda.is_available() else "cpu"
    run,manifest,x_base,y,stems,landmarks,identity=_load_sources(base_dir)
    original_path=run/"models/best_EmoT.pt"; payload=_torch_load(original_path,torch)
    if list(payload["class_names"])!=CLASSES:raise ValueError("Checkpoint class order mismatch.")
    mu,sigma=[np.asarray(payload[k],dtype=np.float32).reshape(-1) for k in ("mu","sigma")]
    if mu.shape!=(1240,) or sigma.shape!=(1240,) or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma<=0).any():
        raise ValueError("Invalid canonical checkpoint normalization.")
    sample_idx=select_sanity_samples(manifest,n_samples,seed=42)
    x=temporal_features(x_base[sample_idx]);x-=mu;x/=sigma
    del x_base;gc.collect()
    model=EmoTransformer(D=1240,n_classes=6).to(device);model.load_state_dict(payload["model_state_dict"],strict=True)
    target,prob,logits=_predict(model,x,torch,device)
    stages=[("head",("head.",))]+[(f"head_blocks_{layer+1}_to_6",(f"blocks.{layer}.",)) for layer in range(5,-1,-1)]+[("all_parameters",("input_proj.","cls_token","pos_embed"))]
    config={"version":PROTOCOL_VERSION,"source":identity,"checkpoint_sha256":hash_file(original_path),"sample_idx":sample_idx.tolist(),
            "target_classes":target.tolist(),"steps":16,"random_seeds":list(map(int,random_seeds)),"stages":[s[0] for s in stages],"runtime":_runtime_signature(torch),
            "reference":"zero in standardized feature space","target":"original model predicted class, fixed for all stages",
            "quadrature":"16 right-Riemann points alpha=1/16,...,1","primary_maps":"absolute canonical magnitudes",
            "top_k_tie_policy":"stable original feature index","selection":"fixed test only; round robin race then actor; purposive balanced audit cohort"}
    root=Path(output_dir or Path(base_dir)/"revision_final_20261107")/"parameter_sanity"
    fp=lock_config(root,config);atomic_json(root/"environment.json",_environment(torch))
    selection=manifest.iloc[sample_idx][["file_stem","ActorID","Race","label_name"]].copy()
    selection.insert(0,"sample_idx",sample_idx);selection["original_target_class"]=target;selection["original_predicted_probability"]=prob[np.arange(len(target)),target]
    atomic_csv(root/"sample_manifest.csv",selection)
    baseline_path, baseline_meta = root/"baseline_attributions.npz", root/"baseline.json"
    if baseline_meta.exists():
        cached = json.loads(baseline_meta.read_text(encoding="utf-8"));require_compatible(cached,config)
        if hash_file(baseline_path)!=cached["sha256"]:raise RuntimeError("Baseline attribution hash mismatch.")
        with np.load(baseline_path,allow_pickle=False) as z:
            if not np.array_equal(z["sample_idx"],sample_idx) or not np.array_equal(z["target_classes"],target):
                raise ValueError("Baseline sample/target mismatch.")
            original_views=_attribution_views(z["signed_ig"])
    else:
        baseline=_integrated_gradients(model,x,target,torch,device)
        original_views=_attribution_views(baseline)
        atomic_npz(baseline_path,sample_idx=sample_idx,target_classes=target,signed_ig=baseline,
                   landmark_time_absolute=original_views["landmark_time_absolute"],selected_landmark_ids=landmarks)
        atomic_json(baseline_meta,{"fingerprint":fp,"sha256":hash_file(baseline_path)})
        del baseline;gc.collect()
    rows=[]
    for seed in random_seeds:
        _seed(int(seed),torch)
        fresh=EmoTransformer(D=1240,n_classes=6)
        randomized_state=copy.deepcopy(payload["model_state_dict"])
        for stage,prefixes in stages:
            stage_path=root/f"seed_{seed}"/f"{stage}.json"
            if stage_path.exists():
                cached=json.loads(stage_path.read_text(encoding="utf-8"));require_compatible(cached,config)
                if cached.get("random_seed")!=int(seed) or cached.get("stage")!=stage:
                    raise ValueError("Cached sanity result belongs to a different stage/seed.")
                maps_path=stage_path.with_suffix('.npz')
                if hash_file(maps_path)!=cached["maps_sha256"]:raise RuntimeError("Sanity artifact hash mismatch.")
                rows.extend(cached["rows"])
            # Even when a stage is reused, replay the cumulative parameter replacement.
            fresh_state=fresh.state_dict()
            replaced=[]
            for name,value in fresh_state.items():
                if any(name.startswith(prefix) for prefix in prefixes):
                    randomized_state[name]=value.clone();replaced.append(name)
            if not replaced:raise RuntimeError(f"No parameters matched stage {stage}.")
            if stage_path.exists():continue
            model.load_state_dict(randomized_state,strict=True)
            changed=_integrated_gradients(model,x,target,torch,device);views=_attribution_views(changed)
            new_rows=[]
            for map_name,original in original_views.items():
                k=1000 if map_name=="landmark_time_absolute" else 20
                for i,sample in enumerate(sample_idx):
                    new_rows.append({"random_seed":int(seed),"stage":stage,"sample_idx":int(sample),"ActorID":int(manifest.iloc[sample].ActorID),
                                     "Race":str(manifest.iloc[sample].Race),"original_target_class":int(target[i]),"map":map_name,
                                     **compare_maps(original[i],views[map_name][i],k)})
            maps_path=stage_path.with_suffix('.npz')
            atomic_npz(maps_path,sample_idx=sample_idx,target_classes=target,**views)
            atomic_json(stage_path,{"fingerprint":fp,"status":"completed","random_seed":int(seed),"stage":stage,"rows":new_rows,"maps_sha256":hash_file(maps_path),"parameters_newly_randomized":replaced})
            rows.extend(new_rows);del changed,views;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            print(f"IG parameter sanity: seed={seed} stage={stage} complete",flush=True)
        del fresh,randomized_state
    frame=pd.DataFrame(rows)
    expected=len(random_seeds)*len(stages)*len(sample_idx)*3
    if len(frame)!=expected or frame.duplicated(["random_seed","stage","sample_idx","map"]).any():
        raise RuntimeError("Parameter sanity grid incomplete or duplicated.")
    atomic_csv(root/"sample_comparisons.csv",frame)
    summary=frame.groupby(["stage","map"])[["cosine","spearman","jaccard_top_k"]].agg(["mean","std","min","max"]).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    atomic_csv(root/"summary.csv",summary)
    is_final=int(n_samples)==64 and tuple(random_seeds)==(11,29,47) and len(sample_idx)==64
    result={"status":"completed" if is_final else "partial_grid_only","definitive_grid_complete":is_final,"randomized_stages":len(stages),
            "random_initializations":len(random_seeds),"n_samples":len(sample_idx),"fingerprint":fp,
            "limitations":["Numerical implementation agreement with Captum and parameter dependence are different properties.",
                           "No automatic pass/fail threshold was selected after observing these comparisons.",
                           "Low map similarity is not a universal certificate of explanation validity.",
                           "The balanced fixed-holdout audit cohort does not estimate population fairness.",
                           "Primary maps aggregate absolute magnitudes; signed baseline IG is also persisted."]}
    atomic_json(root/"completion.json",result)
    return result
