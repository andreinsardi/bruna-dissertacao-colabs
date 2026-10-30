"""Generate the resumable Colab Pro version of the CREMA-D experiment.

The existing notebook 06 remains the scientific baseline.  This generator
copies its complete experiment and adds durable recovery mechanics around the
expensive stages without modifying the original notebook.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "notebooks" / "06_cremad_fairness_xai_emot_final_colab.ipynb"
OUTPUT = (
    ROOT
    / "notebooks"
    / "07_cremad_fairness_xai_emot_resumable_colab_pro.ipynb"
)
RESILIENCE_MODULE = ROOT / "scripts" / "colab_resilience.py"


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)
    if value and not value.endswith("\n"):
        cell["source"].append("\n")


def replace_or_fail(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Could not locate transformation block: {label}")
    return source.replace(old, new, 1)


def wrap_stage(stage_id: str, body: str, required_expr: str = "[]") -> str:
    """Execute a baseline cell in global scope while recording durable state."""

    compile(body, f"<cremad-stage-{stage_id}>", "exec")
    return f'''# Etapa durável: {stage_id}
_stage_required = list({required_expr})
with tracked_stage(
    LEDGER,
    RUN_ID,
    {stage_id!r},
    required_artifacts=_stage_required,
    metadata={{"resume_mode": RESUME_MODE, "forced": {stage_id!r} in FORCE_STAGES}},
    log_dir=LOG_DIR,
    progress_dir=PROGRESS_DIR,
    heartbeat_seconds=HEARTBEAT_SECONDS,
    resource_probe=current_gpu_stats,
):
    exec(compile({body!r}, "<cremad-stage-{stage_id}>", "exec"), globals())
print_stage_dashboard(LEDGER, RUN_ID)
'''


SETUP_OLD = '''RAW_DIR = BASE_DIR / "raw"
VIDEO_DIR = RAW_DIR / "videos_flv"
METADATA_DIR = RAW_DIR / "metadata"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CKPT_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"
XAI_DIR = OUTPUT_DIR / "xai"
FAIRNESS_DIR = OUTPUT_DIR / "fairness"
ZIP_DIR = OUTPUT_DIR / "zips"

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
for d in [RAW_DIR, VIDEO_DIR, METADATA_DIR, CACHE_DIR, OUTPUT_DIR, FIG_DIR, TABLE_DIR, CKPT_DIR, LOG_DIR, REPORT_DIR, XAI_DIR, FAIRNESS_DIR, ZIP_DIR]:
    d.mkdir(parents=True, exist_ok=True)
'''


SETUP_NEW = '''# Contrato de recuperação. Para começar do zero, altere apenas
# FORCE_NEW_RUN=True. FORCE_STAGES permite invalidar seletivamente uma etapa.
EXPERIMENT_NAME = "cremad_fairness_xai_emot"
RESUME_MODE = True
FORCE_NEW_RUN = False
FORCE_STAGES = set()
HEARTBEAT_SECONDS = 120
LANDMARK_SHARD_SIZE = 50
CHECKPOINT_EVERY_EPOCHS = 1
AUTO_DOWNLOAD_FINAL_ZIP = False

RAW_DIR = BASE_DIR / "raw"
VIDEO_DIR = RAW_DIR / "videos_flv"
METADATA_DIR = RAW_DIR / "metadata"
CACHE_DIR = BASE_DIR / "cache"
STATE_DIR = BASE_DIR / "state"
RUNS_DIR = BASE_DIR / "runs"
STATE_DB = STATE_DIR / "experiment_state.sqlite"
ACTIVE_RUN_JSON = STATE_DIR / "active_run.json"

for d in [RAW_DIR, VIDEO_DIR, METADATA_DIR, CACHE_DIR, STATE_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

recovery_config = {
    "seed": SEED,
    "full_run": FULL_RUN,
    "fast_dev_samples": FAST_DEV_SAMPLES,
    "t_frames": T_FRAMES,
    "landmark_set": LANDMARK_SET,
    "temporal_derivatives": ADD_TEMPORAL_DERIVATIVES,
    "emot_model_size": EMOT_MODEL_SIZE,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "d_model": D_MODEL,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "label_smoothing": LABEL_SMOOTHING,
    "max_xai_samples": MAX_XAI_SAMPLES,
    "max_ig_samples": MAX_IG_SAMPLES,
    "ig_steps": IG_STEPS,
    "deletion_fractions": DELETION_FRACTIONS,
}
CONFIG_HASH = hashlib.sha256(json_dumps(recovery_config).encode("utf-8")).hexdigest()
preferred_run_id = None
if RESUME_MODE and ACTIVE_RUN_JSON.exists():
    try:
        active_state = json.loads(ACTIVE_RUN_JSON.read_text(encoding="utf-8"))
        if active_state.get("config_hash") == CONFIG_HASH:
            preferred_run_id = active_state.get("run_id")
    except Exception as exc:
        print(f"Aviso: ponteiro de execução ignorado: {exc}")

LEDGER = SQLiteLedger(STATE_DB)
RUN_ID, RUN_WAS_RESUMED = LEDGER.start_or_resume_run(
    experiment_name=EXPERIMENT_NAME,
    preferred_run_id=preferred_run_id,
    config_hash=CONFIG_HASH,
    config=recovery_config,
    force_new=FORCE_NEW_RUN or not RESUME_MODE,
)

RUN_DIR = RUNS_DIR / RUN_ID
OUTPUT_DIR = RUN_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CKPT_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"
XAI_DIR = OUTPUT_DIR / "xai"
FAIRNESS_DIR = OUTPUT_DIR / "fairness"
ZIP_DIR = OUTPUT_DIR / "zips"
PROGRESS_DIR = OUTPUT_DIR / "progress_snapshots"

for d in [RUN_DIR, OUTPUT_DIR, FIG_DIR, TABLE_DIR, CKPT_DIR, LOG_DIR, REPORT_DIR, XAI_DIR, FAIRNESS_DIR, ZIP_DIR, PROGRESS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

atomic_write_json(
    ACTIVE_RUN_JSON,
    {"run_id": RUN_ID, "config_hash": CONFIG_HASH, "updated_at": utc_now()},
)
print(f"Run: {RUN_ID} | retomada={RUN_WAS_RESUMED} | SQLite={STATE_DB}")
'''


DOWNLOAD_SOURCE = r'''
def download_file(url, path, timeout=DOWNLOAD_TIMEOUT, retries=3):
    path = Path(path)
    if file_is_valid(path):
        return {"path": str(path), "status": "exists", "bytes": path.stat().st_size, "error": None}
    last_error = None
    for attempt in range(1, retries + 1):
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(tmp, path)
            return {"path": str(path), "status": "downloaded", "bytes": path.stat().st_size, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if tmp.exists():
                tmp.unlink()
            time.sleep(1.5 * attempt)
    return {"path": str(path), "status": "failed", "bytes": 0, "error": last_error}

records = meta_run[["video_url", "video_path", "file_stem"]].drop_duplicates().to_dict("records")
start = time.time()
results = []
with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
    future_to_record = {
        executor.submit(download_file, row["video_url"], row["video_path"]): row
        for row in records
    }
    for future in tqdm(as_completed(future_to_record), total=len(future_to_record), desc="Baixando vídeos"):
        row = future_to_record[future]
        result = future.result()
        results.append(result)
        LEDGER.upsert_item(
            RUN_ID,
            domain="download",
            item_key=row["file_stem"],
            status="completed" if result["status"] in {"exists", "downloaded"} else "failed",
            path=result["path"],
            byte_count=result["bytes"],
            error=result["error"],
            metadata={"url": row["video_url"], "download_status": result["status"]},
            compute_hash=False,
        )

download_report = pd.DataFrame(results).sort_values("path").reset_index(drop=True)
atomic_write_text(TABLE_DIR / "download_report.csv", download_report.to_csv(index=False))
print(f"Tempo de download: {(time.time() - start) / 60:.1f} min")
display(download_report["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n"))
failed = download_report[download_report["status"] == "failed"]
log_event(
    "download_videos",
    start_time=start,
    n_files=int(len(download_report)),
    n_downloaded=int((download_report["status"] == "downloaded").sum()),
    n_existing=int((download_report["status"] == "exists").sum()),
    n_failed=int(len(failed)),
)
if len(failed):
    display(failed.head(20))
    raise RuntimeError(f"{len(failed)} download(s) falharam; execute novamente para retomar apenas os ausentes.")
'''


LANDMARK_SOURCE = r'''
LANDMARK_SCOPE = "full" if FULL_RUN else f"dev{FAST_DEV_SAMPLES}"
LANDMARK_CACHE = CACHE_DIR / (
    f"cremad_landmarks_T{T_FRAMES}_{LANDMARK_SET}.npz"
    if FULL_RUN else f"cremad_landmarks_T{T_FRAMES}_{LANDMARK_SET}_{LANDMARK_SCOPE}.npz"
)
PROCESSING_REPORT_CSV = TABLE_DIR / f"processing_report_T{T_FRAMES}_{LANDMARK_SET}.csv"
PROCESSING_REPORT_CACHE = CACHE_DIR / f"processing_report_T{T_FRAMES}_{LANDMARK_SET}_{LANDMARK_SCOPE}.csv"
LEGACY_PROCESSING_REPORT = BASE_DIR / "outputs" / "tables" / f"processing_report_T{T_FRAMES}_{LANDMARK_SET}.csv"
LANDMARK_SHARD_DIR = CACHE_DIR / f"landmark_shards_T{T_FRAMES}_{LANDMARK_SET}_{LANDMARK_SCOPE}"
LANDMARK_SHARD_DIR.mkdir(parents=True, exist_ok=True)

def sample_video_frames(video_path, t_frames=T_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, t_frames).round().astype(int)
    return [frames[i] for i in idx]

def normalize_landmarks_xy(coords_xy):
    center = coords_xy.mean(axis=0, keepdims=True)
    centered = coords_xy - center
    span = coords_xy.max(axis=0) - coords_xy.min(axis=0)
    return centered / max(float(span.max()), 1e-6)

def extract_frame_landmarks(frame_bgr, face_landmarker):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if MEDIAPIPE_BACKEND == "solutions":
        result = face_landmarker.process(rgb)
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
    coords = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in SELECTED_LANDMARK_IDS], dtype=np.float32)
    return normalize_landmarks_xy(coords).reshape(-1).astype(np.float32)

def interpolate_missing_frames(seq):
    valid = ~np.isnan(seq).any(axis=1)
    if valid.sum() == 0:
        return None
    if valid.all():
        return seq.astype(np.float32)
    x = np.arange(seq.shape[0])
    out = seq.copy()
    for d in range(seq.shape[1]):
        out[~valid, d] = np.interp(x[~valid], x[valid], seq[:, d][valid])
    return out.astype(np.float32)

def process_one_video(row, face_landmarker):
    frames = sample_video_frames(row["video_path"], T_FRAMES)
    if not frames:
        return None, {"file_stem": row["file_stem"], "status": "no_frames", "n_frames_read": 0, "detected_frames": 0, "detection_rate": 0.0}
    seq = np.full((T_FRAMES, BASE_FEATURE_DIM), np.nan, dtype=np.float32)
    detected = 0
    for i, frame in enumerate(frames):
        landmarks = extract_frame_landmarks(frame, face_landmarker)
        if landmarks is not None:
            seq[i] = landmarks
            detected += 1
    rate = detected / T_FRAMES
    if rate < MIN_FACE_DETECTION_RATE:
        return None, {"file_stem": row["file_stem"], "status": "low_detection_rate", "n_frames_read": len(frames), "detected_frames": detected, "detection_rate": rate}
    seq = interpolate_missing_frames(seq)
    if seq is None:
        return None, {"file_stem": row["file_stem"], "status": "interpolation_failed", "n_frames_read": len(frames), "detected_frames": detected, "detection_rate": rate}
    return seq, {"file_stem": row["file_stem"], "status": "ok", "n_frames_read": len(frames), "detected_frames": detected, "detection_rate": rate}

def shard_path(start_index, end_index):
    return LANDMARK_SHARD_DIR / f"shard_{start_index:05d}_{end_index:05d}.npz"

def load_landmark_shard(path):
    data = np.load(path, allow_pickle=False)
    reports = json.loads(str(data["report_json"].item()))
    return data["X_base"], data["y"], data["file_stems"].astype(str), reports

landmark_start = time.time()
report_candidate = next(
    (path for path in [PROCESSING_REPORT_CACHE, PROCESSING_REPORT_CSV, LEGACY_PROCESSING_REPORT] if file_is_valid(path)),
    None,
)
if file_is_valid(LANDMARK_CACHE) and report_candidate is not None and "landmark_processing" not in FORCE_STAGES:
    cached = np.load(LANDMARK_CACHE, allow_pickle=False)
    X_base = cached["X_base"]
    y = cached["y"]
    valid_file_stems = cached["file_stems"].astype(str)
    processing_report = pd.read_csv(report_candidate)
    atomic_write_text(PROCESSING_REPORT_CSV, processing_report.to_csv(index=False))
    atomic_write_text(PROCESSING_REPORT_CACHE, processing_report.to_csv(index=False))
    landmark_status = "cache_loaded"
    print(f"Cache consolidado carregado: {LANDMARK_CACHE}")
else:
    rows = meta_run.to_dict("records")
    ranges = [(start, min(start + LANDMARK_SHARD_SIZE, len(rows))) for start in range(0, len(rows), LANDMARK_SHARD_SIZE)]
    missing_ranges = [(start, end) for start, end in ranges if not file_is_valid(shard_path(start, end)) or "landmark_processing" in FORCE_STAGES]
    print(f"Shards totais={len(ranges)} | já válidos={len(ranges)-len(missing_ranges)} | pendentes={len(missing_ranges)}")
    if missing_ranges:
        with create_face_landmarker() as face_landmarker:
            for start_index, end_index in tqdm(missing_ranges, desc="Shards de landmarks"):
                shard_X, shard_y, shard_stems, shard_reports = [], [], [], []
                for row in rows[start_index:end_index]:
                    seq, report = process_one_video(row, face_landmarker)
                    shard_reports.append(report)
                    if seq is not None:
                        shard_X.append(seq)
                        shard_y.append(int(row["y"]))
                        shard_stems.append(row["file_stem"])
                X_array = np.stack(shard_X).astype(np.float32) if shard_X else np.empty((0, T_FRAMES, BASE_FEATURE_DIM), dtype=np.float32)
                path = shard_path(start_index, end_index)
                atomic_npz_save(
                    path,
                    X_base=X_array,
                    y=np.asarray(shard_y, dtype=np.int64),
                    file_stems=np.asarray(shard_stems, dtype=str),
                    report_json=np.asarray(json.dumps(shard_reports, ensure_ascii=False)),
                )
                for report in shard_reports:
                    LEDGER.upsert_item(
                        RUN_ID,
                        domain="landmark",
                        item_key=report["file_stem"],
                        status="completed" if report["status"] == "ok" else report["status"],
                        path=path,
                        metadata=report,
                        compute_hash=False,
                    )
                print(f"Shard persistido: {path.name} | válidos={len(shard_X)}/{end_index-start_index}")

    all_X, all_y, all_stems, all_reports = [], [], [], []
    for start_index, end_index in ranges:
        path = shard_path(start_index, end_index)
        if not file_is_valid(path):
            raise RuntimeError(f"Shard ausente ou vazio: {path}")
        shard_X, shard_y, shard_stems, shard_reports = load_landmark_shard(path)
        if len(shard_X):
            all_X.append(shard_X)
            all_y.append(shard_y)
            all_stems.append(shard_stems)
        all_reports.extend(shard_reports)
    if not all_X:
        raise RuntimeError("Nenhum vídeo atingiu o limiar mínimo de detecção facial.")
    X_base = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.int64)
    valid_file_stems = np.concatenate(all_stems).astype(str)
    processing_report = pd.DataFrame(all_reports)
    if len(processing_report) != len(rows):
        raise RuntimeError(f"Consolidação incompleta: {len(processing_report)}/{len(rows)} relatórios")
    atomic_write_text(PROCESSING_REPORT_CSV, processing_report.to_csv(index=False))
    atomic_write_text(PROCESSING_REPORT_CACHE, processing_report.to_csv(index=False))
    atomic_npz_save(
        LANDMARK_CACHE,
        X_base=X_base,
        y=y,
        file_stems=valid_file_stems,
        selected_landmark_ids=np.asarray(SELECTED_LANDMARK_IDS, dtype=np.int64),
    )
    landmark_status = "shards_consolidated"
    print(f"Cache consolidado salvo: {LANDMARK_CACHE}")

print(f"X_base: {X_base.shape} | y: {y.shape}")
display(processing_report["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n"))
log_event(
    "landmark_processing",
    status=landmark_status,
    start_time=landmark_start,
    n_valid_videos=int(len(valid_file_stems)),
    n_total_videos=int(len(processing_report)),
    ok_rate=float((processing_report["status"] == "ok").mean()),
    mean_detection_rate=float(processing_report["detection_rate"].mean()),
)
'''


TRAINING_SOURCE = r'''
# Checkpoints concretos esperados incluem latest_EmoT.pt e best_EmoT.pt.
def clip_gradients(model):
    if GRAD_CLIP_VALUE is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP_VALUE)
    if GRAD_CLIP_NORM is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

def make_pedro_lr_scheduler(optimizer, total_steps):
    warmup_steps = max(1, int(WARMUP_PERC * total_steps))
    def lr_lambda(step):
        step = max(1, step)
        return (D_MODEL ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps

def train_one_epoch(model, loader, optimizer, criterion, scaler=None, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if USE_AMP:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            clip_gradients(model)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def predict_model(model, X_np, batch_size=BATCH_SIZE):
    model.eval()
    logits_all = []
    dummy = np.zeros(len(X_np), dtype=np.int64)
    for xb, _ in make_loader(X_np, dummy, batch_size=batch_size):
        logits_all.append(model(xb.to(DEVICE, non_blocking=True)).detach().cpu().numpy())
    logits_all = np.concatenate(logits_all)
    prob = softmax_np(logits_all)
    return prob.argmax(axis=1), prob, logits_all

def evaluate_on_arrays(model, X_np, y_np):
    pred, prob, logits = predict_model(model, X_np)
    return compute_basic_metrics(y_np, pred), pred, prob, logits

def load_torch_payload(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def fit_model(model_name, model, loader, val_X, val_y, criterion, epochs=EPOCHS):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler, warmup_steps = (None, None)
    if USE_PEDRO_LR_SCHEDULE:
        scheduler, warmup_steps = make_pedro_lr_scheduler(optimizer, max(1, len(loader) * epochs))
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    latest_path = CKPT_DIR / f"latest_{model_name}.pt"
    best_path = CKPT_DIR / f"best_{model_name}.pt"
    history_path = LOG_DIR / f"training_history_{model_name}.csv"
    best_f1, best_epoch, patience_left = -1.0, -1, PATIENCE
    history, start_epoch = [], 1
    start = time.time()

    force_model_training = bool({f"train_{model_name}", "training_all_models"}.intersection(FORCE_STAGES))
    if file_is_valid(latest_path) and not force_model_training:
        payload = load_torch_payload(latest_path, DEVICE)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        history = payload.get("history", [])
        atomic_write_text(history_path, pd.DataFrame(history).to_csv(index=False))
        recorded_epochs = {int(row["epoch"]) for row in LEDGER.training_history(RUN_ID, model_name)}
        missing_history = [row for row in history if int(row["epoch"]) not in recorded_epochs]
        LEDGER.record_training_history(RUN_ID, model_name, missing_history, checkpoint_path=latest_path)
        best_f1 = float(payload.get("best_f1", -1.0))
        best_epoch = int(payload.get("best_epoch", -1))
        patience_left = payload.get("patience_left", PATIENCE)
        start_epoch = int(payload.get("epoch", 0)) + 1
        if payload.get("python_random_state") is not None:
            random.setstate(payload["python_random_state"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.set_rng_state(payload["torch_random_state"])
        if torch.cuda.is_available() and payload.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_random_state_all"])
        print(f"[{model_name}] retomando na época {start_epoch}; última íntegra={start_epoch-1}")
        if payload.get("completed") and file_is_valid(best_path):
            best_payload = load_torch_payload(best_path, DEVICE)
            model.load_state_dict(best_payload["model_state_dict"])
            print(f"[{model_name}] treinamento já concluído; melhor época={best_epoch}")
            return model, pd.DataFrame(history), best_path

    completed = False
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler)
        val_metrics, _, _, _ = evaluate_on_arrays(model, val_X, val_y)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {"model": model_name, "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "lr": current_lr, **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        print(f"[{model_name}] epoch {epoch:04d}/{epochs} loss={train_loss:.4f} train_acc={train_acc:.3f} val_macro_f1={val_metrics['macro_f1']:.3f} lr={current_lr:.2e}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = float(val_metrics["macro_f1"]), epoch
            if PATIENCE is not None:
                patience_left = PATIENCE
            atomic_torch_save({
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "T": T, "D": D, "class_names": CLASS_NAMES, "mu": mu, "sigma": sigma,
            }, best_path)
        elif PATIENCE is not None:
            patience_left -= 1

        early_stop = PATIENCE is not None and patience_left <= 0
        completed = early_stop or epoch == epochs
        checkpoint_payload = {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "patience_left": patience_left,
            "completed": completed,
            "history": history,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if epoch % CHECKPOINT_EVERY_EPOCHS == 0 or completed:
            atomic_torch_save(checkpoint_payload, latest_path)
        atomic_write_text(history_path, pd.DataFrame(history).to_csv(index=False))
        LEDGER.record_training_epoch(RUN_ID, model_name, epoch, row, checkpoint_path=latest_path)
        if early_stop:
            print(f"[{model_name}] early stopping na época {epoch}.")
            break

    if not file_is_valid(best_path):
        raise RuntimeError(f"Checkpoint de melhor modelo não foi produzido: {best_path}")
    best_payload = load_torch_payload(best_path, DEVICE)
    model.load_state_dict(best_payload["model_state_dict"])
    LEDGER.register_artifact(RUN_ID, f"train_{model_name}", latest_path, kind="latest_checkpoint")
    LEDGER.register_artifact(RUN_ID, f"train_{model_name}", best_path, kind="best_checkpoint")
    print(f"Melhor {model_name}: epoch={best_epoch}, val_macro_f1={best_f1:.4f}, tempo={(time.time()-start)/60:.1f} min")
    log_event(f"train_{model_name}", start_time=start, best_epoch=int(best_epoch), best_val_macro_f1=float(best_f1), checkpoint=str(best_path), latest_checkpoint=str(latest_path), params=int(count_parameters(model)), warmup_steps=int(warmup_steps) if warmup_steps is not None else None)
    return model, pd.DataFrame(history), best_path
'''


PACKAGE_SOURCE = r'''
import sqlite3
import zipfile

INCLUDE_OUTPUTS_IN_ZIP = True
INCLUDE_CACHE_IN_ZIP = True
INCLUDE_RAW_VIDEOS_IN_ZIP = False

# Marca provisoriamente como concluída para que o snapshot registre o desfecho.
# Se o empacotamento falhar, tracked_stage reverte o run para "failed".
LEDGER.mark_run(RUN_ID, "completed")
state_snapshot = REPORT_DIR / "experiment_state_snapshot.sqlite"
state_snapshot_tmp = state_snapshot.with_name(state_snapshot.name + f".tmp-{os.getpid()}")
with contextlib.closing(sqlite3.connect(STATE_DB)) as source_db, contextlib.closing(sqlite3.connect(state_snapshot_tmp)) as target_db:
    source_db.backup(target_db)
    target_db.commit()
os.replace(state_snapshot_tmp, state_snapshot)

artifact_zip = ZIP_DIR / f"cremad_fairness_xai_artifacts_{RUN_ID}.zip"
manifest_path = ZIP_DIR / f"download_manifest_{RUN_ID}.csv"

def should_skip_artifact(path):
    path = Path(path)
    if path == LOG_DIR / "stage_logs" / "package_artifacts.log":
        return True
    try:
        path.resolve().relative_to(ZIP_DIR.resolve())
        return True
    except ValueError:
        return path.name.endswith(".tmp") or ".tmp-" in path.name

artifact_sources = []
if INCLUDE_OUTPUTS_IN_ZIP and OUTPUT_DIR.exists():
    artifact_sources.append(("outputs", OUTPUT_DIR))
if INCLUDE_CACHE_IN_ZIP and CACHE_DIR.exists():
    artifact_sources.append(("cache", CACHE_DIR))
if INCLUDE_RAW_VIDEOS_IN_ZIP and VIDEO_DIR.exists():
    artifact_sources.append(("raw/videos_flv", VIDEO_DIR))

manifest_rows = []
for source_name, source_dir in artifact_sources:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and not should_skip_artifact(path):
            manifest_rows.append({
                "source": source_name,
                "relative_path": str(Path(source_name) / path.relative_to(source_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "absolute_path": str(path),
            })

download_manifest = pd.DataFrame(manifest_rows)
atomic_write_text(manifest_path, download_manifest.to_csv(index=False))
atomic_write_text(TABLE_DIR / "download_manifest_latest.csv", download_manifest.to_csv(index=False))

zip_tmp = artifact_zip.with_name(artifact_zip.name + f".tmp-{os.getpid()}")
try:
    with zipfile.ZipFile(zip_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(manifest_path, arcname="outputs/download_manifest.csv")
        for row in manifest_rows:
            archive.write(row["absolute_path"], arcname=row["relative_path"])
    os.replace(zip_tmp, artifact_zip)
finally:
    if zip_tmp.exists():
        zip_tmp.unlink()

LEDGER.register_artifact(RUN_ID, "package_artifacts", manifest_path, kind="manifest")
LEDGER.register_artifact(RUN_ID, "package_artifacts", state_snapshot, kind="sqlite_snapshot")
LEDGER.register_artifact(RUN_ID, "package_artifacts", artifact_zip, kind="zip")
print(f"Arquivo criado: {artifact_zip}")
print(f"Tamanho: {artifact_zip.stat().st_size / (1024 ** 2):.1f} MB")
print(f"Arquivos incluídos: {len(download_manifest):,}")
display(download_manifest.sort_values("bytes", ascending=False).head(20))

if IN_COLAB and AUTO_DOWNLOAD_FINAL_ZIP:
    from google.colab import files
    files.download(str(artifact_zip))
else:
    print("Download automático desativado; o ZIP permanece persistido no Google Drive.")
'''


def build_notebook() -> dict:
    notebook = json.loads(BASELINE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    set_source(
        cells[0],
        "# CREMA-D + MediaPipe + EmoT/AcT — execução resumível no Colab Pro\n\n"
        "Pipeline integral de classificação, fairness e XAI, com Google Drive, "
        "SQLite, checkpoints por época, shards de landmarks, logs, hashes e "
        "prints de progresso. O notebook 06 permanece preservado como baseline.\n",
    )

    recovery_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            "## Como executar e recuperar\n\n"
            "1. Selecione **Ambiente de execução > Alterar tipo > GPU (A100)**.\n"
            "2. Use **Executar tudo**. Autorize a montagem do Google Drive.\n"
            "3. Após desconexão ou erro, abra o mesmo notebook e use **Executar tudo** novamente. "
            "O ledger `state/experiment_state.sqlite` escolhe a execução incompleta; downloads, "
            "shards, épocas e XAI já persistidos são reaproveitados.\n"
            "4. Para uma execução nova, defina `FORCE_NEW_RUN=True`. Para refazer só uma etapa, "
            "adicione seu identificador a `FORCE_STAGES`.\n\n"
            "Cada execução fica isolada em `cremad_fairness_xai/runs/<run_id>/`; `raw/`, `cache/` "
            "e `state/` são duráveis e compartilhados. Nenhuma célula escreve no Overleaf.\n"
        ).splitlines(keepends=True),
    }

    resilience_source = RESILIENCE_MODULE.read_text(encoding="utf-8")
    resilience_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cremad_role": "durable_runtime"},
        "outputs": [],
        "source": [
            "# Módulo operacional embutido: também é testado em scripts/colab_resilience.py\n",
            f"exec(compile({resilience_source!r}, 'colab_resilience.py', 'exec'), globals())\n",
            "print(f'Recovery runtime carregado | schema={SCHEMA_VERSION}')\n",
        ],
    }

    setup = cell_source(cells[5])
    setup = replace_or_fail(setup, SETUP_OLD, SETUP_NEW, "setup directories")
    setup = replace_or_fail(
        setup,
        '''with open(LOG_DIR / f"run_config_{RUN_ID}.json", "w", encoding="utf-8") as f:\n    json.dump(run_config, f, ensure_ascii=False, indent=2)\n''',
        '''run_config["config_hash"] = CONFIG_HASH\nrun_config["run_dir"] = str(RUN_DIR)\nrun_config["resumed"] = RUN_WAS_RESUMED\natomic_write_json(LOG_DIR / f"run_config_{RUN_ID}.json", run_config)\n''',
        "atomic run config",
    )
    setup = replace_or_fail(
        setup,
        '    row.update(extra)\n    RUN_EVENTS.append(row)\n',
        '''    row.update(extra)\n    RUN_EVENTS.append(row)\n    try:\n        LEDGER.event(RUN_ID, level="info", message=f"{stage}: {status}", stage_id=stage, payload=row)\n    except Exception as exc:\n        print(f"[WARN] evento SQLite não persistido: {exc}")\n''',
        "SQLite event bridge",
    )
    set_source(cells[5], setup)

    set_source(cells[17], wrap_stage("download_videos", DOWNLOAD_SOURCE, '[TABLE_DIR / "download_report.csv"]'))
    cells[17]["metadata"]["cremad_stage"] = "download_videos"

    set_source(
        cells[23],
        wrap_stage(
            "landmark_processing",
            LANDMARK_SOURCE,
            '[CACHE_DIR / f"cremad_landmarks_T{T_FRAMES}_{LANDMARK_SET}.npz", TABLE_DIR / f"processing_report_T{T_FRAMES}_{LANDMARK_SET}.csv"]',
        ),
    )
    cells[23]["metadata"]["cremad_stage"] = "landmark_processing"

    set_source(cells[39], TRAINING_SOURCE)
    training_body = cell_source(cells[41])
    training_body = training_body.replace(
        'history_df.to_csv(TABLE_DIR / "training_history.csv", index=False)\nhistory_df.to_csv(LOG_DIR / f"training_history_{RUN_ID}.csv", index=False)',
        'atomic_write_text(TABLE_DIR / "training_history.csv", history_df.to_csv(index=False))\natomic_write_text(LOG_DIR / f"training_history_{RUN_ID}.csv", history_df.to_csv(index=False))',
    )
    training_required = '[CKPT_DIR / "best_FlatMLP.pt", CKPT_DIR / "best_TemporalCNN1D.pt", CKPT_DIR / "best_EmoT.pt", CKPT_DIR / "best_EmoT_Reweighted.pt"]'
    set_source(cells[41], wrap_stage("training_all_models", training_body, training_required))
    cells[41]["metadata"]["cremad_stage"] = "training_all_models"

    attention = cell_source(cells[53])
    attention = replace_or_fail(
        attention,
        '''xai_attention_start = time.time()\nattention_curves, xai_preds, xai_probs = compute_attention_importance(emot_model, X_xai)\nprint(f"Amostras XAI: {len(xai_idx)}")\nprint(f"attention_curves: {attention_curves.shape}")\nmeta_xai.to_csv(XAI_DIR / "xai_sample_manifest.csv", index=False)\nnp.savez_compressed(\n    XAI_DIR / "xai_attention_rollout_emot.npz",\n    attention_curves=attention_curves,\n    xai_preds=xai_preds,\n    xai_probs=xai_probs,\n    xai_idx=xai_idx,\n    y_xai=y_xai,\n)\n''',
        '''xai_attention_start = time.time()\nATTENTION_CACHE = XAI_DIR / "xai_attention_rollout_emot.npz"\nif file_is_valid(ATTENTION_CACHE) and "xai_attention_rollout" not in FORCE_STAGES:\n    attention_data = np.load(ATTENTION_CACHE, allow_pickle=False)\n    attention_curves = attention_data["attention_curves"]\n    xai_preds = attention_data["xai_preds"]\n    xai_probs = attention_data["xai_probs"]\n    print(f"Attention Rollout retomado do cache: {ATTENTION_CACHE}")\nelse:\n    attention_curves, xai_preds, xai_probs = compute_attention_importance(emot_model, X_xai)\n    atomic_npz_save(ATTENTION_CACHE, attention_curves=attention_curves, xai_preds=xai_preds, xai_probs=xai_probs, xai_idx=xai_idx, y_xai=y_xai)\nprint(f"Amostras XAI: {len(xai_idx)}")\nprint(f"attention_curves: {attention_curves.shape}")\natomic_write_text(XAI_DIR / "xai_sample_manifest.csv", meta_xai.to_csv(index=False))\n''',
        "attention cache",
    )
    set_source(cells[53], wrap_stage("xai_attention_rollout", attention, '[XAI_DIR / "xai_attention_rollout_emot.npz"]'))
    cells[53]["metadata"]["cremad_stage"] = "xai_attention_rollout"

    gradient = cell_source(cells[55])
    gradient = replace_or_fail(
        gradient,
        '''grad_x_input_start = time.time()\ngrad_attr = compute_grad_x_input(emot_model, X_xai, xai_preds, batch_size=16)\nlandmark_scores = aggregate_attr_to_landmarks(grad_attr)\nnp.savez_compressed(\n    XAI_DIR / "xai_gradient_landmark_scores_emot.npz",\n    landmark_scores=landmark_scores,\n    selected_landmark_ids=np.array(SELECTED_LANDMARK_IDS, dtype=np.int64),\n)\n''',
        '''grad_x_input_start = time.time()\nGRADIENT_CACHE = XAI_DIR / "xai_gradient_landmark_scores_emot.npz"\nif file_is_valid(GRADIENT_CACHE) and "xai_gradient_x_input" not in FORCE_STAGES:\n    gradient_data = np.load(GRADIENT_CACHE, allow_pickle=False)\n    grad_attr = gradient_data["grad_attr"]\n    landmark_scores = gradient_data["landmark_scores"]\n    print(f"Gradient x Input retomado do cache: {GRADIENT_CACHE}")\nelse:\n    grad_attr = compute_grad_x_input(emot_model, X_xai, xai_preds, batch_size=16)\n    landmark_scores = aggregate_attr_to_landmarks(grad_attr)\n    atomic_npz_save(GRADIENT_CACHE, grad_attr=grad_attr, landmark_scores=landmark_scores, selected_landmark_ids=np.asarray(SELECTED_LANDMARK_IDS, dtype=np.int64))\n''',
        "gradient cache",
    )
    set_source(cells[55], wrap_stage("xai_gradient_x_input", gradient, '[XAI_DIR / "xai_gradient_landmark_scores_emot.npz"]'))
    cells[55]["metadata"]["cremad_stage"] = "xai_gradient_x_input"

    integrated = cell_source(cells[57])
    integrated = replace_or_fail(
        integrated,
        '''ig_attr = compute_integrated_gradients(emot_model, X_ig, xai_preds_ig, steps=IG_STEPS, batch_size=16)\nig_landmark_scores = aggregate_attr_to_landmarks(ig_attr)\n''',
        '''IG_CACHE = XAI_DIR / "xai_integrated_gradients_emot.npz"\nif file_is_valid(IG_CACHE) and "xai_integrated_gradients" not in FORCE_STAGES:\n    ig_data = np.load(IG_CACHE, allow_pickle=False)\n    ig_attr = ig_data["ig_attr"]\n    ig_landmark_scores = ig_data["ig_landmark_scores"]\n    ig_indices = ig_data["ig_indices"].astype(int)\n    X_ig, y_ig = X_xai[ig_indices], y_xai[ig_indices]\n    meta_ig = meta_xai.iloc[ig_indices].reset_index(drop=True)\n    xai_preds_ig = xai_preds[ig_indices]\n    print(f"Integrated Gradients retomado do cache: {IG_CACHE}")\nelse:\n    ig_attr = compute_integrated_gradients(emot_model, X_ig, xai_preds_ig, steps=IG_STEPS, batch_size=16)\n    ig_landmark_scores = aggregate_attr_to_landmarks(ig_attr)\n    atomic_npz_save(IG_CACHE, ig_attr=ig_attr, ig_landmark_scores=ig_landmark_scores, ig_indices=ig_indices, selected_landmark_ids=np.asarray(SELECTED_LANDMARK_IDS, dtype=np.int64))\n''',
        "integrated gradients cache",
    )
    integrated = integrated.replace(
        '''np.savez_compressed(\n    XAI_DIR / "xai_integrated_gradients_emot.npz",\n    ig_landmark_scores=ig_landmark_scores,\n    ig_indices=ig_indices,\n    selected_landmark_ids=np.array(SELECTED_LANDMARK_IDS, dtype=np.int64),\n)\n''',
        "",
    )
    set_source(cells[57], wrap_stage("xai_integrated_gradients", integrated, '[XAI_DIR / "xai_integrated_gradients_emot.npz"]'))
    cells[57]["metadata"]["cremad_stage"] = "xai_integrated_gradients"

    occlusion = cell_source(cells[63])
    occlusion = replace_or_fail(
        occlusion,
        'occlusion_df = region_occlusion_test(emot_model, X_xai, y_xai, meta_xai, xai_preds)\n',
        '''OCCLUSION_CACHE = XAI_DIR / "xai_region_occlusion_samples.csv"\nif file_is_valid(OCCLUSION_CACHE) and "xai_region_occlusion" not in FORCE_STAGES:\n    occlusion_df = pd.read_csv(OCCLUSION_CACHE)\n    print(f"Oclusão retomada do cache: {OCCLUSION_CACHE}")\nelse:\n    occlusion_df = region_occlusion_test(emot_model, X_xai, y_xai, meta_xai, xai_preds)\n''',
        "occlusion cache",
    )
    occlusion = replace_or_fail(
        occlusion,
        'occlusion_df = occlusion_df.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")\n',
        '''if "Race" not in occlusion_df.columns:\n    occlusion_df = occlusion_df.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")\n''',
        "occlusion resume merge",
    )
    for old_line, new_line in {
        'occlusion_df.to_csv(TABLE_DIR / "xai_region_occlusion_samples.csv", index=False)': 'atomic_write_text(TABLE_DIR / "xai_region_occlusion_samples.csv", occlusion_df.to_csv(index=False))',
        'occlusion_summary.to_csv(TABLE_DIR / "xai_region_occlusion_by_race.csv", index=False)': 'atomic_write_text(TABLE_DIR / "xai_region_occlusion_by_race.csv", occlusion_summary.to_csv(index=False))',
        'occlusion_df.to_csv(XAI_DIR / "xai_region_occlusion_samples.csv", index=False)': 'atomic_write_text(XAI_DIR / "xai_region_occlusion_samples.csv", occlusion_df.to_csv(index=False))',
        'occlusion_summary.to_csv(XAI_DIR / "xai_region_occlusion_by_race.csv", index=False)': 'atomic_write_text(XAI_DIR / "xai_region_occlusion_by_race.csv", occlusion_summary.to_csv(index=False))',
    }.items():
        occlusion = replace_or_fail(occlusion, old_line, new_line, "atomic occlusion CSV")
    set_source(cells[63], wrap_stage("xai_region_occlusion", occlusion, '[XAI_DIR / "xai_region_occlusion_samples.csv"]'))
    cells[63]["metadata"]["cremad_stage"] = "xai_region_occlusion"

    deletion = cell_source(cells[65])
    deletion = replace_or_fail(
        deletion,
        'deletion_df = deletion_test_temporal(emot_model, X_xai, attention_curves, xai_preds)\n',
        '''DELETION_CACHE = XAI_DIR / "xai_deletion_test_temporal.csv"\nif file_is_valid(DELETION_CACHE) and "xai_deletion_test" not in FORCE_STAGES:\n    deletion_df = pd.read_csv(DELETION_CACHE)\n    print(f"Deletion Test retomado do cache: {DELETION_CACHE}")\nelse:\n    deletion_df = deletion_test_temporal(emot_model, X_xai, attention_curves, xai_preds)\n''',
        "deletion cache",
    )
    deletion = replace_or_fail(
        deletion,
        'deletion_df = deletion_df.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")\n',
        '''if "Race" not in deletion_df.columns:\n    deletion_df = deletion_df.merge(meta_xai[["Race", "Ethnicity", "Sex", "label_name"]].reset_index(names="sample_idx"), on="sample_idx", how="left")\n''',
        "deletion resume merge",
    )
    for old_line, new_line in {
        'deletion_df.to_csv(TABLE_DIR / "xai_deletion_test_temporal.csv", index=False)': 'atomic_write_text(TABLE_DIR / "xai_deletion_test_temporal.csv", deletion_df.to_csv(index=False))',
        'deletion_df.to_csv(XAI_DIR / "xai_deletion_test_temporal.csv", index=False)': 'atomic_write_text(XAI_DIR / "xai_deletion_test_temporal.csv", deletion_df.to_csv(index=False))',
        'deletion_auc_by_race.to_csv(TABLE_DIR / "xai_deletion_auc_by_race.csv", index=False)': 'atomic_write_text(TABLE_DIR / "xai_deletion_auc_by_race.csv", deletion_auc_by_race.to_csv(index=False))',
        'deletion_auc_by_race.to_csv(XAI_DIR / "xai_deletion_auc_by_race.csv", index=False)': 'atomic_write_text(XAI_DIR / "xai_deletion_auc_by_race.csv", deletion_auc_by_race.to_csv(index=False))',
    }.items():
        deletion = replace_or_fail(deletion, old_line, new_line, "atomic deletion CSV")
    set_source(cells[65], wrap_stage("xai_deletion_test", deletion, '[XAI_DIR / "xai_deletion_test_temporal.csv"]'))
    cells[65]["metadata"]["cremad_stage"] = "xai_deletion_test"

    set_source(cells[76], wrap_stage("package_artifacts", PACKAGE_SOURCE, '[ZIP_DIR / f"cremad_fairness_xai_artifacts_{RUN_ID}.zip"]'))
    cells[76]["metadata"]["cremad_stage"] = "package_artifacts"

    final_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cremad_role": "finalize_run"},
        "outputs": [],
        "source": (
            "LEDGER.mark_run(RUN_ID, 'completed')\n"
            "atomic_write_json(ACTIVE_RUN_JSON, {\n"
            "    'run_id': RUN_ID, 'config_hash': CONFIG_HASH, 'status': 'completed',\n"
            "    'updated_at': utc_now(),\n"
            "})\n"
            "atomic_write_json(STATE_DIR / f'completed_{RUN_ID}.json', {\n"
            "    'run_id': RUN_ID, 'status': 'completed', 'completed_at': utc_now(),\n"
            "    'run_dir': str(RUN_DIR), 'artifact_zip': str(artifact_zip),\n"
            "})\n"
            "save_progress_snapshot(LEDGER, RUN_ID, PROGRESS_DIR, 'run', 'completed')\n"
            "print_stage_dashboard(LEDGER, RUN_ID)\n"
            "print(f'EXECUÇÃO CONCLUÍDA E PERSISTIDA: {RUN_ID}')\n"
            "print(f'Artefatos: {RUN_DIR}')\n"
        ).splitlines(keepends=True),
    }

    # The runtime must be available before the transformed setup cell.
    notebook["cells"] = cells[:5] + [recovery_markdown, resilience_cell] + cells[5:] + [final_cell]
    notebook.setdefault("metadata", {})["accelerator"] = "GPU"
    notebook["metadata"].setdefault("colab", {})["gpuType"] = "A100"
    notebook["metadata"]["colab"]["provenance"] = ["generated-from-notebook-06"]
    notebook["metadata"]["cremad_recovery"] = {
        "schema": 1,
        "state_database": "cremad_fairness_xai/state/experiment_state.sqlite",
        "run_layout": "cremad_fairness_xai/runs/<run_id>",
        "overleaf_write": False,
    }
    return notebook


def main() -> None:
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    if not RESILIENCE_MODULE.is_file():
        raise FileNotFoundError(RESILIENCE_MODULE)
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Notebook generated: {OUTPUT}")


if __name__ == "__main__":
    main()
