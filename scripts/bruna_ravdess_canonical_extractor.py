"""Canonical extraction functions from executed notebook08; no extraction runs on import.
Only the task-file helper is stricter: a preserved model is required, never downloaded as latest.
"""
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

MEDIAPIPE_BACKEND="tasks"
mp_face_mesh=None
BaseOptions=mp_tasks_python.BaseOptions
FaceLandmarker=mp_tasks_vision.FaceLandmarker
FaceLandmarkerOptions=mp_tasks_vision.FaceLandmarkerOptions
VisionRunningMode=mp_tasks_vision.RunningMode
T_FRAMES=100
BASE_FEATURE_DIM=310
MIN_FACE_DETECTION_RATE=0.60
FACE_LANDMARKER_MODEL_PATH=None

def ensure_face_landmarker_model():
    if FACE_LANDMARKER_MODEL_PATH is None or not Path(FACE_LANDMARKER_MODEL_PATH).is_file():
        raise FileNotFoundError("Use the preserved CREMA-D face_landmarker.task; no latest model substitution.")
    return Path(FACE_LANDMARKER_MODEL_PATH)

SELECTED_LANDMARK_IDS=[0, 1, 2, 4, 5, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 45, 46, 48, 52, 53, 54, 55, 58, 61, 63, 64, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 98, 103, 105, 107, 109, 115, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 219, 220, 234, 235, 240, 246, 249, 251, 263, 267, 269, 270, 275, 276, 278, 282, 283, 284, 285, 288, 290, 291, 293, 295, 296, 297, 300, 305, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 332, 334, 336, 338, 344, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 439, 440, 454, 466]

def create_face_landmarker():
    # Cria o detector de landmarks compatível com a API disponível.
    if MEDIAPIPE_BACKEND == "solutions":
        return mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    model_path = ensure_face_landmarker_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)

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
