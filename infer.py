# coding: utf-8
"""
infer.py — Singleton model loader and frame-level inference utility.

Usage:
    import infer
    infer.get_model()          # optional: pre-warm at startup
    result = infer.predict_frame(pil_or_numpy_image)
    # result["binary"]   -> JPEG bytes (grayscale binary segmentation)
    # result["instance"] -> JPEG bytes (RGB instance segmentation)
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.lanenet.LaneNet import LaneNet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "log" / "best_model.pth"

RESIZE_WIDTH = 512
RESIZE_HEIGHT = 256

# ---------------------------------------------------------------------------
# Pre-processing transform (ImageNet stats, same as test.py)
# ---------------------------------------------------------------------------

_transform = transforms.Compose(
    [
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ---------------------------------------------------------------------------
# Thread-safe lazy singleton
# ---------------------------------------------------------------------------

_model: LaneNet | None = None
_model_lock = threading.Lock()


def get_model() -> LaneNet:
    """Return the cached LaneNet model, loading it on the first call.

    Thread-safe: concurrent callers block until the model is ready.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-checked locking — another thread may have loaded it while we
        # were waiting for the lock.
        if _model is not None:
            return _model

        print(f"[infer] Loading model from {MODEL_PATH} onto {DEVICE} …")
        model = LaneNet(arch="ENet")
        state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)
        _model = model
        print("[infer] Model loaded successfully.")

    return _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_frame(image) -> dict[str, bytes]:
    """Run lane detection on a single frame and return JPEG bytes.

    Parameters
    ----------
    image : PIL.Image.Image | numpy.ndarray
        Either a PIL image (any mode) **or** a NumPy array.
        NumPy arrays are assumed to be BGR (OpenCV convention) unless they
        have a ``color_space`` attribute set to ``"RGB"``.  Pass a plain RGB
        array as ``Image.fromarray(arr)`` if you want to be explicit.

    Returns
    -------
    dict with keys:
        "binary"   -> JPEG bytes of the binary segmentation mask (grayscale)
        "instance" -> JPEG bytes of the instance segmentation map (colour)
    """
    # ------------------------------------------------------------------
    # 1. Normalise input to an RGB PIL image
    # ------------------------------------------------------------------
    if isinstance(image, np.ndarray):
        # Assume BGR (OpenCV default); convert to RGB.
        rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_array)
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError(
            f"predict_frame expects a PIL Image or NumPy array, got {type(image)}"
        )

    # ------------------------------------------------------------------
    # 2. Apply transform and add batch dimension
    # ------------------------------------------------------------------
    tensor = _transform(pil_image)  # (3, H, W)
    tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    # ------------------------------------------------------------------
    # 3. Forward pass (no gradient computation needed)
    # ------------------------------------------------------------------
    model = get_model()
    with torch.inference_mode():
        outputs = model(tensor)

    # ------------------------------------------------------------------
    # 4. Post-process outputs → uint8 NumPy arrays
    # ------------------------------------------------------------------
    # binary_seg_pred:      (1, 1, H, W)  long / bool
    binary_pred: np.ndarray = (
        outputs["binary_seg_pred"]
        .squeeze()  # (H, W)
        .cpu()
        .numpy()
        .astype(np.float32)
        * 255.0
    ).astype(np.uint8)  # grayscale, values 0 or 255

    # instance_seg_logits:  (1, C, H, W)  float in [0, 1] after sigmoid
    instance_pred: np.ndarray = (
        (
            outputs["instance_seg_logits"]
            .squeeze()  # (C, H, W)
            .detach()
            .cpu()
            .numpy()
            * 255.0
        )
        .transpose(1, 2, 0)
        .astype(np.uint8)
    )  # (H, W, C)

    # ------------------------------------------------------------------
    # 5. Encode to JPEG bytes in memory — no disk I/O
    # ------------------------------------------------------------------
    ok_bin, buf_bin = cv2.imencode(".jpg", binary_pred)
    if not ok_bin:
        raise RuntimeError("cv2.imencode failed for binary segmentation output")

    ok_inst, buf_inst = cv2.imencode(".jpg", instance_pred)
    if not ok_inst:
        raise RuntimeError("cv2.imencode failed for instance segmentation output")

    return {
        "binary": buf_bin.tobytes(),
        "instance": buf_inst.tobytes(),
    }