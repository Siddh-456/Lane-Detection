import base64
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

try:
    import imageio_ffmpeg as _iio_ffmpeg

    _BUNDLED_FFMPEG = _iio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _BUNDLED_FFMPEG = None

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

import infer

BASE_DIR = Path(__file__).resolve().parent
VIDEO_OUTPUT_DIR = BASE_DIR / "video_output"
INFERENCE_LOCK = threading.Lock()

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

# Load model once at startup
infer.get_model()


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/output/<path:filename>")
def serve_output(filename):
    # Windows often guesses MP4 as application/octet-stream; browsers may refuse
    # inline <video> playback without a video/* Content-Type.
    kwargs = {}
    if filename.lower().endswith(".mp4"):
        kwargs["mimetype"] = "video/mp4"
    return send_from_directory(str(VIDEO_OUTPUT_DIR), filename, **kwargs)


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Please upload an image file."}), 400
    try:
        uploaded_file.stream.seek(0)
        try:
            image = Image.open(uploaded_file.stream)
        except Exception as exc:
            raise ValueError(
                "Unsupported image file. Please upload a valid JPG or PNG image."
            ) from exc
        image = ImageOps.exif_transpose(image).convert("RGB")
        with INFERENCE_LOCK:
            result = infer.predict_frame(image)
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(
        {
            "instance_image": base64.b64encode(result["instance"]).decode("ascii"),
            "binary_image": base64.b64encode(result["binary"]).decode("ascii"),
        }
    )


@app.post("/predict-video")
def predict_video():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Please upload a video file."}), 400

    if not _BUNDLED_FFMPEG:
        return jsonify({"error": "ffmpeg not available on this server."}), 500

    temp_path = None
    bin_writer = None
    inst_writer = None

    try:
        suffix = Path(uploaded_file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            uploaded_file.save(tf)
            temp_path = Path(tf.name)

        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        VIDEO_OUTPUT_DIR.mkdir(exist_ok=True)
        binary_path = VIDEO_OUTPUT_DIR / "binary_output.mp4"
        instance_path = VIDEO_OUTPUT_DIR / "instance_output.mp4"

        # Open both writers — imageio_ffmpeg pipes frames straight into libx264.
        # pix_fmt_in='rgb24' because we send converted RGB bytes.
        writer_kwargs = dict(
            size=(512, 256),
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            output_params=["-movflags", "+faststart", "-crf", "23"],
        )
        bin_writer = _iio_ffmpeg.write_frames(str(binary_path), **writer_kwargs)
        inst_writer = _iio_ffmpeg.write_frames(str(instance_path), **writer_kwargs)
        bin_writer.send(None)  # prime the generators
        inst_writer.send(None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            with INFERENCE_LOCK:
                result = infer.predict_frame(pil_image)

            bin_arr = cv2.imdecode(
                np.frombuffer(result["binary"], np.uint8), cv2.IMREAD_GRAYSCALE
            )
            inst_arr = cv2.imdecode(
                np.frombuffer(result["instance"], np.uint8), cv2.IMREAD_COLOR
            )

            if bin_arr is not None:
                # Grayscale → RGB so the writer gets rgb24 bytes
                bin_rgb = cv2.cvtColor(bin_arr, cv2.COLOR_GRAY2RGB)
                bin_writer.send(bin_rgb.tobytes())

            if inst_arr is not None:
                # OpenCV gives BGR → convert to RGB
                inst_rgb = cv2.cvtColor(inst_arr, cv2.COLOR_BGR2RGB)
                inst_writer.send(inst_rgb.tobytes())

        cap.release()

    except Exception as exc:
        app.logger.exception("Video prediction failed")
        return jsonify({"error": str(exc)}), 500

    finally:
        # Always close writers so ffmpeg flushes and finalises the MP4 atoms.
        for w in (bin_writer, inst_writer):
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return jsonify(
        {
            "binary_video": "/output/binary_output.mp4",
            "instance_video": "/output/instance_output.mp4",
        }
    )


@app.post("/predict-live-frame")
def predict_live_frame():
    uploaded_file = request.files.get("frame")
    if uploaded_file is None:
        return jsonify({"error": "No frame provided."}), 400
    try:
        uploaded_file.stream.seek(0)
        image = Image.open(uploaded_file.stream).convert("RGB")
        with INFERENCE_LOCK:
            result = infer.predict_frame(image)
    except Exception as exc:
        app.logger.exception("Live frame prediction failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(
        {
            "binary_image": base64.b64encode(result["binary"]).decode("ascii"),
            "instance_image": base64.b64encode(result["instance"]).decode("ascii"),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
