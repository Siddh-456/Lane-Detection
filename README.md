<div align="center">
  <h1>SEE LANE</h1>
  <p><strong>Lane detection studio built with PyTorch and Flask, featuring image, video, and live camera inference.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-LaneNet-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
    <img src="https://img.shields.io/badge/Modes-Image%20%7C%20Video%20%7C%20Live-c41028?style=for-the-badge" alt="Modes" />
  </p>
</div>

## Overview

**See Lane** is a lane detection web application built on top of the LaneNet approach from the paper **"Towards End-to-End Lane Detection: an Instance Segmentation Approach."** It loads a trained PyTorch model once at startup and exposes a Flask-powered interface for running lane detection on:

- single images
- uploaded videos
- live browser camera frames

For every inference, the app produces two outputs:

- a **binary segmentation mask** showing lane vs. background
- an **instance segmentation map** coloring each detected lane separately

The current interface is designed around a cinematic landing page, a built-in methodology panel, and three dedicated operating modes for image, video, and live usage.

Instead of limiting the project to a research-only script, this repository turns lane detection into a complete interactive product. The frontend is responsible for making the workflow approachable, while the backend handles inference safely through a shared model instance and a guarded prediction path. That combination makes the project useful for demonstrations, experimentation, and understanding how an ML model can be wrapped into a real user-facing application.

## Why See Lane

See Lane is valuable because it combines three things that are often split across separate projects:

- a trained deep learning lane detector
- a polished interface for interacting with the model
- multiple operating modes for different real-world input types

That means the repository is not just about generating masks. It is also about showing how computer vision inference can be presented, explained, and tested in a way that feels practical. A visitor can inspect the methodology, test a single image, run a recorded road clip, or open the live camera mode without switching tools.

## Core Capabilities

- **Single image inference** for quick testing and side-by-side output review
- **Video processing** for full clip analysis with exported MP4 outputs
- **Live camera inference** for interactive demonstrations and rapid checks
- **Binary segmentation output** to isolate lane markings from the background
- **Instance segmentation output** to visually separate different lane lines
- **Built-in methodology panel** to explain the theory behind the model
- **Flask-based API routes** for image, video, and live frame prediction
- **Thread-safe inference flow** to avoid overlapping access to the same loaded model

## Front Page

The landing screen introduces the project identity and the supported stack before the user enters the detector workflow.

<p align="center">
  <img src="./images/image%20(29).png" alt="See Lane landing page" width="100%" />
</p>

## Methodology

The methodology panel is the in-app theory and documentation surface. It explains the research basis, system structure, algorithm flow, inference pipeline, evaluation metrics, deployment options, and citations directly inside the web experience.

<p align="center">
  <img src="./images/image%20(30).png" alt="See Lane methodology panel" width="100%" />
</p>

This section is useful because it connects the polished UI to the actual ML pipeline behind it instead of treating the project as a black-box demo.

The methodology area also improves the project as a portfolio piece. Someone reviewing the repository can understand not only what the interface looks like, but also what architectural choices, loss functions, and deployment ideas sit behind the visual layer. That makes the README and the web app reinforce each other instead of repeating the same surface-level description.

At the center of the methodology is the idea that lane detection is not only a classification problem. The model must decide which pixels belong to lanes, preserve long thin structures, and separate neighboring lane markings cleanly enough to be useful. That is why this project is built around **LaneNet**, which frames the task as a combination of **binary segmentation** and **instance-aware embedding prediction** rather than simple edge detection or box detection.

## Architecture

`image (23)` shows the **LaneNet dual-branch architecture** used by the project.

<p align="center">
  <img src="./images/image%20(23).png" alt="LaneNet architecture diagram" width="100%" />
</p>

What the diagram shows:

- the input road image is first processed by a **shared encoder**
- the network then splits into a **segmentation branch** and an **embedding branch**
- the segmentation branch predicts the **binary lane mask**
- the embedding branch predicts **pixel embeddings** used to separate one lane instance from another
- clustering combines those embeddings with the segmentation output to generate the final colored lane instances

This is why the app can return both a clean binary lane map and a color-coded instance result from the same forward pass.

From a system design point of view, this is a strong fit for lane analysis because lane markings are thin, elongated structures that need both accurate localization and separation between neighboring lanes. A plain classifier would not be enough, and a detector based only on boxes would lose lane shape. The LaneNet design solves that by keeping dense pixel-level understanding while still distinguishing separate lane instances.

## Methodology and Backbone Choice

This repository supports multiple backbones inside the same LaneNet wrapper:

- **ENet**
- **UNet**
- **DeepLabv3+**

That means the overall task stays the same, but the encoder-decoder backbone can be swapped depending on whether the priority is speed, detail preservation, or heavier semantic understanding.

### Why LaneNet

LaneNet is a strong choice here because it naturally matches the structure of the output the app needs to show:

- one branch predicts a **binary lane mask**
- another branch predicts **instance embeddings**
- both outputs can be visualized immediately in the UI

This is better suited to the project than a plain semantic segmenter because the interface is designed to show not only where lanes are, but also how different lane instances separate from each other.

### ENet

ENet is the default backbone used by the running app. In this repository, `infer.py` loads:

```python
model = LaneNet(arch="ENet")
```

ENet is a practical fit because it was built for real-time semantic segmentation. The encoder uses:

- an initial split between convolution and max-pooling
- lightweight bottleneck modules
- dilated convolutions for larger receptive fields
- asymmetric convolutions for efficiency
- relatively small feature sizes compared with heavier backbones

For a project that includes image mode, full video mode, and live camera mode, ENet gives the best overall balance. It keeps inference responsive enough for an interactive product while still learning lane structure well enough to produce clean binary and instance outputs.

### UNet

UNet is also available in the repo and is a good alternative when preserving fine spatial detail is more important than raw speed. Its strengths here are:

- skip connections between encoder and decoder stages
- strong reconstruction of thin lane boundaries
- simpler segmentation-style structure that is easy to understand and debug

The tradeoff is that UNet is heavier than ENet and usually less attractive for a live interactive application where latency matters. It is a reasonable option for experiments, but not the most practical default for the web app experience this project is built around.

### DeepLabv3+ and ResNet-Style Backbones

The repository also includes a `DeepLabv3+` option, which is the path that brings in a much heavier backbone family. In practice, this is the closest thing in the repo to the "why not ResNet?" question, because DeepLab-style segmentation typically leans on deeper feature extractors for stronger semantics.

Those larger backbones can help when maximum segmentation accuracy is the only goal, but they come with real costs:

- more computation per frame
- higher memory use
- slower inference during video and live processing
- less suitable behavior for a lightweight interactive demo app

### Why ENet Instead of ResNet

The short answer is: **this project values responsiveness as much as accuracy**.

ResNet-based backbones are powerful, but this app is not only an offline benchmark runner. It is meant to:

- respond quickly to image uploads
- process full videos without feeling too heavy
- support live camera predictions in a usable way

For those goals, ENet makes more sense than a heavy ResNet-style encoder. It is lighter, faster, and more aligned with the product feel of the application. The choice is not "ResNet is bad"; it is that ENet is the better engineering decision for a lane detection studio that people actually interact with in real time.

## Model and Data Notes

The repository uses an ENet-based LaneNet implementation and loads weights from `log/best_model.pth`. Frames are resized to `512 x 256` before inference, normalized with ImageNet statistics, and then pushed through the network. The output is transformed into:

- a grayscale binary mask for lane vs non-lane pixels
- a colorized instance map that helps identify separate lane structures

The project is built around the TuSimple-style lane detection setup, which is well suited for road-scene benchmarks and structured driving footage. This makes the app especially effective for demo inputs involving highways, straight roads, and typical dashcam viewpoints.

The dataset loader expects three aligned paths per sample:

- the road image
- the binary lane annotation
- the instance label annotation

During loading, the binary mask is reconstructed by treating non-black label pixels as lane pixels, while the instance label image is kept for the discriminative embedding loss. That makes the training pipeline consistent with the two-output design of LaneNet.

## Training Strategy

The training flow in this repo is defined in `train.py` and `model/lanenet/train_lanenet.py`. The setup is straightforward and practical:

- the dataset is read from `train.txt` and `val.txt`
- images are resized to `512 x 256`
- training images receive `ColorJitter` augmentation
- normalization uses ImageNet mean and standard deviation
- optimization uses **Adam**
- the default learning rate is **0.0001**
- the default batch size is **4**
- the default epoch count is **25**

That makes the training configuration relatively lightweight and accessible, which fits the rest of the project.

### How the Model Is Trained

Training is multi-task. The model does not learn only a single mask output. Instead, it learns two connected tasks at once:

1. classify each pixel as lane or background
2. learn embedding features that separate different lane instances

The binary branch can use either:

- **Focal Loss** as the default
- **CrossEntropyLoss** as an alternative

In this codebase, the default is **Focal Loss**, which is a sensible choice because road scenes are dominated by background pixels. Lane markings occupy a small fraction of the image, so a loss that focuses learning on the harder and rarer lane pixels is more useful than a plain unweighted objective.

For the instance branch, the repo uses **Discriminative Loss**. This encourages:

- pixels from the same lane to move closer in embedding space
- pixels from different lanes to move farther apart

The combined objective in `compute_loss()` is:

```text
total_loss = 10 * binary_loss + 0.3 * var_loss + 1.0 * dist_loss
```

This weighting tells a clear story about the training priorities:

- binary lane detection is heavily emphasized
- instance separation still matters, but it is treated as a supporting objective
- the final model is optimized to stay visually useful in the app, not just theoretically elegant

### What "How I Trained the Model" Means in This Repo

Based on the actual training script, the model training process in this project is:

1. prepare TuSimple-style training and validation text files
2. resize images and labels to `512 x 256`
3. apply color augmentation only on training images
4. create a LaneNet model with the selected backbone
5. optimize with Adam for the requested number of epochs
6. track training and validation loss each epoch
7. keep the best validation checkpoint
8. save `training_log.csv` and `best_model.pth` into `log/`

That means the README can honestly describe this as a trained lane-detection pipeline, not just a pretrained demo dropped into a UI.

## Inference Modes

### Image Mode

Image mode is the fastest way to test the model on a single frame. The user uploads a road image, runs inference, and immediately sees the original frame beside both segmentation outputs.

<p align="center">
  <img src="./images/image%20(31).png" alt="Image mode in See Lane" width="100%" />
</p>

Image mode details:

- upload flow uses the browser form on the main page
- inference is sent to `POST /predict`
- the backend returns base64-encoded binary and instance output images
- good for quick validation, screenshots, and single-frame inspection

This mode is ideal when the goal is clarity. A user can inspect one road frame carefully, compare the original image against both outputs, and understand how the model behaves without the distraction of motion or streaming latency.

### Video Mode

Video mode processes a full uploaded video and writes output videos for both prediction branches.

<p align="center">
  <img src="./images/image%20(32).png" alt="Video mode in See Lane" width="100%" />
</p>

Video mode details:

- accepts a video file from the UI
- sends the upload to `POST /predict-video`
- reads the input video frame by frame with OpenCV
- runs lane detection per frame under a shared inference lock
- exports `binary_output.mp4` and `instance_output.mp4` into `video_output/`
- useful for demo reels, recorded driving clips, and end-to-end playback review

This mode is especially helpful for presenting the project in a more realistic scenario. Instead of proving that the model works on one carefully chosen frame, it shows how predictions evolve over time and how stable the lane outputs remain across a continuous driving sequence.

### Live Mode

Live mode connects to the browser camera and keeps sending fresh frames to the backend for near-real-time predictions.

<p align="center">
  <img src="./images/image%20(33).png" alt="Live mode in See Lane" width="100%" />
</p>

Live mode details:

- starts with browser camera permission
- captures frames from the live feed in the frontend
- sends frames to `POST /predict-live-frame`
- updates binary and instance outputs continuously
- shows approximate FPS in the UI
- best for interactive testing and real-time demo sessions

Live mode gives the project the most immediate feel. It turns the detector into an active system instead of a static showcase and makes it easier to test responsiveness, visual update speed, and user interaction in a classroom, review, or demo setting.

## End-to-End Flow

The high-level flow across the application is straightforward:

1. The browser collects an image, video, or live frame.
2. The frontend sends the selected input to the matching Flask endpoint.
3. The backend normalizes the input into the format expected by the model.
4. The loaded LaneNet model performs inference.
5. Binary and instance outputs are encoded for display or export.
6. The frontend renders the original input beside the prediction results.

That flow is reused consistently across all three modes, which keeps the project easier to maintain and easier to explain.

## How It Works

At runtime, the Flask app preloads the model from [`log/best_model.pth`](./log/best_model.pth). Incoming images are resized to `512 x 256`, normalized with ImageNet statistics, and passed through the ENet-based LaneNet model. The inference helper returns:

- `binary`: JPEG bytes for the lane mask
- `instance`: JPEG bytes for the colored instance map

For video processing, each decoded frame is converted into a PIL image, passed through the same predictor, and then re-encoded into output MP4 files using `imageio_ffmpeg`.

The backend is structured so the model is loaded only once instead of on every request. That reduces startup overhead during repeated predictions. A shared inference lock is also used to prevent multiple requests from stepping on each other when they try to access the model at the same time. For a single-machine deployment, that is a simple and effective safeguard.

On the frontend side, `index.html` switches between image, video, and live sections while keeping the same overall design language. This helps the experience feel like one coherent application rather than three disconnected tools packed into the same page.

## API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Serves the web interface |
| `/predict` | `POST` | Runs image inference |
| `/predict-video` | `POST` | Runs frame-by-frame video inference |
| `/predict-live-frame` | `POST` | Runs inference on one live camera frame |
| `/output/<filename>` | `GET` | Serves generated MP4 output files |

## Tech Stack

- **Python** for the backend runtime and model integration
- **Flask** for routing, request handling, and serving the web app
- **PyTorch** for the LaneNet model and tensor inference
- **OpenCV** for frame handling during video processing
- **Pillow** for image loading and RGB conversion
- **NumPy** for tensor and image array manipulation
- **imageio-ffmpeg** for writing processed MP4 outputs
- **HTML, CSS, and JavaScript** for the interactive frontend experience

## Run Locally

```bash
pip install torch torchvision flask pillow numpy opencv-python pandas scikit-image imageio-ffmpeg
python app.py
```

Then open:

```text
http://localhost:5000
```

After launching the server, you can:

- open the landing page and explore the design
- upload a single image in photo mode
- upload a road video in video mode
- enable camera access and test live mode
- open the methodology panel to review the theory and pipeline explanation

## Use Cases

- showcasing a deep learning project in a portfolio
- demonstrating lane detection visually in a classroom or presentation
- testing how a trained model behaves on custom road images
- comparing still-image results against continuous video output
- creating a base for future autonomous driving or road-scene tooling

## Project Layout

```text
Lane detector/
|-- app.py
|-- infer.py
|-- index.html
|-- README.md
|-- dataloader/
|-- images/
|-- log/
|   |-- best_model.pth
|   `-- training_log.csv
|-- model/
`-- video_output/
```

## Repository Highlights

Some parts of the repository are especially central to how the project works:

- `app.py` wires the web routes, model usage, and output serving together
- `infer.py` handles model loading, preprocessing, and per-frame prediction
- `index.html` contains the full visual interface and mode-switching behavior
- `model/` stores the neural network implementation
- `images/` contains the screenshots used to present the interface
- `video_output/` stores exported processed videos from video mode

## Highlights

- single-page interface with image, video, and live modes
- methodology overlay built directly into the frontend
- LaneNet-style dual-output prediction flow
- Flask backend with separate endpoints for image, video, and live inference
- thread lock to avoid overlapping inference work on the same model instance

## Reference

- Neven et al., **Towards End-to-End Lane Detection: an Instance Segmentation Approach**  
  https://arxiv.org/abs/1802.05591
