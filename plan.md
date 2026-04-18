# Real-Time Lane Detection Expansion Plan

## Goal

Expand the current single-image LaneNet demo into a multi-mode application with:

1. `Photo` mode
   Upload one image and return:
   - original preview
   - binary segmentation
   - instance segmentation

2. `Video` mode
   Upload one video and process it frame by frame to produce:
   - processed binary video
   - processed instance video
   - optional side-by-side preview video

3. `Live` mode
   Ask browser permission for webcam access and process live frames continuously to show:
   - live binary segmentation
   - live instance segmentation
   - optional original camera feed

The final system should honestly support the term `real-time` for the live mode only after FPS is measured on the target hardware.

---

## Current State

The current repo already has a strong starting point:

- `app.py`
  - Flask app
  - serves `index.html`
  - accepts one uploaded image at `/predict`
- `index.html`
  - single-image upload UI
  - sends one file to Flask
  - renders returned base64 images
- `test.py`
  - loads model
  - runs inference for one image
  - writes outputs to `test_output/`
- `model/`
  - existing LaneNet architecture
  - existing weights path: `log/best_model.pth`

### Current limitations

The current implementation is not true real-time because:

- the model is loaded inside the inference path for each request
- output images are written to disk for each prediction
- the frontend only supports one uploaded image
- the Flask route is built around a single file input
- live webcam streaming is not implemented
- video decoding and re-encoding are not implemented

---

## High-Level Strategy

Do not rebuild the model architecture.

Instead:

1. Keep the trained LaneNet model and weights
2. Refactor inference into reusable functions
3. Keep Flask as the backend server
4. Extend the frontend to support mode switching
5. Add video and live-frame processing on top of the same shared inference core

### Important architecture decision

The best path is **not** to keep all inference logic trapped inside `test.py`.

Instead, create a shared inference module that:

- loads the model once
- preprocesses frames in memory
- runs inference without writing temporary output files
- returns arrays/images directly

This lets `app.py` reuse the same pipeline for:

- photo uploads
- video uploads
- live webcam frames

---

## What Can Stay the Same

These parts can remain conceptually the same:

- Flask backend
- existing HTML/CSS base
- existing LaneNet model code
- existing trained weights
- same preprocessing shape: `512 x 256`
- same binary and instance outputs

---

## What Must Change

### Backend

- `app.py` must support multiple routes or one route with mode-specific behavior
- the model should load once at startup, not per request
- inference should run in memory
- video processing support must be added
- live frame processing support must be added

### Frontend

- add mode switch: `Photo | Video | Live`
- show/hide correct controls for each mode
- ask for webcam permission using browser APIs
- capture frames from webcam using `getUserMedia()`
- send frames repeatedly to backend for live mode
- handle video upload and display returned result files or previews

### Inference layer

- extract reusable preprocessing from `test.py`
- convert model call into a reusable `predict_frame()` function
- stop using disk-based intermediary outputs for normal inference

---

## Dependencies

## Python packages

Install these first:

```bash
pip install flask pillow torch torchvision numpy pandas opencv-python scikit-image
```

### Why each dependency is needed

- `flask`
  Backend web server and API routes
- `pillow`
  Image upload parsing and conversion
- `torch`
  Model execution
- `torchvision`
  Resize and normalization transforms
- `numpy`
  Frame and tensor conversion
- `pandas`
  Already used in repo, not central for live mode but part of current project stack
- `opencv-python`
  Video decoding, frame extraction, output video writing, live image conversion
- `scikit-image`
  Already used by current dataloader utilities

## Optional Python packages

These are not required for version 1, but may help later:

```bash
pip install flask-socketio eventlet
```

Use them only if you later want websocket-based live streaming instead of repeated HTTP frame posts.

## System tools

Optional but useful:

- `ffmpeg`
  Helps if OpenCV video encoding behaves inconsistently on your machine

---

## Recommended File Plan

### Files to keep

- `app.py`
- `index.html`
- `test.py`
- `model/`
- `log/best_model.pth`

### Files to add

- `infer.py`
  Shared reusable inference utilities
- `video_utils.py`
  Optional helper for video read/write logic
- `plan.md`
  This roadmap

### Files likely to edit

- `app.py`
- `index.html`
- maybe `README.md` later

### Files to avoid overloading

- `test.py`

Keep `test.py` as a simple CLI tester if possible. Do not keep growing it into the core runtime for photo, video, and live mode.

---

## Detailed Build Phases

## Phase 0: Stabilize Inference Core

This is the foundation. Do this before touching live mode.

### Objective

Create a reusable inference path that works for one frame in memory.

### Tasks

1. Create `infer.py`
2. Add a model loader function:
   - `load_model(model_type="ENet", model_path="log/best_model.pth")`
3. Load the model once
4. Add a preprocessing function:
   - accepts PIL image or NumPy array
   - resizes to `512 x 256`
   - normalizes with ImageNet mean/std
5. Add a prediction function:
   - `predict_frame(image)`
6. Add postprocessing:
   - produce binary mask image
   - produce instance image
7. Return outputs as in-memory arrays or bytes
8. Wrap inference in:
   - `model.eval()`
   - `torch.inference_mode()`

### Why this phase matters

Without this step:

- every live frame will be too expensive
- every video frame will be unnecessarily slow
- code will become duplicated between photo, video, and live routes

### Success criteria

- one uploaded image can still be processed correctly
- outputs visually match the current implementation
- no output files are required for normal inference

---

## Phase 1: Improve Photo Mode

### Objective

Keep the existing photo upload flow, but route it through the new inference core.

### Tasks

1. Update `app.py`
   - replace `lane_test.test()`-style flow with `infer.py`
2. Keep a route such as:
   - `POST /predict-photo`
   or
   - continue using `POST /predict`
3. Make the route:
   - accept one uploaded image
   - parse with Pillow
   - call shared inference
   - return base64 output or direct image URLs
4. Update `index.html`
   - rename current single upload mode to `Photo`
   - preserve current preview behavior

### Recommended response format

For photo mode, JSON with base64 is okay:

```json
{
  "binary_image": "...",
  "instance_image": "..."
}
```

### Success criteria

- photo mode works exactly like now or better
- output quality stays the same
- response time improves because model is not reloaded each request

---

## Phase 2: Add Video Upload Mode

### Objective

Allow the user to upload a video and receive processed video outputs.

### Key design choice

Decide what the user should get back:

1. `binary` video only
2. `instance` video only
3. both separate videos
4. one merged side-by-side comparison video

### Best version for your goal

Return:

- one binary output video
- one instance output video
- optionally one merged preview video later

### Tasks

1. Frontend
   - add `Video` mode tab/button
   - show video file input when selected
   - accept common formats:
     - `.mp4`
     - `.avi`
     - `.mov`
     - `.mkv` if supported locally
   - show progress message

2. Backend
   - add route such as `POST /predict-video`
   - save uploaded video temporarily
   - open video using `cv2.VideoCapture`
   - read:
     - width
     - height
     - fps
     - total frames if available
   - process frame by frame
   - run shared `predict_frame()` on each frame
   - write output videos using `cv2.VideoWriter`

3. Output handling
   - generate temporary or static output files
   - return URLs or filenames to the frontend
   - let the browser display downloadable `<video>` elements

### Important implementation notes

- for video mode, disk output is acceptable
- for live mode, disk output is not acceptable for per-frame inference
- preserve original FPS if possible
- if original FPS metadata is invalid, fall back to a safe default like `20` or `25`

### Performance expectations

This mode does not need to be real-time while processing.

It can be asynchronous or “process and wait.”

### Success criteria

- uploaded video is processed completely
- output video is playable
- lanes appear consistently frame to frame
- no server crash on medium-length videos

---

## Phase 3: Add Live Camera Mode

### Objective

Allow the browser to request webcam access and show live processed segmentation.

### Core truth

Camera permission is handled by the browser, not by Flask.

### Frontend tasks

1. Add `Live` mode tab/button
2. Add a `Start Camera` button
3. Use:

```javascript
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
```

4. Show original camera stream in a `<video>` element
5. Draw current frame to a hidden `<canvas>`
6. Convert frame to JPEG or blob
7. Send frames repeatedly to backend
8. Show returned binary and instance outputs
9. Add `Stop Camera` button
10. Release camera tracks cleanly on stop

### Backend tasks

1. Add route such as `POST /predict-live-frame`
2. Accept one frame image from the browser
3. Convert to PIL or NumPy
4. Run shared inference
5. Return binary and instance outputs

### First implementation choice

Use repeated HTTP requests first, not websockets.

Why:

- simpler
- easier to debug
- works fine for a first version

### Live frame request strategy

Do not send every possible frame from the browser.

Instead:

- start with a controlled loop
- send one frame
- wait for response
- then send next frame

This prevents request piling and backend overload.

### Initial target

Start with:

- `320 x 180` or `416 x 234` capture frames for live transfer
- keep model input at `512 x 256` after backend preprocessing

### Why lower browser capture size helps

- smaller upload payload
- less browser-side encoding cost
- less network overhead even on localhost
- smoother perceived live mode

### Success criteria

- camera permission popup appears correctly
- original live feed shows in browser
- binary and instance outputs update continuously
- frame loop does not freeze browser

---

## Phase 4: Optimize for Real-Time Performance

This phase is what turns the live mode from “working” into “honestly real-time.”

### Backend optimizations

1. Load model once at startup
2. Use `torch.inference_mode()`
3. Keep tensors on the same device
4. Avoid unnecessary PIL <-> NumPy conversions
5. Avoid disk writes during live mode
6. Keep ENet as default model for live mode
7. Optionally use CUDA if available

### Frontend optimizations

1. Limit live frame send rate
2. Send compressed JPEG blobs rather than raw image data
3. Process next frame only after prior frame response completes
4. Show FPS indicator
5. Allow quality presets:
   - low
   - medium
   - high

### Optional advanced optimizations

1. Half precision on GPU if safe
2. WebSocket streaming
3. Batchless persistent CUDA warm state
4. Reuse allocated buffers where possible

### Important realism note

Do not claim `real-time` unless you measure actual FPS on your machine.

Good examples:

- `Live camera segmentation at 12 FPS on CPU`
- `Live camera segmentation at 24 FPS on NVIDIA GPU`

---

## Suggested API Design

Keep the API simple and explicit.

### Option A: separate routes

- `POST /predict-photo`
- `POST /predict-video`
- `POST /predict-live-frame`

This is the clearest and easiest to maintain.

### Option B: one route with mode parameter

- `POST /predict`
  - `mode=photo`
  - `mode=video`
  - `mode=live`

This works, but becomes messy faster.

### Recommendation

Use **separate routes**.

---

## Suggested Frontend Structure

## Main sections

1. Mode selector
   - Photo
   - Video
   - Live

2. Input area
   - image picker
   - video picker
   - live camera controls

3. Output area
   - original
   - binary
   - instance

4. Status area
   - ready
   - processing
   - error
   - FPS for live mode

## Recommended behavior

### Photo mode

- choose image
- preview image
- click run
- display binary and instance outputs

### Video mode

- choose video
- show filename and maybe original preview
- click run
- display progress text
- after processing, show output video players or download links

### Live mode

- click start camera
- browser asks permission
- show original live video
- start live inference loop
- update binary and instance panels continuously
- click stop camera to stop everything

---

## Detailed Technical Risks

## Risk 1: model reload on every frame

### Problem

If the model loads on every request, live mode will be too slow.

### Fix

Load model once globally or with lazy singleton initialization.

## Risk 2: disk writes for every live frame

### Problem

Saving images for every frame destroys performance.

### Fix

Keep everything in memory for photo and live mode.

## Risk 3: too many live requests

### Problem

If the browser sends frames too quickly, requests pile up.

### Fix

Use sequential request loop:

- send frame
- wait response
- send next frame

## Risk 4: video codec issues

### Problem

OpenCV sometimes writes videos with codec/container mismatch.

### Fix

- start with `.mp4` output
- test codecs like `mp4v`
- use `ffmpeg` later if needed

## Risk 5: browser performance

### Problem

Large webcam frame sizes can slow down capture and upload.

### Fix

Start with smaller capture resolution and scale carefully.

## Risk 6: inconsistent aspect ratio

### Problem

Your model input is fixed to `512 x 256`, but source media may vary.

### Fix

- resize consistently
- decide whether to preserve aspect ratio with padding or direct resize
- document the choice

---

## Recommended Implementation Order

Do the work in this exact order:

1. Create `infer.py`
2. Move reusable inference logic out of `test.py`
3. Update photo mode in Flask to use `infer.py`
4. Confirm photo mode still works
5. Add mode switch UI in `index.html`
6. Add video upload route and processing
7. Confirm video output generation works
8. Add live webcam UI
9. Add frame-by-frame live route
10. Add live FPS throttling and status handling
11. Optimize performance
12. Update `README.md` only after features are actually working

---

## Testing Checklist

## Photo mode tests

- upload JPG
- upload PNG
- upload invalid file
- confirm binary output appears
- confirm instance output appears
- confirm no crash on repeated uploads

## Video mode tests

- upload short MP4
- upload medium-length MP4
- test odd resolution video
- verify output video duration matches input approximately
- verify output video opens in browser

## Live mode tests

- camera permission allow
- camera permission deny
- start camera
- stop camera
- switch away from live mode cleanly
- confirm no orphan camera use after stop
- confirm repeated frame updates happen without piling requests

## Performance tests

- measure photo response time
- measure average video processing FPS
- measure live mode FPS
- compare CPU vs GPU if available

---

## Definition of Done

The project can be considered successful when:

- `Photo` mode works for standard image uploads
- `Video` mode processes uploaded videos frame by frame
- `Live` mode requests webcam permission and shows continuously updating segmentation
- the backend uses one shared reusable inference core
- the model is not reloaded per frame/request
- live mode performance is measured and documented honestly

---

## Recommended README Positioning After Completion

After implementation, use wording like this:

### If live mode is working and measured

`Multi-mode lane detection system supporting photo, video, and live camera inference with binary and instance segmentation.`

### If live mode works but FPS is modest

`Lane detection system for photo, video, and live camera segmentation with near real-time inference depending on hardware.`

### If only photo and video are complete

`Lane detection and segmentation system for image and video analysis.`

---

## Final Recommendation

Yes, your goal is fully achievable with this repo.

But the best implementation is:

- keep the current model
- keep Flask
- keep the frontend base
- add a reusable inference module
- then build photo, video, and live on top of it

Trying to keep the backend completely untouched is not recommended if you want good live performance.

The smallest smart change is:

- refactor inference once
- then reuse it everywhere

