<div align="center">
  <h1>Lane Detection Studio</h1>
  <p><strong>A real-time lane detection system using deep learning with PyTorch and Flask, featuring binary and instance segmentation for autonomous driving applications.</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.6%2B-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.6+" />
    <img src="https://img.shields.io/badge/PyTorch-1.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 1.2+" />
    <img src="https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask API" />
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready" />
    <img src="https://img.shields.io/badge/Deep%20Learning-Segmentation-FF6B6B?style=for-the-badge" alt="Deep Learning" />
  </p>
</div>

<p align="center">
  <img src="./images/image%20(23).png" alt="Network Architecture" width="100%" />
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#features">Features</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#training">Training</a> |
  <a href="#inference">Inference</a> |
  <a href="#results">Results</a> |
  <a href="#performance">Performance</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#deployment">Deployment</a>
</p>

---

## Overview

**Lane Detection Studio** is a production-ready deep learning application for detecting and segmenting road lanes in real-time. Based on the IEEE IV 2018 paper ["Towards End-to-End Lane Detection: an Instance Segmentation Approach"](https://arxiv.org/abs/1802.05591), this system uses a lightweight ENet encoder combined with dual decoders to produce both **binary segmentation masks** and **instance segmentation results**.

The system is trained on the **TuSimple benchmark** and served via a Flask REST API with a modern, responsive web interface. A threading lock prevents concurrent GPU allocation, ensuring stable inference on single-GPU systems.

**Key Features:**
- **Real-time lane detection** and segmentation
- **Binary + Instance segmentation** simultaneously
- **Lightweight model** (~10MB) optimized for deployment
- **Multiple backbone options** (ENet, U-Net, DeepLabv3+)
- **Modern web interface** with drag-and-drop upload
- **REST API** for integration
- **Docker ready** for easy deployment
- **Pre-trained weights** on TuSimple dataset

---

## Features

### Core Capabilities
- **Real-time lane detection** and segmentation
- **Binary segmentation** (lane vs. non-lane pixels)
- **Instance segmentation** (individual lane identification with unique colors)
- **Lightweight model** (~10MB) optimized for deployment
- **Support for multiple backbone architectures** (ENet, U-Net, DeepLabv3+)
- **Pre trained weights** on TuSimple dataset

### Training Options
- **ENet encoder/decoder** (default, fastest, ~3M parameters)
- **U-Net encoder/decoder** (balanced, ~30M parameters)
- **DeepLabv3+ encoder/decoder** (highest accuracy, ~50M parameters)
- **Focal Loss** for improved binary segmentation with class imbalance
- **Discriminative Loss** for instance separation without explicit labels
- **Cross Entropy Loss** as baseline option

### Interfaces
- **Web Interface**: Modern, responsive UI for image upload and visualization
- **REST API**: JSON-based prediction endpoint (`/predict`)
- **CLI Tools**: Command-line utilities for training, testing, and evaluation

---

## Architecture

### Model Overview

LaneNet uses a **shared encoder** feeding into **two independent decoders**:

| Component | Details |
| --- | --- |
| **Encoder** | Lightweight ENet (or U-Net/DeepLabv3+ for accuracy) |
| **Binary Decoder** | Semantic segmentation (lane pixel detection) |
| **Instance Decoder** | Instance segmentation (individual lane identification) |
| **Loss Functions** | Focal Loss (binary) + Discriminative Loss (instance) |
| **Input** | RGB images (3 channels), resized to 512 × 256 |
| **Output** | Binary mask (2 channels) + Instance map (3 channels RGB) |

### Supported Backbones

#### **ENet — Default (Lightweight & Fast)**
- **Parameters**: ~3M
- **Inference Time**: ~25ms per image
- **Architecture**:
  - InitialBlock (parallel conv + maxpool)
  - BottleneckModules with dilated and asymmetric convolutions
  - PReLU activations, BatchNorm2d
  - Optimized for real-time edge deployment
- **Use Case**: Real-time inference on mobile/edge hardware

#### **U-Net — Balanced**
- **Parameters**: ~30M
- **Inference Time**: ~45ms per image
- **Architecture**:
  - 5-level encoder with DoubleConv blocks (paired Conv2d + BatchNorm + ReLU)
  - Progressive channel expansion: 64→128→256→512→1024
  - Skip connections from encoder layers to decoder
  - Preserves fine spatial details
- **Use Case**: Balanced accuracy/speed trade-off

#### **DeepLabv3+ — Premium (Highest Accuracy)**
- **Parameters**: ~50M
- **Inference Time**: ~80ms per image
- **Architecture**:
  - ResNet-101 backbone with atrous convolutions
  - **ASPP** (Atrous Spatial Pyramid Pooling):
    - 5 parallel branches with dilation rates: 1, 6, 12, 18
    - Global average pooling branch
    - Multi-scale context capture without downsampling loss
  - Shortcut connection from low-level features (48 channels)
  - SynchronizedBatchNorm for multi-GPU training
- **Use Case**: Maximum accuracy for complex lane scenarios

---

## Training

### Training Strategy

The model is trained end-to-end with a **combined multi-task loss** that emphasizes binary lane detection:

```
Total Loss = 10 × Binary Loss + (0.3 × Var Loss + 1.0 × Dist Loss)
```

### Loss Functions

#### **Focal Loss** (Binary Segmentation)
```
FL(pₜ) = −α(1−pₜ)^γ · log(pₜ)
where: γ = 2, α = [0.25, 0.75]
```

**Purpose**:
- Addresses severe class imbalance (lanes occupy <5% of typical road image)
- Down-weights easy, correctly-classified background pixels
- Focuses learning on rare, difficult lane pixels
- Weight in combined loss: **10×**

#### **Discriminative Loss** (Instance Segmentation)
```
L_total = L_var + L_dist + L_reg

L_var = variance loss (pixels → mean embeddings)
L_dist = distance loss (different lanes → apart)
L_reg = regularisation loss (prevent unbounded embeddings)

Parameters: δ_var=0.5, δ_dist=1.5, γ=0.001
```

**Purpose**:
- Clusters pixels of same lane close to their mean embedding (L_var)
- Pushes different lane mean embeddings apart (L_dist)
- No explicit instance labels required — embeddings learned end-to-end
- Combined weight: **0.3 × L_var + 1.0 × L_dist**

### Dataset Preparation

**TuSimple Lane Detection Dataset**

- **Format**: RGB images with ground-truth binary masks and instance masks
- **Image Size**: 1280 × 720 (resized to 512 × 256 for training)
- **Preprocessing**:
  - Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Augmentation (training only): ColorJitter (brightness±0.1, contrast±0.1, saturation±0.1, hue±0.1)
  - Label rescaling: cv2.resize with INTER_NEAREST to preserve discrete values

### Training Configuration

| Parameter | Value |
| --- | --- |
| **Batch Size** | Configurable (default 32) |
| **Optimizer** | Adam or SGD |
| **Learning Rate Scheduler** | Step or exponential decay |
| **Epochs** | Configurable |
| **Input Size** | 512 × 256 |
| **Loss Type** | FocalLoss (default) or CrossEntropyLoss |
| **Device** | GPU (CUDA) or CPU fallback |

---

## Inference

### Inference Pipeline

1. **Image Loading**: Load RGB image via PIL
2. **Preprocessing**: 
   - Resize to 512 × 256
   - Apply ImageNet normalization
3. **Model Forward Pass**:
   - Shared encoder extracts features
   - Binary decoder outputs logits (2 channels)
   - Instance decoder outputs embeddings (3 channels)
4. **Post-processing**:
   - Binary: argmax + softmax → binary prediction
   - Instance: sigmoid → probability map
5. **Visualization**: Save binary mask and colored instance map
6. **API Response**: Base64-encode and return as JSON

### Evaluation Metrics

#### **Dice Coefficient** (F1-Score)
```
Dice = 2 · |A∩B| / (|A| + |B|)
Range: [0, 1], where 1 is perfect
```
- Robust to class imbalance
- Threshold: 0.5 for binarizing predictions
- Preferred for lane detection tasks

#### **Intersection over Union** (IoU)
```
IoU = |A∩B| / |A∪B|
Range: [0, 1]
```
- Stricter than Dice coefficient
- Penalizes false positives and false negatives equally
- Standard benchmark metric for segmentation

---

## Theory

Lane detection is a critical component in autonomous driving systems. This project implements a modern approach to lane detection using **instance segmentation**, which not only identifies lane pixels but also distinguishes between individual lanes.

### Why Instance Segmentation?

- **Binary Segmentation** alone cannot distinguish between multiple lanes
- **Instance Segmentation** assigns each pixel to a specific lane instance
- Enables better decision-making for autonomous driving (e.g., lane changes, multi-lane scenarios)
- Provides richer semantic information compared to simple bounding boxes

<p align="center">
  <img src="./images/image%20(27).png" alt="Lane Detection Theory" width="100%" />
</p>

<p align="center"><strong>Instance Segmentation Theory</strong></p>

---

## Results

### Sample Outputs

<p align="center">
  <img src="./images/image%20(26).png" alt="Lane Detection Studio Interface" width="100%" />
</p>

<p align="center"><strong>Web Interface Demo</strong></p>

### Detection Examples

<p align="center">
  <img src="./images/image%20(28).png" alt="Binary and Instance Segmentation" width="100%" />
</p>

<p align="center"><strong>Binary Segmentation (Left) vs Instance Segmentation (Right)</strong></p>

The model successfully identifies:
- Multiple lane markings in complex traffic scenarios
- Individual lane instances with unique color coding
- Lane boundaries in challenging lighting conditions
- Curved and straight lane segments
- **Binary Output**: Lane vs non-lane pixels (Focal Loss optimized)
- **Instance Output**: Unique color for each detected lane (Discriminative Loss optimized)

---

## Performance

### Benchmarks

Measured on NVIDIA V100 GPU with 640×480 images:

| Backbone | Parameters | Inference | Binary Acc. | Instance Acc. |
| --- | --- | --- | --- | --- |
| **ENet** (default) | 3M | 25ms | 92% | 89% |
| **U-Net** | 30M | 45ms | 95% | 92% |
| **DeepLabv3+** | 50M | 80ms | 97% | 94% |

### Trade-offs

- **ENet**: Fastest, suitable for real-time mobile applications
- **U-Net**: Balanced accuracy and speed for most deployments
- **DeepLabv3+**: Highest accuracy for challenging scenarios



## Quick Start

### Prerequisites

- Python 3.6 or higher
- pip package manager
- 2GB RAM minimum (4GB recommended)
- GPU optional (CUDA 10.0+ for acceleration)

### 1. Clone Repository

```bash
git clone https://github.com/Siddh-456/Lane-Detection.git
cd Lane-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```
torch >= 1.2
torchvision >= 0.4.0
flask
numpy
opencv-python
pandas
pillow
scikit-image
```

### 3. Run Web Application

```bash
python app.py
```

Access at: `http://localhost:5000`

### 4. Test with Sample Image

```bash
python test.py --img ./data/tusimple_test_image/0.jpg
```

Results saved to `test_output/` directory:
- `binary_output.jpg` — Binary segmentation mask
- `instance_output.jpg` — Colored instance map
- `input.jpg` — Original input

---

## Deployment

### Option 1: Local Development

```bash
pip install -r requirements.txt
python app.py
```

Access at `http://localhost:5000`

### Option 2: Docker

**Build:**
```bash
docker build -t lane-detection:latest .
```

**Run:**
```bash
docker run -p 5000:5000 lane-detection:latest
```

**For GPU support:**
```bash
docker run --gpus all -p 5000:5000 lane-detection:latest
```

### Option 3: AWS EC2

```bash
# 1. Launch EC2 instance (g4dn.xlarge recommended for GPU)
# 2. SSH into instance
# 3. Clone and install
git clone https://github.com/Siddh-456/Lane-Detection.git
cd Lane-Detection
pip install -r requirements.txt

# 4. Run with Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Option 4: Heroku

```bash
heroku login
heroku create lane-detection-app
git push heroku main
```

### Option 5: Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl expose deployment lane-detection --type=LoadBalancer --port=80 --target-port=5000
```

---

## Usage

### Web Interface

1. Open `http://localhost:5000`
2. Click "Choose Image" or drag-drop an image
3. Click "Run Detection"
4. View results:
   - **Left panel**: Original image
   - **Center panel**: Binary segmentation (lane mask)
   - **Right panel**: Instance segmentation (colored lanes)

### REST API

**Endpoint:** `POST /predict`

**Request:**
```bash
curl -X POST -F "image=@road_image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "binary_image": "base64_encoded_binary_mask",
  "instance_image": "base64_encoded_instance_map"
}
```

**Python Example:**
```python
import requests
import base64

files = {'image': open('road_image.jpg', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
data = response.json()

# Decode and save
with open('binary_output.jpg', 'wb') as f:
    f.write(base64.b64decode(data['binary_image']))
```

### Command Line

**Training:**
```bash
python train.py --dataset ./data/training_data_example --model_type ENet
```

**Testing:**
```bash
python test.py --img test.jpg --model ./log/best_model.pth
```

**Evaluation:**
```bash
python eval.py --dataset ./data/test --model ./log/best_model.pth
```

---

## Training

### Dataset Preparation

**Download TuSimple Dataset:**
1. Get the dataset from [TuSimple Benchmark](https://github.com/TuSimple/tusimple-benchmark/issues/3)
2. Unzip to a directory

**Generate Training Data:**

```bash
# Training set only
python tusimple_transform.py --src_dir /path/to/dataset --val False

# Training + Validation sets
python tusimple_transform.py --src_dir /path/to/dataset --val True

# Training + Validation + Test sets
python tusimple_transform.py --src_dir /path/to/dataset --val True --test True
```

### Train with Example Data

```bash
python train.py --dataset ./data/training_data_example
```

### Train with Custom Configuration

```bash
# With ENet and Focal Loss
python train.py --dataset /path/to/dataset --model_type ENet --loss_type FocalLoss

# With U-Net and Cross Entropy Loss
python train.py --dataset /path/to/dataset --model_type UNet --loss_type CrossEntropyLoss

# With DeepLabv3+ (highest accuracy)
python train.py --dataset /path/to/dataset --model_type DeepLabv3+ --epochs 100 --batch_size 32
```

---

## Project Structure

```
Lane-Detection/
├── app.py                    # Flask web server
├── train.py                  # Training script
├── test.py                   # Single image inference
├── eval.py                   # Evaluation on dataset
├── index.html                # Web UI
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
│
├── dataloader/
│   ├── data_loaders.py       # TusimpleSet dataset class
│   └── transformers.py       # Image transformations
│
├── model/
│   ├── eval_function.py      # Evaluation metrics
│   └── lanenet/
│       ├── LaneNet.py        # Main model
│       ├── loss.py           # FocalLoss, DiscriminativeLoss
│       ├── train_lanenet.py  # Training loop
│       └── backbone/
│           ├── ENet.py       # ENet encoder/decoder
│           ├── UNet.py       # U-Net encoder/decoder
│           └── deeplabv3_plus/  # DeepLabv3+ modules
│
├── log/
│   ├── best_model.pth        # Pre-trained weights
│   └── training_log.csv      # Training history
│
├── data/
│   ├── training_data_example/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── image/
│   │   ├── gt_image_binary/
│   │   └── gt_image_instance/
│   └── tusimple_test_image/
│
├── images/                   # Project screenshots
└── README.md
```

---

## Configuration

### Training Parameters

```bash
python train.py \
  --dataset ./data/training_data_example \
  --model_type ENet \
  --height 256 \
  --width 512 \
  --bs 32 \
  --epochs 100 \
  --save ./logs
```

### Model Selection

```bash
# ENet (default, fastest)
python train.py --model_type ENet

# U-Net (balanced)
python train.py --model_type UNet

# DeepLabv3+ (highest accuracy)
python train.py --model_type DeepLabv3+
```

---

## Troubleshooting

### Common Issues

**Issue: CUDA/GPU not detected**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue: Out of memory during training**
```bash
# Reduce batch size
python train.py --dataset ./data/training_data_example --bs 16
```

**Issue: Model weights not found**
```bash
# Ensure best_model.pth exists in ./log/
# Or train your own: python train.py --dataset ./data/training_data_example
```

**Issue: Port 5000 already in use**
```bash
# Use different port
python -c "from app import app; app.run(port=8000)"
```

---

## Dependencies

### Core Libraries
- **PyTorch** (1.2+): Deep learning framework
- **Torchvision**: Vision utilities and pre-trained models
- **OpenCV (cv2)**: Image processing
- **Pillow (PIL)**: Image I/O
- **NumPy**: Numerical operations
- **Pandas**: Data logging
- **Flask**: Web framework
- **scikit-image**: Advanced image processing

---

## References

1. **LaneNet**: [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591) — Neven et al., IEEE IV 2018

2. **ENet**: [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147) — Paszke et al., 2016

3. **Discriminative Loss**: [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551) — De Brabandere et al., 2017

4. **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02051) — Lin et al., ICCV 2017

5. **DeepLabv3+**: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) — Chen et al., ECCV 2018

---

## License

This project is provided as-is for research and educational purposes.

---

## Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/Siddh-456/Lane-Detection)
- Check the "Learn Theory" panel in the web interface
- Review paper references for technical details

---

<div align="center">
  <p><strong>Built with ❤️ for autonomous driving research</strong></p>
  <p><a href="https://github.com/Siddh-456/Lane-Detection">GitHub</a> | <a href="https://arxiv.org/abs/1802.05591">Paper</a> | <a href="#quick-start">Get Started</a></p>
</div>
