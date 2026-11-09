# 🤟 ASL Sign Language Recognition

An end-to-end American Sign Language (ASL) alphabet recognition system built
with PyTorch, OpenCV, MediaPipe, and Streamlit.

The included web app accepts an uploaded image or a camera snapshot, locates
the hand, removes distracting face regions, crops the hand, and classifies it
with a compact convolutional neural network (CNN). It recognizes the 26
letters plus `del`, `nothing`, and `space`.

> **Important:** This project recognizes isolated hand signs. It is not a
> continuous sign-language translator and does not interpret words, grammar,
> facial expressions, body movement, or two-handed signing.

## ✨ Features

- Upload `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` images.
- Capture a sign directly from a device camera.
- MediaPipe hand landmarks for robust hand localization when available.
- OpenCV skin-segmentation fallback when MediaPipe cannot detect a hand.
- Face detection and exclusion to reduce false hand crops.
- Top-5 predictions with probabilities and a confidence score.
- CPU and CUDA inference support.
- Reproducible training workflow in [`main.ipynb`](./main.ipynb).
- Pre-trained model included at [`models/asl_cnn_model.pth`](./models/asl_cnn_model.pth).

## 🖥️ Demo

Run the app locally:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually
`http://localhost:8501`.

For best results:

1. Use a well-lit image with one clearly visible hand.
2. Keep the hand reasonably large and inside the camera frame.
3. Use a plain or uncluttered background.
4. Avoid covering the hand or placing it directly over the face.

## 🚀 Quick start

### 1. Clone and enter the project

```bash
git clone <repository-url>
cd Sign-Language-Recognition-System
```

### 2. Create and activate a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

The repository already contains both model assets required by the app:

- `models/asl_cnn_model.pth` — trained CNN checkpoint and class mapping.
- `models/hand_landmarker.task` — MediaPipe hand-landmark model.

If `hand_landmarker.task` is unavailable or MediaPipe fails to initialize,
the app automatically falls back to OpenCV skin segmentation.

## 🧠 How it works

```text
Image upload / camera
        │
        ▼
Face detection ───────────────┐
        │                     │
        ▼                     │
MediaPipe hand detection      │
        │                     │
        └── fallback: OpenCV skin mask
                              │
                              ▼
                 Face exclusion + hand crop
                              │
                              ▼
                 Resize to 64 × 64 + normalize
                              │
                              ▼
                     PyTorch CNN inference
                              │
                              ▼
                 Sign, confidence, and top 5
```

### Inference preprocessing

The app applies the same deterministic evaluation preprocessing used by the
notebook:

- Convert to RGB.
- Resize to `64 × 64`.
- Convert to a tensor.
- Normalize each channel with mean `0.5` and standard deviation `0.5`.

The CNN contains three convolution/max-pooling blocks followed by a
512-unit fully connected layer and a 29-class output layer. The checkpoint
stores the model weights and class ordering, so inference does not depend on
reconstructing the label mapping manually.

## 📊 Dataset and reported results

The notebook uses an ASL alphabet image dataset arranged in
`ImageFolder` format:

| Split | Images | Purpose |
|---|---:|---|
| Training source | 87,000 | Model training and validation |
| Training subset | 69,600 | 80% stratified training split |
| Validation subset | 17,400 | 20% stratified validation split |
| Held-out test set | 28 | One image for each available test label |

The saved checkpoint was produced after five training epochs:

| Metric | Result |
|---|---:|
| Best validation accuracy | **99.33%** |
| Held-out test accuracy | **100.00% (28/28)** |
| Held-out test loss | **0.7385** |

These are notebook results, not a guarantee of real-world performance. The
held-out test set contains only 28 images, so it is too small to represent
different cameras, lighting conditions, skin tones, backgrounds, poses, or
signers. Confidence scores are model probabilities, not calibrated certainty.

## 🏋️ Training or reproducing the model

The complete data preparation, augmentation, training, evaluation, and model
export workflow is in [`main.ipynb`](./main.ipynb).

Start Jupyter from the project root:

```bash
python -m pip install jupyter pandas matplotlib seaborn scikit-learn torchsummary
jupyter notebook main.ipynb
```

The notebook expects this layout:

```text
Raw_Data/
├── asl_alphabet_train/asl_alphabet_train/
│   ├── A/
│   ├── B/
│   └── .../
└── asl_alphabet_test/asl_alphabet_test/
    ├── A_test.jpg
    ├── B_test.jpg
    └── .../
```

Training details:

- Input size: `64 × 64`.
- Batch size: `128`.
- Train/validation split: stratified `80/20`, seed `42`.
- Augmentation: random resized crops, rotation, affine translation/shear,
  color jitter, and random erasing.
- Horizontal flipping is intentionally disabled because mirroring can change
  the meaning of a hand sign.
- Optimizer: AdamW with learning rate `0.001` and weight decay `1e-4`.
- Loss: cross-entropy with label smoothing `0.1`.
- Scheduler: `ReduceLROnPlateau`.
- Device: CUDA when available, otherwise CPU.

Running the export cell writes the checkpoint to
`models/asl_cnn_model.pth`. Keep the checkpoint's `class_names` ordering in
sync with the model whenever replacing it.

## 🗂️ Project structure

```text
.
├── app.py                         # Streamlit inference application
├── main.ipynb                     # Dataset, training, and evaluation workflow
├── requirements.txt               # Python dependencies
├── models/
│   ├── asl_cnn_model.pth          # Trained CNN checkpoint
│   └── hand_landmarker.task       # MediaPipe hand model
├── Raw_Data/
│   ├── asl_alphabet_train/        # Training images by class
│   └── asl_alphabet_test/         # Held-out test images
└── data/
    ├── train_data.csv             # Training image paths and labels
    └── test_data.csv              # Test image paths and labels
```

## 🧪 Troubleshooting

### `Model not found`

Run the notebook's model export cell, or place a compatible checkpoint at
`models/asl_cnn_model.pth`. The checkpoint must contain:

```python
{
    "model_state_dict": ...,
    "class_names": [...],
}
```

### No hand detected

Use brighter, more even lighting; move the hand closer; simplify the
background; and keep the hand away from the face. The OpenCV fallback uses
skin-color segmentation, so extreme lighting or unusual backgrounds can reduce
its reliability.

### MediaPipe errors

Confirm that `mediapipe` is installed and that
`models/hand_landmarker.task` exists. The app is designed to continue with
OpenCV-only detection when MediaPipe is unavailable.

### CUDA or memory issues

The app automatically uses CPU when CUDA is unavailable. Training is much
more resource-intensive than inference; reduce the notebook batch size or
`num_workers` if the training environment runs out of memory.

## 🔒 Privacy

Images are processed by the local Streamlit process. This repository does not
include a hosted backend or automatic image upload service. Review your
deployment and Streamlit configuration before exposing the app publicly.

## 🤝 Contributing

Contributions are welcome. For changes to the model or preprocessing:

1. Update the notebook and explain the change.
2. Re-run validation and the held-out test evaluation.
3. Verify that the exported checkpoint loads in `app.py`.
4. Update this README when commands, assets, classes, or reported metrics
   change.

## 📄 License

No license file is currently included. Add a license before redistributing
this project or its dataset/model assets.
