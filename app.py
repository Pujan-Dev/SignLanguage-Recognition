# Streamlit app for the ASL sign-language CNN trained in main.ipynb
import os

# Keep MediaPipe's C++ logging quiet in the Streamlit terminal.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_cnn_model.pth")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")


# Same architecture as in main.ipynb (cell 18)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Same deterministic preprocessing used for validation/test in the notebook
eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@st.cache_resource
def load_face_cascade():
    """Load OpenCV's bundled frontal-face detector."""
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


@st.cache_resource
def load_hand_landmarker():
    """Load the MediaPipe hand model used to locate the hand.

    OpenCV still does all image conversion, cropping, drawing and the skin-mask
    fallback. MediaPipe is optional; if it is unavailable the app uses OpenCV
    skin segmentation only.
    """
    if not MEDIAPIPE_AVAILABLE or not os.path.exists(HAND_MODEL_PATH):
        return None

    try:
        base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.05,
            min_hand_presence_confidence=0.05,
            min_tracking_confidence=0.05,
        )
        return vision.HandLandmarker.create_from_options(options)
    except Exception:
        return None


def detect_faces(image_bgr, face_cascade):
    """Return face boxes so they can be excluded from the hand search."""
    if face_cascade is None:
        return ()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )


def remove_face_regions(mask, faces):
    """Blank out detected faces so only hand skin remains in the mask."""
    for (x, y, w, h) in faces:
        # Remove a slightly larger box to include the chin/neck area.
        pad_x = int(0.15 * w)
        pad_y = int(0.15 * h)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(mask.shape[1], x + w + pad_x)
        y1 = min(mask.shape[0], y + h + pad_y)
        mask[y0:y1, x0:x1] = 0
    return mask


def box_overlaps_faces(box, faces, threshold=0.5):
    """Reject a hand box if it mostly overlaps a detected face."""
    x0, y0, x1, y1 = box
    box_area = max((x1 - x0) * (y1 - y0), 1)

    for (fx, fy, fw, fh) in faces:
        ix0 = max(x0, fx)
        iy0 = max(y0, fy)
        ix1 = min(x1, fx + fw)
        iy1 = min(y1, fy + fh)
        intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if intersection / box_area > threshold:
            return True

    return False


def detect_hand_with_mediapipe(image, landmarker):
    """Use MediaPipe hand landmarks to get a tight hand box and mask."""
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    empty_mask = np.zeros((height, width), dtype=np.uint8)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, empty_mask

    # If more than one hand is visible, keep the largest one.
    best_hull = None
    best_area = 0.0

    for hand in result.hand_landmarks:
        points = np.array(
            [
                [
                    int(np.clip(point.x, 0.0, 1.0) * width),
                    int(np.clip(point.y, 0.0, 1.0) * height),
                ]
                for point in hand
            ],
            dtype=np.int32,
        )
        hull = cv2.convexHull(points)
        area = cv2.contourArea(hull)
        if area > best_area:
            best_area = area
            best_hull = hull

    if best_hull is None:
        return None, empty_mask

    x, y, w, h = cv2.boundingRect(best_hull)
    pad = int(0.20 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, best_hull, 255)

    return (x0, y0, x1, y1), mask


def detect_hand(image, face_cascade, landmarker=None):
    """Find the hand and return (bounding_box, mask, detected_faces).

    MediaPipe landmarks are tried first because they ignore the face and work
    with different skin tones. If that fails, OpenCV skin segmentation is used.
    """
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    faces = detect_faces(image_bgr, face_cascade)

    # Preferred path: a real hand detector, not skin-colour guessing.
    if landmarker is not None:
        hand_box, hand_mask = detect_hand_with_mediapipe(image, landmarker)
        if hand_box is not None and not box_overlaps_faces(hand_box, faces):
            box_area = (hand_box[2] - hand_box[0]) * (hand_box[3] - hand_box[1])
            # Ignore tiny detections; they are usually fingers/background noise.
            if box_area >= 0.03 * width * height:
                return hand_box, hand_mask, faces

    # Fallback path: OpenCV skin segmentation, with faces removed first.
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin_hsv = cv2.inRange(
        hsv,
        np.array([0, 20, 20], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8),
    )

    mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    mask = remove_face_regions(mask, faces)

    # Clean up noise and connect gaps inside the hand.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.005 * width * height
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not candidates:
        return None, mask, faces

    hand_contour = max(candidates, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(hand_contour)

    # Add padding so the whole hand is included in the crop.
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)

    # Reject a "hand" crop that is actually the face/neck region.
    if box_overlaps_faces((x0, y0, x1, y1), faces, threshold=0.1):
        return None, mask, faces

    # Also reject tiny skin blobs that cannot be a hand.
    if (x1 - x0) * (y1 - y0) < 0.03 * width * height:
        return None, mask, faces

    return (x0, y0, x1, y1), mask, faces


def annotate_hand(image, box):
    """Draw the detected hand box on a copy of the image."""
    annotated = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            "HAND",
            (x0, max(y0 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    class_names = checkpoint["class_names"]
    model = CNNModel(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, device


def predict(image, model, class_names, device):
    image = image.convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()

    top_probs, top_idx = probs.topk(min(5, len(class_names)))
    return [
        (class_names[idx.item()], float(prob))
        for prob, idx in zip(top_probs, top_idx)
    ]


st.set_page_config(page_title="ASL Sign Language Recognition", page_icon="🤟")

st.title("🤟 ASL Sign Language Recognition")
st.write(
    "Upload a photo of an American Sign Language hand sign (or use your camera). "
    "OpenCV and MediaPipe locate and crop only the hand, excluding the face and "
    "background, then the CNN trained in `main.ipynb` predicts the sign."
)

try:
    model, class_names, device = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Run the training notebook first to create it."
    )
    st.stop()

face_cascade = load_face_cascade()
hand_landmarker = load_hand_landmarker()

st.sidebar.header("Model info")
st.sidebar.write(f"**Classes:** {len(class_names)}")
st.sidebar.write(f"**Device:** `{device}`")
st.sidebar.write(
    "**Hand detector:** "
    + ("MediaPipe + OpenCV" if hand_landmarker is not None else "OpenCV skin mask")
)
st.sidebar.caption("Classes: " + ", ".join(class_names))

input_method = st.radio(
    "Choose an input method:",
    ["Upload image", "Use camera"],
    horizontal=True,
)

image = None
if input_method == "Upload image":
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is not None:
        image = Image.open(uploaded)
else:
    camera_photo = st.camera_input("Take a picture of your hand sign")
    if camera_photo is not None:
        image = Image.open(camera_photo)

if image is not None:
    hand_box, hand_mask, faces = detect_hand(image, face_cascade, hand_landmarker)
    annotated = annotate_hand(image, hand_box)

    left, right = st.columns([1, 1])

    with left:
        st.image(
            annotated,
            caption="OpenCV hand detection (green box = region used by the model)",
            use_container_width=True,
        )
        if len(faces) > 0:
            st.caption(f"Removed {len(faces)} detected face region(s) from the hand search.")

    if hand_box is None:
        with right:
            st.warning(
                "No hand detected. Make sure your hand is clearly visible, "
                "well lit, and not covered."
            )
    else:
        hand_image = image.crop(hand_box)
        predictions = predict(hand_image, model, class_names, device)
        label, confidence = predictions[0]

        with right:
            st.image(
                hand_image,
                caption="Cropped hand passed to the CNN",
                use_container_width=True,
            )
            st.subheader("Prediction")
            st.metric("Sign", label)
            st.metric("Confidence", f"{confidence:.1%}")

            st.subheader("Top 5 probabilities")
            st.bar_chart({name: prob for name, prob in predictions})
            st.dataframe(
                [
                    {"Sign": name, "Probability": f"{prob:.2%}"}
                    for name, prob in predictions
                ],
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("Show hand-detection mask"):
        st.image(
            hand_mask,
            caption="White = the hand region used for cropping.",
            use_container_width=True,
        )
else:
    st.info("Upload an image or take a photo to get a prediction.")
