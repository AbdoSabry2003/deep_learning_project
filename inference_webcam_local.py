# inference_webcam_local.py
# Run: python inference_webcam_local.py
# Edit settings below (Config) — no CLI args needed.

from dataclasses import dataclass
import sys, time, platform
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

# Optional: MediaPipe for hand detection (ROI)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# =======================
# Config (edit freely)
# =======================
@dataclass
class Config:
    WEIGHTS: str = "best_vit_gesture.pth"   # path to trained weights
    CAMERA: int = 0
    WIDTH: int = 640
    HEIGHT: int = 480
    MIRROR: bool = True                     # flip webcam image horizontally

    # Match training
    GRAYSCALE: bool = True                  # True if you trained with grayscale
    NORM_BG: str = "otsu"                   # "auto" | "ycrcb" | "otsu" | "none"

    # Detection & ROI
    USE_MEDIAPIPE: bool = True              # enable hand detector
    REQUIRE_DETECTION: bool = True          # classify ONLY when a hand is detected
    MARGIN: float = 1.6                     # ROI expansion around detected hand
    MIN_AREA_RATIO: float = 0.06            # min ROI area relative to frame (0.06 ~ 6%)
    MIN_SKIN_RATIO: float = 0.12            # set 0.0 to disable (only if NORM_BG!='none')
    FIX_LEFT_TO_RIGHT: bool = True          # flip left hand to match right‑hand dataset

    # Decision & smoothing
    UNKNOWN_THRESH: float = 0.1            # min confidence to accept top-1
    MIN_GAP: float = 0.10                   # min (top1 - top2) gap
    SMOOTH_WINDOW: int = 7                  # EMA window; 0 = off
    SHOW_TOPK: bool = True
    TOPK: int = 3

    # Drawing
    DRAW_ROI: bool = True
    SHOW_ROI: bool = False

    # Camera backend (Windows)
    FORCE_CAP_DSHOW: bool = True            # use DirectShow on Windows to avoid camera issues

CFG = Config()

GESTURE_NAMES = ['palm','l','fist','fist_moved','thumb','index','ok','palm_moved','c','down']

# =======================
# Utils
# =======================
def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(weights_path, num_classes=10, device=torch.device("cpu")):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    # safe load (PyTorch >= 2.1 supports weights_only=True)
    try:
        sd = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def make_transforms(use_grayscale=True):
    mean = [0.485,0.456,0.406]; std = [0.229,0.224,0.225]
    prepend = [transforms.Grayscale(num_output_channels=3)] if use_grayscale else []
    return transforms.Compose(prepend + [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def mediapipe_hands():
    if not MP_AVAILABLE:
        return None
    return mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

def crop_with_mediapipe(frame_bgr, mp_hands, margin=1.5):
    """Return (crop_bgr, handedness_str, bbox, det_ok)"""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None, None, False

    handed = None
    if res.multi_handedness:
        handed = res.multi_handedness[0].classification[0].label  # 'Left' | 'Right'

    xs, ys = [], []
    for lm in res.multi_hand_landmarks[0].landmark:
        xs.append(lm.x * w); ys.append(lm.y * h)
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    cx, cy = (x0+x1)//2, (y0+y1)//2
    side = int(max(x1-x0, y1-y0) * margin)
    x0, y0 = cx - side//2, cy - side//2
    x1, y1 = x0 + side, y0 + side
    x0, y0 = max(0,x0), max(0,y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1-x0 < 20 or y1-y0 < 20:
        return None, None, None, False
    crop = frame_bgr[y0:y1, x0:x1]
    return crop, handed, (x0,y0,x1,y1), True

def normalize_background_black(roi_bgr, use_grayscale=True, method="auto", return_mask=False):
    if method == "none":
        return (roi_bgr, None) if return_mask else roi_bgr
    if method == "auto":
        method = "otsu" if use_grayscale else "ycrcb"

    if method == "ycrcb":
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0,133,77], dtype=np.uint8)
        upper = np.array([255,173,127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
    else:  # otsu
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    canvas = np.zeros_like(roi_bgr)
    fg = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask)
    inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(canvas, canvas, mask=inv)
    out = cv2.add(fg, bg)
    return (out, mask) if return_mask else out

def overlay_header(frame, text, sub=None, color=(0,255,0)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w-10, 110 if sub else 80), (0,0,0), -1)
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

# =======================
# Main
# =======================
def main():
    device = get_device()
    print("Device:", device)

    if CFG.REQUIRE_DETECTION and (not CFG.USE_MEDIAPIPE):
        print("REQUIRE_DETECTION=True but USE_MEDIAPIPE=False — enable detector or set REQUIRE_DETECTION=False.")
        sys.exit(1)
    if CFG.USE_MEDIAPIPE and not MP_AVAILABLE:
        print("mediapipe not installed. pip install mediapipe, or set USE_MEDIAPIPE=False.")
        if CFG.REQUIRE_DETECTION:
            sys.exit(1)

    model = build_model(CFG.WEIGHTS, num_classes=len(GESTURE_NAMES), device=device)
    preprocess = make_transforms(use_grayscale=CFG.GRAYSCALE)
    mp_hands = mediapipe_hands() if (CFG.USE_MEDIAPIPE and MP_AVAILABLE) else None

    # Open camera
    cap = None
    if platform.system() == "Windows" and CFG.FORCE_CAP_DSHOW:
        cap = cv2.VideoCapture(CFG.CAMERA, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CFG.CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.HEIGHT)
    if not cap.isOpened():
        print("Could not open webcam. Try CAMERA=1 or 2 in Config.")
        sys.exit(1)

    ac = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    torch.set_grad_enabled(False)

    prob_avg = None
    alpha = 2.0 / (max(CFG.SMOOTH_WINDOW,1) + 1.0)
    fps = 0.0; t_last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break
        if CFG.MIRROR: frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # Detect ROI
        crop, handed, bbox, det_ok = (None, None, None, False)
        if mp_hands is not None:
            crop, handed, bbox, det_ok = crop_with_mediapipe(frame, mp_hands, margin=CFG.MARGIN)

        # Require detection?
        if CFG.REQUIRE_DETECTION and not det_ok:
            overlay_header(frame, "No hand", sub=f"{fps:.1f} FPS", color=(0,255,255))
            prob_avg = None
            cv2.imshow("Gesture (q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            continue

        # Fallback if allowed and not detected
        if not det_ok:
            side = min(H, W); y0 = (H - side)//2; x0 = (W - side)//2
            bbox = (x0, y0, x0+side, y0+side)
            crop = frame[y0:y0+side, x0:x0+side]
            handed = None

        # Flip left→right to match dataset
        if CFG.FIX_LEFT_TO_RIGHT and handed == "Left":
            crop = cv2.flip(crop, 1)

        # Min area gate
        if bbox is not None:
            x0,y0,x1,y1 = bbox
            area_ratio = ((x1-x0)*(y1-y0)) / float(H*W)
            if area_ratio < CFG.MIN_AREA_RATIO:
                overlay_header(frame, "No hand (too small)", sub=f"{fps:.1f} FPS", color=(0,255,255))
                prob_avg = None
                cv2.imshow("Gesture (q to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

        # Background normalization + skin ratio gate
        crop_norm, mask = normalize_background_black(
            crop, use_grayscale=CFG.GRAYSCALE, method=CFG.NORM_BG, return_mask=True
        )
        if CFG.MIN_SKIN_RATIO > 0 and mask is not None:
            skin_ratio = (mask > 0).mean()
            if skin_ratio < CFG.MIN_SKIN_RATIO:
                overlay_header(frame, "No hand (low skin)", sub=f"{fps:.1f} FPS", color=(0,255,255))
                prob_avg = None
                cv2.imshow("Gesture (q to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

        # Preprocess + inference
        rgb = cv2.cvtColor(crop_norm, cv2.COLOR_BGR2RGB)
        x = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with ac:
            probs = torch.softmax(model(x), dim=1).detach().cpu().numpy()[0]

        # EMA smoothing
        if CFG.SMOOTH_WINDOW > 0:
            prob_avg = probs if prob_avg is None else (1 - alpha)*prob_avg + alpha*probs
            p = prob_avg
        else:
            p = probs

        order = np.argsort(p)[::-1]
        top1, top2 = p[order[0]], (p[order[1]] if len(order) > 1 else 0.0)
        gap = top1 - top2
        pred = GESTURE_NAMES[int(order[0])]
        is_unknown = (top1 < CFG.UNKNOWN_THRESH) or (gap < CFG.MIN_GAP)
        label = "Unknown" if is_unknown else pred

        # FPS
        now = time.time(); fps = 0.9*fps + 0.1*(1.0/max(now - t_last, 1e-9)); t_last = now

        # Overlay
        sub = f"{(handed or 'N/A')} | {fps:.1f} FPS | p1={top1:.2f} gap={gap:.2f}"
        overlay_header(frame, label, sub=sub, color=(0,255,0) if not is_unknown else (0,255,255))
        if CFG.DRAW_ROI and bbox:
            x0,y0,x1,y1 = bbox
            color = (0,255,0) if det_ok else (0,255,255)
            cv2.rectangle(frame, (x0,y0), (x1,y1), color, 2)

        # Show Top‑K
        if CFG.SHOW_TOPK:
            ytxt = 130
            for i, idx in enumerate(order[:CFG.TOPK]):
                txt = f"{GESTURE_NAMES[int(idx)]}: {p[idx]*100:.1f}%"
                cv2.putText(frame, txt, (20, ytxt + 26*i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if CFG.SHOW_ROI:
            cv2.imshow("ROI", crop_norm)

        cv2.imshow("Gesture (q to quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =======================
if __name__ == "__main__":
    main()