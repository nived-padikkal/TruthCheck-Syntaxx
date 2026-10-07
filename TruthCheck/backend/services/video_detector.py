import os

import cv2
import numpy as np

_model = None
_hf_pipe = None
_effnet = None
_effnet_tried = False
FRAME_SAMPLE_RATE = 5
MAX_FRAMES = 32
FACE_DETECTION_THRESHOLD = 0.9
HF_MODEL = "Organika/sdxl-detector"
HF_FALLBACK_NAME = "MesoNet(SDXL-AI-detector fallback)"

# The repo's own trained detector (same weights the Gradio app uses), located
# two directories above TruthCheck/backend.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_EFFNET_ONNX = os.path.join(_REPO_ROOT, "checkpoints", "efficientnet.onnx")
_EFFNET_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "model.pth")


def _load_local():
    global _model
    if _model is not None:
        return _model
    for path in ("models/mesonet.pth", "models/mesonet.h5"):
        if os.path.exists(path):
            if path.endswith(".pth"):
                import torch
                from models.mesonet_net import Meso4
                net = Meso4()
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                net.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
                net.eval()
                _model = net
            else:
                from tensorflow import keras
                _model = keras.models.load_model(path)
            return _model
    return None


def _get_hf():
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import pipeline
        _hf_pipe = pipeline("image-classification", model=HF_MODEL, device=-1)
    return _hf_pipe


def _fake_score_hf(pipe, face_rgb):
    from PIL import Image
    results = pipe(Image.fromarray(face_rgb))
    fake_p, real_p = 0.0, 0.0
    for r in results:
        lab = r["label"].lower()
        if any(k in lab for k in ("fake", "deep", "synthetic", "ai", "artificial", "generated")):
            fake_p = max(fake_p, float(r["score"]))
        else:
            real_p = max(real_p, float(r["score"]))
    if fake_p == 0.0 and real_p == 0.0:
        top = max(results, key=lambda r: r["score"])
        fake_p = float(top["score"]) if any(k in top["label"].lower() for k in ("fake", "deep", "ai", "artificial", "generated")) else 1 - float(top["score"])
    return fake_p


def _load_efficientnet():
    """Load the repo's face-crop EfficientNet (weights used by the Gradio app).

    Organika/sdxl-detector is an AI-*image* detector: applied to video face
    crops it systematically mislabels real footage as artificial and real
    deepfakes as human, so it is only used when these checkpoints are absent.
    """
    global _effnet, _effnet_tried
    if _effnet is not None or _effnet_tried:
        return _effnet
    _effnet_tried = True
    if not (os.path.exists(_EFFNET_ONNX) and os.path.exists(_EFFNET_CKPT)):
        return None
    try:
        import onnx
        import torch
        from onnx2pytorch import ConvertModel

        net = ConvertModel(onnx.load(_EFFNET_ONNX))
        try:
            ckpt = torch.load(_EFFNET_CKPT, map_location="cpu")
        except Exception:
            ckpt = torch.load(_EFFNET_CKPT, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["rgb_encoder"], strict=True)
        net.eval()
        _effnet = net
    except Exception:
        _effnet = None
    return _effnet


def _fake_score_effnet(face_rgb):
    """P(fake) in [0, 1] for one RGB face crop, or None if unavailable."""
    net = _load_efficientnet()
    if net is None:
        return None
    import torch
    face = cv2.resize(face_rgb, (224, 224)).astype(np.float32) / 255.0
    # NHWC on purpose: that is the layout the ONNX graph was exported with.
    x = torch.tensor(face, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = net(x)
    probs = out.detach().cpu().numpy().reshape(-1)
    return float(probs[1]) if probs.size > 1 else float(probs[0])


def analyze_video(path: str) -> dict:
    model = _load_local()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file (corrupted or unsupported).")

    try:
        return _score_video(cap, path, model)
    finally:
        # Always release: if we don't, the file stays locked on Windows and the
        # caller's cleanup raises PermissionError, turning a valid 200 into a 500.
        cap.release()


def _score_video(cap, path, model):
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, select_largest=True)
    except Exception:
        mtcnn = None

    # Pick the frame scorer: trained local MesoNet > repo EfficientNet >
    # SDXL image detector (last resort, biased on video frames).
    if model is not None:
        frame_scorer, model_name = None, "MesoNet"
    elif _load_efficientnet() is not None:
        frame_scorer, model_name = _fake_score_effnet, "EfficientNetV2(face-crop)"
    else:
        pipe = _get_hf()
        frame_scorer, model_name = (lambda f: _fake_score_hf(pipe, f)), HF_FALLBACK_NAME

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(range(0, max(n, 1), max(FRAME_SAMPLE_RATE, 1)))
    frames_done, fake_scores = 0, []
    fi = 0
    while cap.isOpened() and frames_done < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in idxs:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = rgb
            if mtcnn is not None:
                try:
                    boxes, probs = mtcnn.detect(rgb)
                    if boxes is not None and len(boxes) and (probs is None or probs[0] >= FACE_DETECTION_THRESHOLD):
                        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                        crop = rgb[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0:
                            face = crop
                except Exception:
                    pass

            if model is not None:
                face256 = cv2.resize(face, (256, 256)).astype(np.float32) / 255.0
                try:
                    import torch
                    is_torch = isinstance(model, torch.nn.Module)
                except Exception:
                    is_torch = False
                if is_torch:
                    import torch
                    x = torch.from_numpy(face256.transpose(2, 0, 1)).unsqueeze(0)
                    with torch.no_grad():
                        out = model(x)
                    score = float(out.detach().cpu().numpy().reshape(-1)[0])
                else:
                    x = np.expand_dims(face256, 0)
                    score = float(np.asarray(model.predict(x, verbose=0)).reshape(-1)[0])
            else:
                score = frame_scorer(face)
                if score is None:
                    score = _fake_score_hf(_get_hf(), face)
                    model_name = HF_FALLBACK_NAME

            fake_scores.append(min(max(score, 0.0), 1.0))
            frames_done += 1
        fi += 1

    if frames_done == 0:
        raise ValueError("No usable frames found in video.")

    # Aggregate evidence like inference_2.py: the mean is the default signal,
    # but a partially-spiked (partial) deepfake can average below 0.5 while
    # every top-quartile frame is strongly fake. The top quartile only counts
    # when at least two frames corroborate each other, so one stray frame can
    # no longer flip the verdict.
    scores = np.asarray(fake_scores, dtype=np.float64)
    suspicious = int((scores >= 0.5).sum())
    mean_fake = float(scores.mean())
    k = max(1, int(np.ceil(len(scores) * 0.25)))
    topk_fake = float(np.sort(scores)[-k:].mean())
    corroborated = int((scores > 0.6).sum()) >= 2
    fake_p = max(mean_fake, topk_fake) if corroborated else mean_fake
    real_p = 1.0 - fake_p
    prediction = "DEEPFAKE" if fake_p >= 0.8 else ("REAL" if real_p >= 0.8 else "SUSPICIOUS")
    return {
        "type": "video",
        "prediction": prediction,
        "confidence": round(max(fake_p, real_p), 3),
        "real_probability": round(real_p, 3),
        "fake_probability": round(fake_p, 3),
        "model": model_name,
        "frames_analyzed": frames_done,
        "suspicious_frames": suspicious,
        "risk": "HIGH" if fake_p >= 0.8 else ("MEDIUM" if fake_p >= 0.5 else "LOW"),
    }
