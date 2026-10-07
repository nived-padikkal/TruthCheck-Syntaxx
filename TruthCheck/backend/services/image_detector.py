import os

import numpy as np
from PIL import Image

_model = None
_hf_pipe = None

HF_MODEL = "Organika/sdxl-detector"


def _load_local():
    global _model
    if _model is not None:
        return _model
    for path in ("models/cifake_model.pth", "models/cifake_model.h5"):
        if os.path.exists(path):
            if path.endswith(".pth"):
                import torch
                _model = torch.load(path, map_location="cpu", weights_only=False)
                _model.eval()
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


def analyze_image(path: str) -> dict:
    model = _load_local()

    if model is not None:
        img = Image.open(path).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        try:
            import torch
            is_torch = isinstance(model, torch.nn.Module)
        except Exception:
            is_torch = False
        if is_torch:
            import torch
            x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                out = model(x)
            out = out.detach().cpu().numpy().squeeze()
            fake_p = float(out[1]) if np.ndim(out) else float(out)
        else:
            out = np.asarray(model.predict(np.expand_dims(arr, 0), verbose=0)).squeeze()
            fake_p = float(out[1]) if np.ndim(out) else float(out)
        model_name = "CIFAKE"
    else:
        # Pretrained deepfake-vs-real ViT (honest fallback when CIFAKE weights absent)
        pipe = _get_hf()
        results = pipe(Image.open(path).convert("RGB"))
        fake_p = 0.0
        real_p = 0.0
        for r in results:
            lab = r["label"].lower()
            if any(k in lab for k in ("fake", "deep", "synthetic", "ai", "artificial", "generated")):
                fake_p = max(fake_p, float(r["score"]))
            else:
                real_p = max(real_p, float(r["score"]))
        if fake_p == 0.0 and real_p == 0.0:
            top = max(results, key=lambda r: r["score"])
            fake_p = float(top["score"]) if any(k in top["label"].lower() for k in ("fake", "deep", "ai", "artificial", "generated")) else 1 - float(top["score"])
        model_name = "CIFAKE(SDXL-AI-detector fallback)"

    fake_p = min(max(fake_p, 0.0), 1.0)
    real_p = 1.0 - fake_p
    prediction = "AI_GENERATED" if fake_p >= 0.8 else ("REAL" if real_p >= 0.8 else "SUSPICIOUS")
    return {
        "type": "image",
        "prediction": prediction,
        "confidence": round(max(fake_p, real_p), 3),
        "real_probability": round(real_p, 3),
        "fake_probability": round(fake_p, 3),
        "model": model_name,
        "risk": "HIGH" if fake_p >= 0.8 else ("MEDIUM" if fake_p >= 0.5 else "LOW"),
    }
