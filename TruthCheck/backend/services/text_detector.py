import os
import pickle
import re

_model = None
_hf_pipe = None

HF_MODEL = "jy46604790/Fake-News-Bert-Detect"


def _load_local():
    global _model
    if _model is not None:
        return _model
    path = "models/isot_text_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            _model = pickle.load(f)
        return _model
    return None


def _get_hf():
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import pipeline
        _hf_pipe = pipeline("text-classification", model=HF_MODEL, device=-1)
    return _hf_pipe


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze_text(text: str) -> dict:
    model = _load_local()
    cleaned = _clean(text)

    if model is not None:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned])[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            fake_idx = classes.index(1) if 1 in classes else int(proba.argmax())
            fake_p = float(proba[fake_idx])
        else:
            pred = model.predict([cleaned])[0]
            fake_p = 1.0 if str(pred).lower() in ("1", "fake") else 0.0
        model_name = "ISOT"
    else:
        pipe = _get_hf()
        r = pipe(text[:512])[0]
        lab = r["label"].lower()
        score = float(r["score"])
        # Polarity verified against jy46604790/Fake-News-Bert-Detect:
        #   LABEL_0 -> fake, LABEL_1 -> real (its config only exposes opaque
        #   LABEL_0/LABEL_1, so guessing by name is not safe).
        if "real" in lab or "label_1" in lab:
            fake_p = 1.0 - score
        elif "fake" in lab or "label_0" in lab:
            fake_p = score
        else:
            fake_p = 1.0 - score
        model_name = "ISOT(BERT-fake-news fallback)"

    real_p = 1.0 - fake_p
    prediction = "FAKE" if fake_p >= 0.8 else ("REAL" if real_p >= 0.8 else "SUSPICIOUS")
    return {
        "type": "text",
        "prediction": prediction,
        "confidence": round(max(fake_p, real_p), 3),
        "real_probability": round(real_p, 3),
        "fake_probability": round(fake_p, 3),
        "model": model_name,
        "risk": "HIGH" if fake_p >= 0.8 else ("MEDIUM" if fake_p >= 0.5 else "LOW"),
    }
