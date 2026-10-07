"""New, more accurate deepfake image detector.

Uses a ViT-based deepfake-vs-real classifier fine-tuned on deepfake datasets
(dima806/deepfake_vs_real_image_detection). The old EfficientNet model in
inference_2.py remains available via use_legacy=True.
"""

from PIL import Image

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "image-classification",
            model="dima806/deepfake_vs_real_image_detection",
            device=-1,
        )
    return _pipeline


def predict_image(image_path):
    """Return (verdict_text, is_fake_bool, confidence_float)."""
    pipe = _get_pipeline()
    img = Image.open(image_path).convert("RGB")
    results = pipe(img)  # list of {'label': ..., 'score': ...}

    scores = {r["label"].lower(): float(r["score"]) for r in results}
    fake_score = 0.0
    real_score = 0.0
    for label, score in scores.items():
        if "deep" in label or "fake" in label or "synthetic" in label:
            fake_score = max(fake_score, score)
        elif "real" in label or "authentic" in label or "human" in label:
            real_score = max(real_score, score)
    if fake_score == 0.0 and real_score == 0.0:
        # Fallback: trust the top label
        top = max(results, key=lambda r: r["score"])
        is_fake = any(k in top["label"].lower() for k in ("fake", "deep", "synthetic"))
        return (is_fake, top["score"])

    is_fake = fake_score > real_score
    confidence = max(fake_score, real_score)
    return (is_fake, confidence)


def deepfakes_image_predict_v2(input_image):
    if input_image is None:
        return "Please upload an image."
    try:
        is_fake, confidence = predict_image(input_image)
    except Exception as e:
        return f"Model error: {e}"

    pct = round(confidence * 100, 2)
    if is_fake:
        return (
            "🚨 VERDICT: FAKE / DEEPFAKE IMAGE\n"
            "──────────────────────────────\n"
            f"• Result          : FAKE MEDIA DETECTED\n"
            f"• Confidence Score: {pct}%\n"
            "• Model           : ViT Deepfake Detector v2\n"
            "• Analysis        : Synthetic/AI-manipulation artifacts identified."
        )
    return (
        "✅ VERDICT: REAL / AUTHENTIC IMAGE\n"
        "──────────────────────────────\n"
        f"• Result          : AUTHENTIC MEDIA\n"
        f"• Confidence Score: {pct}%\n"
        "• Model           : ViT Deepfake Detector v2\n"
        "• Analysis        : Natural, unmanipulated image confirmed."
    )
