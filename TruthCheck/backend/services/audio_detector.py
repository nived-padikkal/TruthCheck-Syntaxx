"""Audio deepfake detection using the parent project's RawNet model
(AI-Generated-Video-Detector repo: checkpoints/model.pth -> 'spec_encoder').

The parent repo also has a package named ``models`` which collides with this
backend's own ``models`` package, so we import it with a temporary module
swap and then restore the backend's modules.
"""

import os
import sys
import types

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

AUDIO_ARGS = {
    "nb_samp": 64600,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
}

_model = None
_load_error = None


def _load_spec_model():
    """Load the parent repo's RawNet spec encoder, isolating its ``models``
    package from this backend's ``models`` package."""
    global _model, _load_error
    if _model is not None:
        return _model
    if _load_error is not None:
        raise RuntimeError(f"Audio model unavailable: {_load_error}")

    import torch

    # Detach this backend's 'models' package while the parent's is imported.
    # NOTE: a regular package (this backend's models/, has __init__.py) always
    # beats the parent's namespace package, so the backend dir must also leave
    # sys.path temporarily.
    saved = {}
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            saved[key] = sys.modules.pop(key)
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != backend_dir]
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    try:
        from models.image import RawNet

        args = types.SimpleNamespace(
            device="cpu",
            pretrained_audio_encoder=False,
            freeze_audio_encoder=False,
            **AUDIO_ARGS,
        )
        model = RawNet(args)
        ckpt_path = os.path.join(PARENT, "checkpoints", "model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing audio checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["spec_encoder"], strict=True)
        model.eval()
    except Exception as e:  # noqa: BLE001
        _load_error = str(e)
        raise RuntimeError(f"Audio model unavailable: {e}") from e
    finally:
        # Remove the parent's 'models.*' modules and restore the backend's.
        for key in list(sys.modules):
            if (key == "models" or key.startswith("models.")) and key not in saved:
                sys.modules.pop(key)
        sys.modules.update(saved)
        sys.path[:] = saved_path

    _model = model
    return _model


def _load_audio(path):
    import numpy as np

    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        pass
    # Fallback: librosa/audioread (mp3, m4a, ogg, ...)
    import librosa

    data, sr = librosa.load(path, sr=16000, mono=True)
    return np.asarray(data, dtype=np.float32), int(sr)


def _preprocess(audio, sr):
    """Replicates inference_2.preprocess_audio -> (1, 64600) float tensor."""
    import numpy as np
    import torch

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        audio = np.zeros(AUDIO_ARGS["nb_samp"], dtype=np.float32)
    if float(audio.max()) > 1.0:
        audio = audio / 32768.0
    if sr and sr != 16000 and sr > 0:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception:
            pass
    n = AUDIO_ARGS["nb_samp"]
    if audio.shape[0] < n:
        audio = np.pad(audio, (0, n - audio.shape[0]))
    else:
        audio = audio[:n]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def analyze_audio(path: str) -> dict:
    import numpy as np
    import torch

    model = _load_spec_model()
    audio, sr = _load_audio(path)
    tensor = _preprocess(audio, sr)
    with torch.no_grad():
        out = model.forward(tensor)
    # RawNet output is log-softmax -> exponentiate to get probabilities.
    probs = np.exp(out.cpu().detach().numpy().squeeze())
    real_p = float(probs[0])
    fake_p = float(probs[1])

    prediction = "FAKE" if fake_p >= 0.8 else ("REAL" if real_p >= 0.8 else "SUSPICIOUS")
    return {
        "type": "audio",
        "prediction": prediction,
        "confidence": round(max(real_p, fake_p), 3),
        "real_probability": round(real_p, 3),
        "fake_probability": round(fake_p, 3),
        "model": "RawNet (AI-Generated-Video-Detector)",
        "risk": "HIGH" if fake_p >= 0.8 else ("MEDIUM" if fake_p >= 0.5 else "LOW"),
    }
