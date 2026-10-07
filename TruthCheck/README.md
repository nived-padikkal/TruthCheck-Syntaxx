# TruthCheck

AI-powered Chrome extension that detects whether **text**, **images**, and **videos** are likely REAL or FAKE/AI-generated.

## Structure

```
TruthCheck/
├── extension/   # Chrome MV3 extension (popup dashboard)
└── backend/     # Flask API + ISOT / CIFAKE / MesoNet models
```

## Backend setup

```bash
cd TruthCheck/backend
pip install -r requirements.txt
python app.py
```

API runs on `http://127.0.0.1:5000`:
- `POST /api/detect/text` (form field `text`)
- `POST /api/detect/image` (file `file`)
- `POST /api/detect/video` (file `file`)
- `GET /api/health`, `GET /api/model-status`

## Model placement (REQUIRED — real models only, no fake outputs)

Place trained files in `TruthCheck/backend/models/`:

| File | Purpose |
|---|---|
| `isot_text_model.pkl` | ISOT fake-news text classifier (sklearn pipeline with `predict_proba`) |
| `cifake_model.pth` or `cifake_model.h5` | CIFAKE real-vs-AI image classifier |
| `mesonet.pth` or `mesonet.h5` | MesoNet deepfake video frame classifier |

You can train MesoNet with the repo's `train_mesonet.py` using CASIA v2.0 / DFDC / Celeb-DF and copy the resulting checkpoint here as `mesonet.pth`.

If a model file is missing, the API returns a clear `503` "model not found" error — it will **not** fake a prediction.

## Extension setup

1. Chrome → `chrome://extensions` → Developer Mode → **Load unpacked** → select `TruthCheck/extension/`
2. Keep the backend running, then open the TruthCheck popup.

## Thresholds

- `>= 0.80` → strong REAL/FAKE prediction
- `0.60–0.79` → SUSPICIOUS
- `< 0.60` → treated as SUSPICIOUS/unable to determine

Confidence is always reported as a model probability; nothing here claims 100% accuracy.
