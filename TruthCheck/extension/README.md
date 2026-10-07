# TruthCheck Extension

MV3 Chrome extension popup for TEXT / IMAGE / VIDEO detection.

## Load
1. `chrome://extensions` → Developer mode → Load unpacked → this folder.
2. Start backend: `cd ../backend && python app.py`.

The popup reads model readiness from `GET /api/model-status` and posts files to `POST /api/detect/*`. Results, confidence bars, risk levels, and a 5-item history (chrome.storage.local) are shown in the dashboard.
