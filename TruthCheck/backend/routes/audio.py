import os
import tempfile
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from services.audio_detector import analyze_audio

audio_bp = Blueprint("audio", __name__)
ALLOWED = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "opus", "wma"}


@audio_bp.post("/api/detect/audio")
def detect_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    path = os.path.join(tempfile.gettempdir(), secure_filename(f.filename) or f"audio.{ext}")
    try:
        f.save(path)
        return jsonify(analyze_audio(path))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Audio analysis failed: {e}"}), 500
    finally:
        _remove_quietly(path)


def _remove_quietly(path):
    """Best-effort temp cleanup; a locked file must not cause a 500."""
    import time
    for attempt in range(3):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except OSError:
            if attempt == 2:
                return
            time.sleep(0.2)
