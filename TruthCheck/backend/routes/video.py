import os
import tempfile
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from services.video_detector import analyze_video

video_bp = Blueprint("video", __name__)
ALLOWED = {"mp4", "avi", "mov", "webm"}


@video_bp.post("/api/detect/video")
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    path = os.path.join(tempfile.gettempdir(), secure_filename(f.filename))
    try:
        f.save(path)
        return jsonify(analyze_video(path))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Video analysis failed: {e}"}), 500
    finally:
        _remove_quietly(path)


def _remove_quietly(path):
    """Best-effort temp cleanup.

    The detector may still hold the file briefly on Windows; failing to delete
    must not replace an otherwise valid response with a 500.
    """
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
