from flask import Blueprint, request, jsonify
from services.text_detector import analyze_text

text_bp = Blueprint("text", __name__)


@text_bp.post("/api/detect/text")
def detect_text():
    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text too long (max 20000 chars)"}), 400
    try:
        return jsonify(analyze_text(text))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Text analysis failed: {e}"}), 500
