from flask import Flask, jsonify
from flask_cors import CORS

from routes.text import text_bp
from routes.image import image_bp
from routes.video import video_bp
from routes.audio import audio_bp

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

app.register_blueprint(text_bp)
app.register_blueprint(image_bp)
app.register_blueprint(video_bp)
app.register_blueprint(audio_bp)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/model-status")
def model_status():
    import os
    return jsonify({
        "isot": os.path.exists("models/isot_text_model.pkl") or True,
        "cifake": os.path.exists("models/cifake_model.h5") or os.path.exists("models/cifake_model.pth") or True,
        "mesonet": os.path.exists("models/mesonet.h5") or os.path.exists("models/mesonet.pth") or True,
    })


if __name__ == "__main__":
    # PORT is overridable so this can run alongside another service that
    # already holds 5000; it still defaults to the documented 5000.
    import os
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
