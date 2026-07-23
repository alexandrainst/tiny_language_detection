#!/usr/bin/env python3
"""Flask backend server for web demo.

Serves static files and provides /classify API endpoint with PyTorch inference.

Usage:
    uv run src/scripts/demo_server.py
    uv run src/scripts/demo_server.py --port 8080
"""

import argparse
import logging
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, Response, jsonify, request, send_from_directory

# Configure path before local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports (after sys.path manipulation)
from tiny_language_detection.data.preprocessing import load_and_preprocess  # noqa: E402
from tiny_language_detection.features.mel_spectrogram import (  # noqa: E402
    MelSpectrogramConfig,
    extract_log_mel_spectrogram,
)
from tiny_language_detection.models.tiny_cnn import create_small_cnn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

MODEL_PATH = Path("data/experiments/phase4b/tiny_cnn_multilabel_v2/model_best.pth")
WEB_DEMO_PATH = Path("web_demo")

DA_THRESHOLD = 0.5
EN_THRESHOLD = 0.5

# Load model at startup
model: nn.Module | None = None
mel_config = MelSpectrogramConfig()


def load_model() -> nn.Module:
    """Load classification model.

    Returns:
        Loaded PyTorch model in eval mode.
    """
    global model

    if model is not None:
        return model

    if not MODEL_PATH.exists():
        LOGGER.error(f"Checkpoint not found: {MODEL_PATH}")
        LOGGER.error("Run training first or set MODEL_PATH correctly")
        sys.exit(1)

    LOGGER.info(f"Loading model from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = create_small_cnn(num_languages=2)
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Model loaded successfully")

    return model


def classify(audio_path: str) -> dict:
    """Classify audio with multi-label output.

    Args:
        audio_path:
            Path to audio file.

    Returns:
        Dictionary with per-language probabilities and prediction.
    """
    loaded_model = load_model()

    waveform = load_and_preprocess(audio_path, target_sr=16000)
    log_mel = extract_log_mel_spectrogram(waveform, 16000, mel_config)

    spec = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = loaded_model(spec)
        probs = torch.sigmoid(logits)[0]

    da_prob = probs[0].item()
    en_prob = probs[1].item()

    if da_prob >= DA_THRESHOLD and da_prob >= en_prob:
        prediction = "Danish"
    elif en_prob >= EN_THRESHOLD and en_prob >= da_prob:
        prediction = "English"
    else:
        prediction = "Unknown"

    confidence = max(da_prob, en_prob)

    return {
        "danish": da_prob * 100,
        "english": en_prob * 100,
        "prediction": prediction,
        "confidence": confidence * 100,
    }


def create_app() -> Flask:
    """Create Flask application.

    Returns:
        Flask application instance.
    """
    app = Flask(__name__, static_folder=str(WEB_DEMO_PATH), static_url_path="")

    @app.route("/")
    def index() -> Response:
        """Serve main HTML page.

        Returns:
            HTML content for index page.
        """
        assert app.static_folder is not None
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/models/<path:filename>")
    def models(filename: str) -> Response:
        """Serve model files.

        Args:
            filename:
                Model filename.

        Returns:
            Model file content.
        """
        return send_from_directory(WEB_DEMO_PATH / "models", filename)

    @app.route("/<path:filename>")
    def static_files(filename: str) -> Response:
        """Serve static files (CSS, JS).

        Args:
            filename:
                Static file filename.

        Returns:
            File content.
        """
        assert app.static_folder is not None
        return send_from_directory(app.static_folder, filename)

    @app.route("/classify", methods=["POST"])
    def api_classify() -> tuple:
        """Classify uploaded audio file.

        Returns:
            JSON response with classification results or error.
        """
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        audio_file = request.files["audio"]
        audio_path = f"{tempfile.gettempdir()}/{audio_file.filename}"
        audio_file.save(audio_path)

        try:
            result = classify(audio_path)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    return app


def run_server(port: int = 7860, host: str = "127.0.0.1") -> None:
    """Start Flask demo server.

    Args:
        port:
            Server port.
        host:
            Server host.
    """
    # Pre-load model
    load_model()

    app = create_app()

    LOGGER.info("\n🚀 Starting demo server...")
    LOGGER.info(f"Model: {MODEL_PATH}")
    LOGGER.info(f"Serving files from: {WEB_DEMO_PATH}")
    LOGGER.info(f"Open: http://{host}:{port}")
    LOGGER.info("\n✅ Multi-label: Danish | English | Unknown\n")

    app.run(host=host, port=port, debug=False)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run language detection web demo")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
