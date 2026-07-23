#!/usr/bin/env python3
"""Run web demo for language detection.

Combines model export and demo server into single pipeline:
1. Export PyTorch model to ONNX (optional)
2. Start Flask server with web UI
3. Serve classification API

Usage:
    # Export only
    uv run src/scripts/run_demo.py --export-only

    # Run server (default)
    uv run src/scripts/run_demo.py

    # Custom port
    uv run src/scripts/run_demo.py --port 8080
"""

import argparse
import json
import logging
import sys
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from flask import Flask, Response, jsonify, request, send_from_directory

if TYPE_CHECKING:
    pass

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
OUTPUT_DIR = Path("web_demo/models")

DA_THRESHOLD = 0.5
EN_THRESHOLD = 0.5


def export_model() -> None:
    """Export PyTorch model to ONNX format.

    Exports FP32 model for web deployment.
    """
    if not MODEL_PATH.exists():
        LOGGER.error(f"Checkpoint not found: {MODEL_PATH}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading checkpoint from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    model = create_small_cnn(num_languages=2)
    model.load_state_dict(checkpoint)
    model.eval()

    output_path = OUTPUT_DIR / "model.onnx"
    dummy_input = torch.randn(1, 1, 80, 100)

    LOGGER.info(f"Exporting model to {output_path}")

    torch.onnx.export(
        model,
        (dummy_input,),  # Wrap in tuple for type checker
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "time_frames"},
            "output": {0: "batch_size"},
        },
    )

    # Validate ONNX model
    try:
        import onnx  # noqa: PLC0415 - runtime validation only

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
    except ImportError:
        LOGGER.warning("ONNX validation skipped (onnx not installed)")

    file_size = output_path.stat().st_size
    LOGGER.info(f"Exported {output_path.name}: {file_size / 1024:.1f} KB")

    config = {
        "models": [
            {
                "name": "Model",
                "file": "model.onnx",
                "disk_kb": round(file_size / 1024),
                "ram_kb": 886,
                "precision": "float32",
                "accuracy": "96.65%",
            }
        ],
        "input_shape": [1, 1, 80, None],
        "sample_rate": 16000,
        "n_mels": 80,
        "labels": ["Danish", "English"],
    }

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    LOGGER.info(f"Saved config to {config_path}")
    LOGGER.info("\n✅ Export complete!")


def classify(audio_path: str) -> dict:
    """Classify audio with multi-label output.

    Args:
        audio_path:
            Path to audio file.

    Returns:
        Dictionary with per-language probabilities and prediction.
    """
    waveform = load_and_preprocess(audio_path, target_sr=16000)
    log_mel = extract_log_mel_spectrogram(waveform, 16000, MelSpectrogramConfig())

    spec = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model = create_small_cnn(num_languages=2)
        model.load_state_dict(state_dict)
        model.eval()

        logits = model(spec)
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
        """Serve model files for ONNX inference.

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
        "--export-only",
        action="store_true",
        help="Export model to ONNX only, don't run server",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    if args.export_only:
        export_model()
    else:
        run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
