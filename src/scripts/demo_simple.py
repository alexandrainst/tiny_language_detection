#!/usr/bin/env python3
"""Simple HTTP demo for language detection with multi-label output."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
WEB_DEMO_PATH = PROJECT_ROOT / "web_demo"
MODEL_PATH = PROJECT_ROOT / "data/experiments/phase4b/tiny_cnn_multilabel_v2/model_best.pth"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from tiny_language_detection.features.mel_spectrogram import MelSpectrogramConfig, extract_log_mel_spectrogram
from tiny_language_detection.data.preprocessing import load_and_preprocess
from tiny_language_detection.models.tiny_cnn import create_small_cnn

# Load multi-label model
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = create_small_cnn(num_languages=2)
model.load_state_dict(state_dict)
model.eval()

mel_config = MelSpectrogramConfig()

# Thresholds for classification
DA_THRESHOLD = 0.5
EN_THRESHOLD = 0.5

def classify(audio_path: str) -> dict:
    """Classify audio with multi-label output.
    
    Returns independent probabilities for each language.
    Prediction is the language with highest probability if above threshold.
    """
    waveform = load_and_preprocess(audio_path, target_sr=16000)
    log_mel = extract_log_mel_spectrogram(waveform, 16000, mel_config)
    
    # Shape: [batch=1, channel=1, freq, time]
    spec = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(spec)
        probs = torch.sigmoid(logits)[0]
    
    da_prob = probs[0].item()
    en_prob = probs[1].item()
    
    # Determine prediction based on highest probability
    if da_prob >= DA_THRESHOLD and da_prob >= en_prob:
        prediction = "Danish"
    elif en_prob >= EN_THRESHOLD and en_prob >= da_prob:
        prediction = "English"
    else:
        prediction = "Unknown"
    
    # Confidence is the max probability
    confidence = max(da_prob, en_prob)
    
    return {
        "danish": da_prob * 100,
        "english": en_prob * 100,
        "prediction": prediction,
        "confidence": confidence * 100,
    }


from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=str(WEB_DEMO_PATH), static_url_path='')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/models/<path:filename>')
def models(filename):
    return send_from_directory(WEB_DEMO_PATH / 'models', filename)

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/classify', methods=['POST'])
def api_classify():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    audio_path = f"/tmp/{audio_file.filename}"
    audio_file.save(audio_path)
    
    try:
        result = classify(audio_path)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("\n🚀 Starting demo server...")
    print(f"Model: {MODEL_PATH}")
    print(f"Serving files from: {WEB_DEMO_PATH}")
    print("Open: http://localhost:7860")
    print("\n✅ Multi-label: Danish | English | Unknown\n")
    app.run(host='127.0.0.1', port=7860, debug=False)
