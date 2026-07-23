#!/usr/bin/env python3
"""
Language Detection Demo with Gradio

Run: python3 src/scripts/demo.py
Requires: pip install gradio torchaudio torch
"""

import sys
from pathlib import Path

# Use the project's virtual environment if available
venv_path = Path(__file__).parent.parent.parent / ".venv"
if venv_path.exists():
    import site
    site.addsitedir(str(venv_path / "lib" / "python3.12" / "site-packages"))

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid __init__.py issues
import torchaudio
import torchaudio.transforms as T
from tiny_language_detection.models.tiny_cnn import create_small_cnn

MODEL_PATH = Path("data/experiments/phase4b/tiny_cnn_kd/model_best.pth")

# Load model
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = create_small_cnn(num_languages=2)
model.load_state_dict(checkpoint)
model.eval()

def classify(audio_path: str) -> str:
    """Classify audio as Danish or English."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Extract mel spectrogram
    mel_spec = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
    )(waveform)
    log_mel = (mel_spec + 1e-10).log()
    
    # Normalize
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    
    # Add batch and channel dims
    spec = log_mel.unsqueeze(0).unsqueeze(0)
    
    # Classify
    with torch.no_grad():
        output = model(spec)
        probs = torch.softmax(output, dim=1)[0]
    
    danish = probs[0].item() * 100
    english = probs[1].item() * 100
    
    if danish > english:
        return f"🇩🇰 Danish ({danish:.1f}%)\nEnglish: {english:.1f}%"
    else:
        return f"🇬🇧 English ({english:.1f}%)\nDanish: {danish:.1f}%"

with gr.Blocks(title="Danish vs English Detection") as demo:
    gr.Markdown("# 🇩🇰 vs 🇬🇧 Language Detection")
    gr.Markdown("Record or upload audio to detect the language.")
    
    audio = gr.Audio(label="Audio Input", type="filepath")
    result = gr.Textbox(label="Result")
    
    audio.change(fn=classify, inputs=audio, outputs=result)
    
    gr.Markdown("\\n**Model:** Compact CNN (175k params) | **Accuracy:** 96.76%")

if __name__ == "__main__":
    print("Launching at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
