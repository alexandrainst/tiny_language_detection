# Web Demo – Tiny Language Detection

Interactive web interface for Danish vs English language classification.

## Quick Start

**Prerequisites:** Model must be trained first (see `docs/phase4b-results.md`)

```bash
uv run src/scripts/demo_server.py
```

Server runs at <http://localhost:7860>. Flask is installed as part of the project dependencies.

### Open Browser

```
http://localhost:7860
```

### Custom Port

```bash
uv run src/scripts/demo_server.py --port 8080
```

## Features

- **Microphone recording** – Record audio directly in browser
- **File upload** – Upload existing audio files (WAV, MP3, etc.)
- **Multi-label output** – Danish, English, or Unknown (both < 50%)
- **Audio visualisation** – Real-time waveform display during recording
- **Confidence scores** – Per-language probabilities with confidence display

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Browser   │────▶│   run_demo.py    │────▶│  PyTorch    │
│ (HTML/JS)   │     │   (Flask API)    │     │  Inference  │
└─────────────┘     └──────────────────┘     └─────────────┘
      │                       │
      │                       ▼
      │              ┌──────────────────┐
      │              │   Model (FP32)   │
      │              │   ~96.65% acc    │
      │              └──────────────────┘
      │
      └─── Display Results
```

1. **Record/upload audio** – Browser captures audio via Web Audio API
2. **Send to server** – Audio uploaded to `/classify` endpoint
3. **Server inference** – PyTorch model classifies audio
4. **Display results** – JSON response rendered in UI

## Technical Details

### Model

| Property | Value |
|----------|-------|
| Architecture | Compact CNN (Phase 4b) |
| Parameters | 175k |
| Accuracy | 96.65% (DA: 98.05%, EN: 95.10%) |
| Input | 16kHz mono, log-mel spectrogram (80 bands) |
| Output | Independent probabilities per language (sigmoid) |

### API Endpoint

**POST `/classify`**

Request: `multipart/form-data` with `audio` field (audio file)

Response:

```json
{
  "danish": 95.2,
  "english": 12.3,
  "prediction": "Danish",
  "confidence": 95.2
}
```

### Unknown Detection

Multi-label output enables Unknown detection:

- Danish ≥ 50% → "Danish"
- English ≥ 50% → "English"
- Both < 50% → "Unknown"

Useful for non-speech audio or languages not in training set.

## Browser Support

| Browser | Support |
|---------|---------|
| Firefox | ✅ Recommended |
| Chrome | ✅ |
| Edge | ✅ |
| Safari | ⚠️ Limited (older versions) |

**Requirements:**

- Microphone access (for recording)
- Web Audio API
- Modern browser (2020+)

## Files

```
src/web_demo/
├── index.html      # Main UI
├── app.js          # Frontend logic (audio capture, API calls)
├── models/         # ONNX model + config (exported by demo_server.py)
└── README.md       # This file
```

## Development

### Export New Model

```bash
uv run src/scripts/demo_server.py --export-only
```

Exports PyTorch model to `src/web_demo/models/model.onnx`.

### Modify Features

Edit `src/tiny_language_detection/features/mel_spectrogram.py` to adjust:

- Number of mel bands (default: 80)
- Hop length (default: 160 samples @ 16kHz = 10ms)
- Window length (default: 400 samples @ 16kHz = 25ms)

### Server Implementation

See `src/scripts/demo_server.py`:

- Flask app serving static files
- `/classify` endpoint for inference
- Model loading and feature extraction

## Performance

| Metric | Value |
|--------|-------|
| Inference time | ~50–100 ms (single sample) |
| Model size | ~686 KB (FP32) |
| RAM usage | ~700 MB (server process) |
| Audio latency | < 200 ms (end-to-end) |

## Privacy

**Server-side inference:**

- Audio processed on server (not in browser)
- No persistent storage – files deleted after inference
- Local-only by default (`localhost:7860`)

**No external services:**

- No cloud APIs
- No tracking
- No analytics

## Troubleshooting

### Microphone Access Denied

**Error:** "Microphone access denied"

**Solution:**

1. Allow microphone permission when prompted
2. Check browser settings: `about:preferences#privacy` (Firefox)
3. Ensure HTTPS or localhost (required for `getUserMedia`)

### Inference Failed

**Error:** "Classification failed"

**Possible causes:**

1. Model not exported – run `uv run src/scripts/demo_server.py --export-only`
2. Server not running – ensure Flask is running on port 7860
3. Invalid audio format – use WAV, MP3, or WebM format

### High Latency

**Symptom:** Slow classification (> 500 ms)

**Solutions:**

1. Reduce model size (use smaller Phase 4b model)
2. Run on faster CPU
3. Close other resource-intensive applications

## License

MIT License – see root `LICENSE` file.
