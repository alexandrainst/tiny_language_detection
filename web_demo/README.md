# Web Demo – Tiny Language Detection

Interactive web demo for Danish vs English language classification using the Phase 4b
Compact CNN model.

## Features

- **Record audio** directly from your microphone
- **Upload audio files** (WAV, MP3, or any browser-supported format)
- **Playback** recorded/uploaded audio before classification
- **Select model precision** (FP32, FP16, or INT8)
- **Real-time inference** in the browser using ONNX Runtime Web
- **Visual feedback** with audio visualiser and confidence bars

## Quick Start

### Option 1: Python HTTP Server

```bash
cd web_demo
python3 -m http.server 8080
```text

Then open <http://localhost:8080> in your browser.

### Option 2: Node.js HTTP Server

```bash
cd web_demo
npx serve
```text

### Option 3: VS Code Live Server

Open `web_demo/index.html` in VS Code and click "Go Live" (Live Server extension).

## How It Works

1. **Audio Input**: User records via microphone or uploads an audio file
2. **Preprocessing**: Audio is resampled to 16kHz and converted to mel spectrogram
3. **Inference**: ONNX model runs locally in browser via WebAssembly
4. **Output**: Classification result with confidence score

## Technical Details

### Model

- **Architecture**: Compact CNN (175k parameters)
- **Input**: 80-bin log-mel spectrogram (variable time frames)
- **Output**: Binary classification (Danish vs English)

### Available Models

| Model | Disk | RAM | Accuracy | Notes |
|-------|------|-----|----------|-------|
| **FP32** | 708 KB | 886 KB | 96.76% | Full precision |
| **FP16** ⭐ | 708 KB | **543 KB** | 96.65% | **39% RAM savings**, minimal accuracy loss |
| **INT8** | 708 KB | 886 KB | 95.66% | NOT compressed — dequantised to FP32 |

### Why INT8 Isn't Compressed

**ONNX Runtime Web doesn't support INT8 tensors in WebAssembly.** WebAssembly only has `f32` and `f64` instructions — no native INT8 compute.

**What happens:**

1. Original INT8 checkpoint: 175 KB (weights at 1 byte/param)
2. We dequantise to FP32 before ONNX export: 686 KB (weights at 4 bytes/param)
3. Runtime RAM: 886 KB (same as FP32)

**Result:** INT8 gives us **no disk or RAM savings** in the browser. The only benefit is slightly lower accuracy (95.66% vs 96.76%), which is actually a disadvantage!

**Recommendation:** Use **FP16** — it actually saves RAM (543 KB vs 886 KB) while maintaining near-identical accuracy (96.65% vs 96.76%).

### Audio Processing

- **Sample rate**: 16 kHz
- **FFT size**: 512
- **Hop length**: 160 samples (10 ms)
- **Window length**: 400 samples (25 ms)
- **Mel bins**: 80

### Inference

- **Runtime**: ONNX Runtime Web (WebAssembly)
- **Execution**: CPU-only (no GPU required)
- **Optimisation**: SIMD and multi-threading enabled
- **Typical latency**: 50–200 ms depending on device

## Browser Support

- **Chrome/Edge**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support (iOS 14.5+)
- **Opera**: Full support

**Required APIs:**

- `MediaDevices.getUserMedia()` (microphone access)
- `AudioContext` (audio processing)
- `WebAssembly` (ONNX Runtime)

## Files

```text
web_demo/
├── index.html          # Main HTML page
├── app.js              # JavaScript application logic
├── README.md           # This file
└── models/
    ├── config.json     # Model configuration
    ├── model_fp32.onnx       # Full precision model
    ├── model_fp32.onnx.data  # Model weights (FP32)
    ├── model_float16.onnx    # Half precision model
    └── model_float16.onnx.data # Model weights (Float16)
```text

## Development

### Export New Models

To export updated PyTorch models to ONNX:

```bash
uv run src/scripts/export_phase4b_onnx.py
```text

This generates ONNX models in `web_demo/models/`.

### Modify Features

Edit `web_demo/app.js` for:

- Audio processing parameters (in `CONFIG` object)
- UI behaviour
- Inference logic

Edit `web_demo/index.html` for:

- Layout and styling
- UI elements

## Performance

| Device            | Model    | Inference Time | Total Time* |
|-------------------|----------|----------------|-------------|
| MacBook Pro M1    | FP32     | ~50 ms         | ~200 ms     |
| MacBook Pro M1    | Float16  | ~45 ms         | ~190 ms     |
| iPhone 13         | FP32     | ~80 ms         | ~250 ms     |
| Desktop (i7)      | FP32     | ~150 ms        | ~400 ms     |

*Total time includes audio decoding, spectrogram extraction, and inference.

## Privacy

**All processing happens locally in your browser.** No audio data is sent to any server.
The model is downloaded once and cached, then runs entirely client-side.

## Known Limitations

- **Recording length**: Limited to 10 seconds (configurable in `app.js`)
- **Upload length**: Files >30 seconds are rejected
- **Mobile browsers**: May have stricter microphone permissions
- **iOS Safari**: Requires user interaction before audio context can start

## Troubleshooting

**"Failed to load models"**

- Check browser console for errors
- Ensure you're serving files via HTTP (not opening `file://` directly)
- Verify model files exist in `web_demo/models/`

**"Microphone access denied"**

- Grant microphone permissions in browser settings
- Try a different browser
- Use file upload instead of recording

**"Inference failed"**

- Check audio format (should be convertible to 16kHz mono)
- Try a shorter audio clip
- Check browser console for detailed error

## License

MIT License — see project root for details.
