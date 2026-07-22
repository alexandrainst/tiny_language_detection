/**
 * Tiny Language Detection – Web Demo
 *
 * Danish vs English language classification using ONNX Runtime Web.
 * Features: audio recording, file upload, mel spectrogram extraction, and real-time inference.
 */

// Import models config (will be loaded dynamically)
let CONFIG_MODELS = null;

const CONFIG = {
  sampleRate: 16000,
  nMels: 80,
  nFft: 512,
  hopLength: 160, // 10ms at 16kHz
  winLength: 400, // 25ms at 16kHz
  minFreq: 0,
  maxFreq: 8000,
  recordingMaxLength: 10, // seconds
};

// State
let currentModel = null;
let currentSession = null;
let mediaRecorder = null;
let audioContext = null;
let analyser = null;
let microphone = null;
let recordedChunks = [];
let audioBuffer = null;
let isRecording = false;
let models = [];

// DOM elements
const modelSelect = document.getElementById('model-select');
const statusEl = document.getElementById('status');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const playBtn = document.getElementById('playBtn');
const fileInput = document.getElementById('fileInput');
const resultEl = document.getElementById('result');
const resultLanguage = document.getElementById('resultLanguage');
const resultConfidence = document.getElementById('resultConfidence');
const confidenceFill = document.getElementById('confidenceFill');
const detailDuration = document.getElementById('detailDuration');
const detailPrecision = document.getElementById('detailPrecision');
const detailInference = document.getElementById('detailInference');
const canvas = document.getElementById('visualiser');
const canvasCtx = canvas.getContext('2d');

/**
 * Load available models from config.
 */
async function loadModels() {
  try {
    // Fetch config
    const response = await fetch('models/config.json');
    CONFIG_MODELS = await response.json();

    models = CONFIG_MODELS.models.map((m) => ({
      ...m,
      url: `models/${m.file}`,
    }));

    modelSelect.innerHTML = '';
    models.forEach((model, index) => {
      const option = document.createElement('option');
      option.value = index;
      option.textContent = `${model.name} (${model.size_kb.toFixed(1)} KB)`;
      modelSelect.appendChild(option);
    });

    modelSelect.disabled = false;
    setStatus('idle', 'Select a model and click "Start Recording"');

    // Load default model
    await loadModel(0);
  } catch (error) {
    console.error('Failed to load models:', error);
    modelSelect.innerHTML = '<option value="">Failed to load models</option>';
    setStatus('idle', 'Error loading models. Check console for details.');
  }
}

/**
 * Load and initialise ONNX model.
 * @param {number} modelIndex - Index of model to load.
 */
async function loadModel(modelIndex) {
  if (currentSession) {
    await currentSession.release();
    currentSession = null;
  }

  const model = models[modelIndex];
  currentModel = model;

  setStatus('predicting', `Loading ${model.name}...`);

  try {
    // Explicitly use WebAssembly (CPU-only execution in browser)
    currentSession = await ort.InferenceSession.create(model.url, {
      executionProviders: [{
        wasm: {
          // Use SIMD if available for faster inference
          simd: true,
          // Enable multi-threading
          numThreads: navigator.hardwareConcurrency || 4,
        },
      }],
      graphOptimizationLevel: 'all',
    });

    setStatus('idle', `${model.name} loaded. Ready to record or upload.`);
    recordBtn.disabled = false;
    fileInput.disabled = false;
  } catch (error) {
    console.error('Failed to load model:', error);
    setStatus('idle', `Error: ${error.message}`);
    recordBtn.disabled = true;
    fileInput.disabled = true;
  }
}

/**
 * Handle uploaded audio file.
 * @param {File} file - Audio file to process.
 */
async function handleFileUpload(file) {
  try {
    setStatus('predicting', `Loading ${file.name}...`);

    // Ensure audio context exists
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: CONFIG.sampleRate,
      });
    }

    // Read file as array buffer
    const arrayBuffer = await file.arrayBuffer();

    // Decode audio
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));

    // Check duration
    if (audioBuffer.duration > 30) {
      setStatus('idle', `Audio too long (${audioBuffer.duration.toFixed(1)}s). Please use clips under 30 seconds.`);
      fileInput.value = '';
      fileInput.disabled = false;
      return;
    }

    setStatus('ready', `Loaded ${file.name} (${audioBuffer.duration.toFixed(1)}s). Press Play to classify.`);
    playBtn.disabled = false;
    recordBtn.disabled = true;
    fileInput.disabled = true;
    resultEl.classList.add('hidden');

    // Auto-play after short delay
    setTimeout(playAudio, 500);
  } catch (error) {
    console.error('Failed to load audio file:', error);
    setStatus('idle', `Error loading file: ${error.message}`);
    fileInput.value = '';
    fileInput.disabled = false;
  }
}

/**
 * Update status display.
 * @param {string} state - State class (idle, recording, ready, predicting).
 * @param {string} message - Status message.
 */
function setStatus(state, message) {
  statusEl.className = `status ${state}`;
  statusEl.textContent = message;
}

/**
 * Start audio recording.
 */
async function startRecording() {
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: CONFIG.sampleRate,
    });

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: CONFIG.sampleRate,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);

    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.start(100);
    isRecording = true;

    // UI updates
    recordBtn.disabled = true;
    recordBtn.classList.add('recording');
    stopBtn.disabled = false;
    playBtn.disabled = true;
    fileInput.disabled = true;
    resultEl.classList.add('hidden');

    setStatus('recording', 'Recording... Speak now (max 10s)');
    drawVisualiser();
  } catch (error) {
    console.error('Failed to start recording:', error);
    setStatus('idle', `Error: ${error.message}`);
  }
}

/**
 * Stop audio recording.
 */
async function stopRecording() {
  if (!isRecording) return;

  isRecording = false;
  mediaRecorder.stop();

  // Stop all tracks
  mediaRecorder.stream.getTracks().forEach((track) => track.stop());

  // UI updates
  recordBtn.classList.remove('recording');
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  fileInput.disabled = false;

  // Wait for final chunk
  await new Promise((resolve) => setTimeout(resolve, 100));

  // Combine chunks and decode
  const blob = new Blob(recordedChunks, { type: 'audio/webm' });
  const arrayBuffer = await blob.arrayBuffer();

  try {
    audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    setStatus('ready', `Recording complete (${audioBuffer.duration.toFixed(1)}s). Press Play to classify.`);
    playBtn.disabled = false;
  } catch (error) {
    console.error('Failed to decode audio:', error);
    setStatus('idle', 'Error decoding audio. Try again.');
  }

  // Clear visualiser
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Reset to allow new recording or upload.
 */
function resetForNewInput() {
  playBtn.disabled = false;
  playBtn.textContent = '▶ Play';
  recordBtn.disabled = false;
  fileInput.disabled = false;
}

/**
 * Play recorded audio.
 */
async function playAudio() {
  if (!audioBuffer) return;

  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  source.start();

  playBtn.disabled = true;
  playBtn.textContent = '▶ Playing...';

  source.onended = () => {
    playBtn.disabled = false;
    playBtn.textContent = '▶ Play';
    // Run inference after playback
    runInference();
  };
}

/**
 * Run inference on recorded audio.
 */
async function runInference() {
  if (!audioBuffer || !currentSession) return;

  setStatus('predicting', 'Running inference...');

  const startTime = performance.now();

  try {
    // Extract mel spectrogram
    const spectrogram = extractMelSpectrogram(audioBuffer);

    // Prepare input tensor
    const inputTensor = new ort.Tensor(
      'float32',
      spectrogram,
      [1, 1, CONFIG.nMels, spectrogram.length / CONFIG.nMels]
    );

    // Run inference
    const feeds = { input: inputTensor };
    const results = await currentSession.run(feeds);
    const output = results.output.data;

    // Calculate softmax probabilities
    const probabilities = softmax(Array.from(output));

    const inferenceTime = performance.now() - startTime;

    // Display results
    displayResults(probabilities, inferenceTime);
  } catch (error) {
    console.error('Inference failed:', error);
    setStatus('idle', `Inference error: ${error.message}`);
  }
}

/**
 * Extract mel spectrogram from audio buffer.
 * @param {AudioBuffer} audioBuffer - Audio buffer.
 * @returns {Float32Array} Mel spectrogram as flat array.
 */
function extractMelSpectrogram(audioBuffer) {
  const audioData = audioBuffer.getChannelData(0);
  const nSamples = audioData.length;
  const nFrames = Math.floor((nSamples - CONFIG.winLength) / CONFIG.hopLength) + 1;

  // Create mel filterbank
  const melFilterbank = createMelFilterbank();

  // Extract spectrogram
  const spectrogram = new Float32Array(CONFIG.nMels * nFrames);

  for (let frame = 0; frame < nFrames; frame++) {
    const start = frame * CONFIG.hopLength;
    const end = start + CONFIG.winLength;
    const frameData = audioData.slice(start, end);

    // Apply Hann window
    const windowed = applyHannWindow(frameData);

    // FFT
    const spectrum = computeFFT(windowed);

    // Apply mel filterbank
    const melSpectrum = applyMelFilterbank(spectrum, melFilterbank);

    // Log compression
    for (let i = 0; i < CONFIG.nMels; i++) {
      melSpectrum[i] = Math.log(Math.max(melSpectrum[i], 1e-10));
    }

    // Store in output
    for (let i = 0; i < CONFIG.nMels; i++) {
      spectrogram[frame * CONFIG.nMels + i] = melSpectrum[i];
    }
  }

  return spectrogram;
}

/**
 * Create mel filterbank.
 * @returns {Float32Array[]} Mel filterbank weights.
 */
function createMelFilterbank() {
  const nFft = CONFIG.nFft;
  const freqBins = nFft / 2 + 1;

  // Convert min/max frequencies to mel
  const minMel = frequencyToMel(CONFIG.minFreq);
  const maxMel = frequencyToMel(Math.min(CONFIG.maxFreq, CONFIG.sampleRate / 2));

  // Create mel bin edges
  const melBins = [];
  for (let i = 0; i <= CONFIG.nMels + 1; i++) {
    const mel = minMel + i * (maxMel - minMel) / (CONFIG.nMels + 1);
    melBins.push(mel);
  }

  // Convert mel bins to FFT bin indices
  const fftBins = melBins.map((mel) => {
    const freq = melToFrequency(mel);
    return Math.floor((freq * nFft) / CONFIG.sampleRate);
  });

  // Create filterbank
  const filterbank = [];
  for (let i = 0; i < CONFIG.nMels; i++) {
    const filter = new Float32Array(freqBins);
    const start = fftBins[i];
    const center = fftBins[i + 1];
    const end = fftBins[i + 2];

    for (let j = start; j < center && j < freqBins; j++) {
      filter[j] = (j - start) / (center - start);
    }
    for (let j = center; j < end && j < freqBins; j++) {
      filter[j] = (end - j) / (end - center);
    }

    filterbank.push(filter);
  }

  return filterbank;
}

/**
 * Convert frequency to mel scale.
 * @param {number} freq - Frequency in Hz.
 * @returns {number} Frequency in mel.
 */
function frequencyToMel(freq) {
  return 1127 * Math.log(1 + freq / 700);
}

/**
 * Convert mel scale to frequency.
 * @param {number} mel - Frequency in mel.
 * @returns {number} Frequency in Hz.
 */
function melToFrequency(mel) {
  return 700 * (Math.exp(mel / 1127) - 1);
}

/**
 * Apply Hann window to signal.
 * @param {Float32Array|number[]} signal - Input signal.
 * @returns {number[]} Windowed signal.
 */
function applyHannWindow(signal) {
  const windowed = new Array(signal.length);
  for (let i = 0; i < signal.length; i++) {
    windowed[i] = signal[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (signal.length - 1)));
  }
  return windowed;
}

/**
 * Compute FFT (simplified DFT for real input).
 * @param {number[]} signal - Input signal.
 * @returns {number[]} Magnitude spectrum.
 */
function computeFFT(signal) {
  const N = signal.length;
  const spectrum = new Array(N / 2 + 1).fill(0);

  for (let k = 0; k <= N / 2; k++) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      real += signal[n] * Math.cos(angle);
      imag += signal[n] * Math.sin(angle);
    }
    spectrum[k] = Math.sqrt(real * real + imag * imag);
  }

  return spectrum;
}

/**
 * Apply mel filterbank to spectrum.
 * @param {number[]} spectrum - Magnitude spectrum.
 * @param {Float32Array[]} filterbank - Mel filterbank.
 * @returns {number[]} Mel spectrum.
 */
function applyMelFilterbank(spectrum, filterbank) {
  const melSpectrum = new Array(CONFIG.nMels).fill(0);

  for (let i = 0; i < CONFIG.nMels; i++) {
    for (let j = 0; j < spectrum.length; j++) {
      melSpectrum[i] += spectrum[j] * filterbank[i][j];
    }
  }

  return melSpectrum;
}

/**
 * Compute softmax probabilities.
 * @param {number[]} logits - Raw model outputs.
 * @returns {number[]} Probabilities.
 */
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((x) => Math.exp(x - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map((x) => x / sumExp);
}

/**
 * Display classification results.
 * @param {number[]} probabilities - Class probabilities.
 * @param {number} inferenceTime - Inference time in ms.
 */
function displayResults(probabilities, inferenceTime) {
  const danishProb = probabilities[0];
  const englishProb = probabilities[1];

  const isDanish = danishProb > englishProb;
  const language = isDanish ? '🇩🇰 Danish' : '🇬🇧 English';
  const confidence = Math.max(danishProb, englishProb) * 100;

  resultLanguage.textContent = language;
  resultLanguage.style.color = isDanish ? '#c53030' : '#2b6cb0';
  resultConfidence.textContent = `Confidence: ${confidence.toFixed(1)}%`;
  confidenceFill.style.width = `${confidence}%`;

  detailDuration.textContent = `${audioBuffer.duration.toFixed(1)}s`;
  detailPrecision.textContent = currentModel.precision.toUpperCase();
  detailInference.textContent = `${inferenceTime.toFixed(1)} ms`;

  resultEl.classList.remove('hidden');
  setStatus('idle', 'Classification complete. Record or upload another sample.');

  resetForNewInput();
}

/**
 * Draw audio visualiser.
 */
function drawVisualiser() {
  if (!isRecording || !analyser) return;

  requestAnimationFrame(drawVisualiser);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  canvasCtx.fillStyle = '#f7fafc';
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  const barWidth = (canvas.width / bufferLength) * 2.5;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const barHeight = (dataArray[i] / 255) * canvas.height;

    const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
    gradient.addColorStop(0, '#667eea');
    gradient.addColorStop(1, '#764ba2');

    canvasCtx.fillStyle = gradient;
    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

    x += barWidth + 1;
  }
}

// Event listeners
modelSelect.addEventListener('change', (e) => {
  loadModel(parseInt(e.target.value));
});

recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
playBtn.addEventListener('click', playAudio);

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    handleFileUpload(file);
  }
});

// Initialise
loadModels();
