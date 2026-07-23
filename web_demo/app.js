/**
 * Tiny Language Detection – Web Demo
 * Server-side inference via Flask API
 */

const API_URL = '/classify';

// DOM elements
let statusEl = document.getElementById('status');
let recordBtn = document.getElementById('recordBtn');
let stopBtn = document.getElementById('stopBtn');
let playBtn = document.getElementById('playBtn');
let resultEl = document.getElementById('result');
let resultLanguage = document.getElementById('resultLanguage');
let resultConfidence = document.getElementById('resultConfidence');
let confidenceFill = document.getElementById('confidenceFill');
let detailDuration = document.getElementById('detailDuration');
let recordPrompt = document.getElementById('recordPrompt');
let resultSection = document.getElementById('resultSection');
let canvas = document.getElementById('visualiser');
let canvasCtx = canvas.getContext('2d');

// State
let mediaRecorder = null;
let audioChunks = [];
let audioBlob = null;
let audioUrl = null;
let isRecording = false;
let animationId = null;

// Audio context for visualiser
let audioContext = null;
let analyser = null;
let microphone = null;

/**
 * Set status display.
 * @param {string} state - 'idle', 'recording', 'ready', 'processing', or 'error'
 * @param {string} message - Status message
 */
function setStatus(state, message) {
    statusEl.textContent = message;
    statusEl.className = `status status--${state}`;
    
    if (state === 'recording') {
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        playBtn.disabled = true;
    } else if (state === 'ready') {
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        playBtn.disabled = false;
    } else if (state === 'processing') {
        recordBtn.disabled = true;
        stopBtn.disabled = true;
        playBtn.disabled = true;
    } else {
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        playBtn.disabled = true;
    }
}

/**
 * Start audio recording.
 */
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        audioContext = new AudioContext();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        microphone = audioContext.createMediaStreamSource(stream);
        microphone.connect(analyser);
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioUrl = URL.createObjectURL(audioBlob);
            
            const audio = document.getElementById('audioPlayback');
            audio.src = audioUrl;
            audio.style.display = 'block';
            
            // Auto-classify after recording stops
            classifyAudio();
        };
        
        mediaRecorder.start();
        isRecording = true;
        setStatus('recording', 'Recording...');
        
        // Start visualiser
        drawVisualiser();
        
    } catch (err) {
        console.error('Microphone error:', err);
        setStatus('error', 'Microphone access denied. Please allow microphone access and refresh.');
    }
}

/**
 * Stop audio recording.
 */
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Stop all tracks
        if (microphone) {
            const tracks = microphone.mediaStream.getTracks();
            tracks.forEach(track => track.stop());
        }
        
        // Stop visualiser
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        
        setStatus('processing', 'Classifying...');
    }
}

/**
 * Play recorded audio.
 */
function playAudio() {
    const audio = document.getElementById('audioPlayback');
    if (audio && audioUrl) {
        audio.play();
    }
}

/**
 * Classify audio via server API.
 */
async function classifyAudio() {
    if (!audioBlob) {
        setStatus('error', 'No audio recorded');
        return;
    }
    
    setStatus('processing', 'Classifying...');
    
    const startTime = performance.now();
    
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Classification failed');
        }
        
        const result = await response.json();
        const inferenceTime = performance.now() - startTime;
        
        displayResults(result, inferenceTime);
        setStatus('idle', 'Ready');
        
    } catch (err) {
        console.error('Classification error:', err);
        setStatus('error', `Error: ${err.message}`);
    }
}

/**
 * Display classification results.
 * @param {Object} result - Classification result from API
 * @param {number} inferenceTime - Inference time in ms
 */
function displayResults(result, inferenceTime) {
    resultSection.style.display = 'block';
    
    const { danish, english, prediction, confidence } = result;
    
    resultLanguage.textContent = prediction;
    resultConfidence.textContent = `${confidence.toFixed(1)}%`;
    confidenceFill.style.width = `${confidence}%`;
    
    // Set language-specific styling
    if (prediction === 'Danish') {
        resultLanguage.style.color = '#C8102E';
    } else if (prediction === 'English') {
        resultLanguage.style.color = '#012169';
    } else {
        resultLanguage.style.color = '#666';
    }
    
    // Show details
    detailDuration.textContent = `${audioBlob.size / 1024} KB`;
    document.getElementById('detailDaProb').textContent = `${danish.toFixed(1)}%`;
    document.getElementById('detailEnProb').textContent = `${english.toFixed(1)}%`;
    document.getElementById('detailInferenceTime').textContent = `${inferenceTime.toFixed(0)} ms`;
    
    // Hide record prompt
    recordPrompt.style.display = 'none';
}

/**
 * Draw audio visualiser.
 */
function drawVisualiser() {
    if (!analyser || !isRecording) return;
    
    animationId = requestAnimationFrame(drawVisualiser);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    
    canvasCtx.fillStyle = 'rgb(255, 255, 255)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const barHeight = dataArray[i] / 2;
        
        canvasCtx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
    }
}

/**
 * Handle file upload.
 * @param {Event} event - File input change event
 */
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    audioBlob = file;
    
    // Show uploaded file info
    const audio = document.getElementById('audioPlayback');
    audio.src = URL.createObjectURL(file);
    audio.style.display = 'block';
    
    setStatus('processing', 'Classifying...');
    await classifyAudio();
}

// Event listeners
recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
playBtn.addEventListener('click', playAudio);
document.getElementById('fileInput').addEventListener('change', handleFileUpload);

// Hide stop button initially
stopBtn.disabled = true;
playBtn.disabled = true;
