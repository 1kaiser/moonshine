# Moonshine: Optimized JAX ASR & Gemini CLI Integration

This repository provides a high-performance, private, and offline Speech-to-Text (ASR) system based on the Moonshine architecture. It is uniquely optimized for **JAX/Flax** to leverage multi-core CPU parallelism and is designed to integrate seamlessly as a **Gemini CLI Extension**.

---

## 🚀 Key Features

- **JAX-Powered Backend**: Utilizes JAX and Flax for high-speed inference.
- **16-Core Parallelism**: Automatically scales across all available CPU cores using `jax.pmap`.
- **Gemini CLI Native**: Integrated `/voice-jax` command for real-time voice input in your terminal.
- **Privacy First**: 100% local processing; no audio data ever leaves your machine.
- **Remote Ready**: Support for voice capture over SSH from another computer.
- **Continuous Logging**: Automated 1-hour session logging with 2-minute chunking.

---

## 🛠 Installation

### 1. Environment Setup
Create and configure the dedicated Conda environment:

\`\`\`bash
# Create environment
conda create -n moonshine python=3.10 -y
conda activate moonshine

# Install JAX & Flax (CPU version for multi-core optimization)
pip install jax[cpu] flax

# Install Audio & CLI dependencies
pip install moonshine-voice sounddevice librosa tokenizers einops
\`\`\`

### 2. Download Weights
The system uses native JAX weights (`.msgpack`). You can download them directly from the repository releases:

\`\`\`bash
gh release download v1.0.0-jax --repo 1kaiser/moonshine --pattern "*.msgpack"
\`\`\`

---

## 💻 Usage

### 1. Local Live Transcription
Run the CLI transcription script directly:

\`\`\`bash
export XLA_FLAGS="--xla_force_host_platform_device_count=16"
python scripts/transcribe_jax_cli.py
\`\`\`

### 2. Parallel Batch Processing
To process multiple WAV files simultaneously across 16 cores:

\`\`\`bash
python jax/inference_parallel_jax.py
\`\`\`

### 3. Gemini CLI Integration
To use the `/voice-jax` command in your Gemini CLI sessions:

1. Copy `voice-jax.toml` to your Gemini skills folder (usually `~/.gemini/skills/voice-jax/`).
2. Ensure the `PYTHONPATH` in the `.toml` points to this repository.
3. Simply type \`/voice-jax\` during any Gemini session.

---

## 🌐 Remote Usage (Voice via SSH)

If you are connected to a powerful server via SSH and want to use the microphone on your **local** laptop, use one of the following methods:

### Method A: The Direct Pipe (Recommended)
Record locally and pipe the audio data directly into the remote JAX engine:

\`\`\`bash
# Run this on your LOCAL machine
arecord -f S16_LE -r 44100 -c 1 -d 10 | ssh <server-ip> \
"cat > /tmp/voice.wav && conda run -n moonshine python $(pwd)/scripts/transcribe_jax_cli.py --file /tmp/voice.wav"
\`\`\`

### Method B: Audio Forwarding
Forward your local PulseAudio/PipeWire stream to the server:

1. **Local**: \`pactl load-module module-native-protocol-tcp auth-anonymous=1\`
2. **SSH**: \`ssh -R 4713:localhost:4713 <server-ip>\`
3. **Remote**: \`export PULSE_SERVER=tcp:localhost:4713\`
4. Run \`/voice-jax\` as if you were sitting at the server.

---

## 📊 Monitoring & Logging

### ASCII Decibel Meter
Every transcription session includes a real-time ASCII level meter to monitor your environment:
\`\`\`text
[##########----------] -24.5 dB | Monitoring...
\`\`\`

### Automatic Archiving
All recordings are automatically saved for history:
- **Audio**: \`/root/.gemini/extensions/gemini-moonshine/recordings/*.wav\`
- **Transcripts**: \`/root/.gemini/extensions/gemini-moonshine/recordings/*.txt\`

---
*Maintained by 1kaiser. Optimized for JAX Multi-Core Research.*
