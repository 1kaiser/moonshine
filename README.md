# Moonshine: Optimized JAX ASR & Gemini CLI Integration

![Moonshine Voice Logo](images/logo.png)

This repository provides a high-performance, private, and offline Speech-to-Text (ASR) system based on the Moonshine architecture. It is uniquely optimized for **JAX/Flax** to leverage multi-core CPU parallelism and is designed to integrate seamlessly as a **Gemini CLI Extension**.

---

## 🚀 Key Features

- **JAX-Powered Backend**: Utilizes JAX and Flax for high-speed inference.
- **16-Core Parallelism**: Automatically scales across all available CPU cores using `jax.pmap`.
- **Gemini CLI Native**: Integrated `/voice-local` (ONNX) and `/voice-jax` commands for real-time voice input.
- **Privacy First**: 100% local processing; no audio data ever leaves your machine.
- **Remote Ready**: Support for voice capture over SSH from another computer.
- **Continuous Logging**: Automated session logging with audio archiving.

---

## 🛠 Installation

### 1. Environment Setup
Create and configure the dedicated Conda environment:

```bash
# Create environment
conda create -n moonshine python=3.10 -y
conda activate moonshine

# Install JAX & Flax (CPU version for multi-core optimization)
pip install jax[cpu] flax

# Install Audio & CLI dependencies
pip install moonshine-voice sounddevice librosa tokenizers einops numpy wave rich
```

### 2. Gemini CLI Integration

To use voice commands in your Gemini CLI sessions:

1.  **Clone this repository**: `git clone https://github.com/1kaiser/moonshine`
2.  **Install the extension**:
    -   Copy `gemini/voice-local.toml` to `~/.gemini/skills/voice-local.toml`.
    -   Update the path in the `.toml` to point to `gemini/transcribe.py` in this repo.
3.  **Usage**: Simply type `/voice-local` during any Gemini session.

---

## 💻 Usage

### 1. Local Live Transcription (JAX)
Run the CLI transcription script directly:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=16"
python scripts/transcribe_jax_cli.py
```

### 2. Parallel Batch Processing
To process multiple WAV files simultaneously across 16 cores:

```bash
python jax/inference_parallel_jax.py
```

---

## 🌐 Remote Usage (Voice via SSH)

If you are connected to a powerful server via SSH and want to use the microphone on your **local** laptop:

### Method: The Direct Pipe (Recommended)
Record locally and pipe the audio data directly into the remote JAX engine:

```bash
# Run this on your LOCAL machine
arecord -f S16_LE -r 16000 -c 1 | ssh <server-ip> \
"conda run -n moonshine python /path/to/moonshine/gemini/transcribe.py"
```

---

## 📊 Monitoring & Logging

### Automatic Archiving
All recordings are automatically saved for history:
- **Audio**: `gemini/recordings/*.wav`

---

## 📖 About Moonshine
[Moonshine](https://moonshine.ai) Voice is an open source AI toolkit for developers building real-time voice applications. It offers higher accuracy than Whisper Large V3 at a fraction of the parameter count, optimized for live streaming.

### Models
| Language   | Architecture     | # Parameters | WER/CER |
| ---------- | ---------------- | ------------ | ------- |
| English    | Medium Streaming | 245 million  | 6.65%   |
| English    | Small Streaming  | 123 million  | 7.84%   |
| English    | Tiny Streaming   | 34 million   | 12.00%  |

---
*Maintained by 1kaiser. Optimized for JAX Multi-Core Research.*
