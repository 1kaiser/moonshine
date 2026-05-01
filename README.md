![Moonshine Voice Logo](images/logo.png)

# Moonshine: Optimized JAX ASR & Gemini CLI Integration

This repository provides a high-performance, private, and offline Speech-to-Text (ASR) system based on the Moonshine architecture. It is uniquely optimized for JAX/Flax to leverage multi-core CPU parallelism and is designed to integrate seamlessly as a Gemini CLI Extension.

---

## 🚀 Key Features

*   **JAX-Powered Backend**: Utilizes JAX and Flax for high-speed inference.
*   **16-Core Parallelism**: Automatically scales across all available CPU cores using `jax.pmap`.
*   **Gemini CLI Native**: Integrated `/voice-local` command for real-time voice input in your terminal.
*   **Privacy First**: 100% local processing; no audio data ever leaves your machine.
*   **Remote Ready**: Support for voice capture over SSH from another computer.
*   **Continuous Logging**: Automated session logging in the `recordings/` folder.

---

## 🛠 Installation

### 1. Environment Setup

Create and configure the dedicated Conda environment:

```bash
# Create environment
conda create -n moonshine python=3.10 -y
conda activate moonshine

# Install dependencies
pip install jax[cpu] flax
pip install moonshine-voice sounddevice librosa tokenizers einops numpy wave rich
```

### 2. Gemini CLI Integration

To use the `/voice-local` command in your Gemini CLI sessions:

1.  Clone this repository: `git clone https://github.com/1kaiser/moonshine`
2.  Copy `gemini/voice-local.toml` to your Gemini skills folder (usually `~/.gemini/skills/voice-local.toml`).
3.  Update the path in `voice-local.toml` to point to the `gemini/transcribe.py` in your cloned repository.
4.  Simply type `/voice-local` during any Gemini session.

---

## 💻 Usage

### 1. Local Live Transcription (JAX)

Run the CLI transcription script directly:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=16"
PYTHONPATH=. python jax_moonshine/inference_moonshine_jax.py --model tiny
```

### 2. Parallel Batch Processing

To process multiple WAV files simultaneously across 16 cores:

```bash
python jax/inference_parallel_jax.py
```

---

## 🌐 Remote Usage (Voice via SSH)

If you are connected to a powerful server via SSH and want to use the microphone on your local laptop:

### Method: The Direct Pipe
Record locally and pipe the audio data directly into the remote engine:

```bash
# Run this on your LOCAL machine
arecord -f S16_LE -r 16000 -c 1 | ssh remote_server "conda run -n moonshine python /path/to/gemini/transcribe.py"
```

---

## 📊 Monitoring & Logging

### Automatic Archiving
All recordings are automatically saved for history:
*   **Audio**: `gemini/recordings/*.wav`

---

## License
Maintained by 1kaiser. Optimized for JAX Multi-Core Research.
The core Moonshine models are released under the MIT License and Moonshine Community License.
