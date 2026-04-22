# Moonshine JAX 🚀

A high-performance implementation of the **Moonshine ASR** model in **JAX/Flax**, optimized for NVIDIA GPUs (via CUDA) and TPUs.

## 🌟 Key Features

-   **Native JAX Implementation**: Built with Flax for maximum performance and bit-level parity with the original models.
-   **GPU-Accelerated Dashboard**: Live transcription with real-time GPU load, VRAM monitoring, and inference latency tracking.
-   **Multi-Mic Support**: Transcribe from multiple microphones simultaneously with independent volume meters.
-   **GUI & CLI Selection**: Easy microphone setup via a graphical window or command-line flags.
-   **Latency Optimized**: JIT-compiled inference paths for sub-200ms response times on modern GPUs.

## 🚀 Quick Start

### 1. Requirements

It is recommended to use the `moonshine-jax` environment.

```bash
# Install system dependencies (Linux)
sudo apt-get install libportaudio2

# Create and setup environment
conda create -n moonshine-jax python=3.10 -y
conda activate moonshine-jax
uv pip install "jax[cuda12]" flax msgpack sounddevice librosa einops tokenizers rich pynvml
```

### 2. Download Weights

Download the JAX weights and tokenizer into the `weights/` directory:

```bash
mkdir -p weights/moonshine
gh release download v1.0.0-jax --repo 1kaiser/moonshine --dir weights/
# Move tokenizer.json to weights/moonshine/ if not present
```

### 3. Run Inference

#### 🎤 Live Microphone (Dashboard Mode)
```bash
PYTHONPATH=. python jax_moonshine/inference_moonshine_jax.py --mic --model tiny
```
*This will open a GUI to select your microphones. Use `Ctrl+C` to exit the dashboard.*

#### 📁 File Transcription
```bash
PYTHONPATH=. python jax_moonshine/inference_moonshine_jax.py --audio test-assets/beckett.wav --model tiny
```

#### 🛠️ Advanced Usage
```bash
# List all audio devices
PYTHONPATH=. python jax_moonshine/inference_moonshine_jax.py --list-devices

# Use specific devices by ID
PYTHONPATH=. python jax_moonshine/inference_moonshine_jax.py --devices 3,7 --no-gui
```

## ⚖️ Model Weights

| Model | Size | Release |
| :--- | :--- | :--- |
| **Moonshine Base (JAX)** | 235 MB | [v1.0.0-jax](https://github.com/1kaiser/moonshine/releases/tag/v1.0.0-jax) |
| **Moonshine Tiny (JAX)** | 104 MB | [v1.0.0-jax](https://github.com/1kaiser/moonshine/releases/tag/v1.0.0-jax) |

---
Ported to JAX by [1kaiser](https://github.com/1kaiser)
