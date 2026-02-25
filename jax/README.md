# Moonshine JAX 🚀

A high-performance implementation of the **Moonshine ASR** model in **JAX/Flax**, optimized for TPUs and GPUs.

## 🌟 Key Features
- **Native JAX Implementation:** Built from the ground up using Flax for maximum performance.
- **Architectural Parity:** Verified against the original Keras implementation with bit-level transcription accuracy.
- **Repetition Fix:** Resolved the common repetition issue in early ports through proper weight tying and padding management.
- **Lightweight Inference:** Efficient execution using converted `.msgpack` weights.

## 🚀 Quick Start

### 1. Requirements
```bash
pip install jax flax msgpack soundfile librosa
```

### 2. Run Inference
```python
python jax/inference_moonshine_jax.py --audio test.wav --model tiny
```

## ⚖️ Model Weights
JAX-compatible weights (`.msgpack`) are available in the [GitHub Releases](https://github.com/1kaiser/moonshine/releases).

| Model | Size |
| :--- | :--- |
| **Moonshine Base (JAX)** | 235 MB |
| **Moonshine Tiny (JAX)** | 104 MB |

---
Ported to JAX by [1kaiser](https://github.com/1kaiser)
