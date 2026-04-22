import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import serialization
import numpy as np
import librosa
import tokenizers
import os
import sys
import time
import sounddevice as sd
from jax_moonshine.models.moonshine import Moonshine
from einops import rearrange
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
import pynvml
import tkinter as tk
from tkinter import ttk, messagebox

class MoonshineInference:
    def __init__(self, model_name="tiny"):
        if model_name == "tiny":
            self.config = {'dim': 288, 'inner_dim': 288, 'n_head': 8, 'enc_n_layers': 6, 'dec_n_layers': 6}
        else:
            self.config = {'dim': 416, 'inner_dim': 416, 'n_head': 8, 'enc_n_layers': 8, 'dec_n_layers': 8}
            
        self.model = Moonshine(**self.config)
        weights_path = f"weights/moonshine_{model_name}.msgpack"
        with open(weights_path, "rb") as f:
            self.variables = {'params': serialization.from_bytes(None, f.read())}
            
        tokenizer_path = "weights/moonshine/tokenizer.json"
        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        
        self.jit_preprocess = jax.jit(lambda params, audio: self.model.apply(params, audio, method=self.model.preprocess))
        self.jit_encode = jax.jit(lambda params, x: self.model.apply(params, x, method=self.model.encode))
        self.jit_decode = jax.jit(lambda params, tokens, context: self.model.apply(params, tokens, context, method=self.model.decode))
        
        self.transcribe_audio(jnp.zeros((1, 16000, 1)))

    def transcribe_audio(self, audio):
        audio_preprocessed = self.jit_preprocess(self.variables, audio)
        context = self.jit_encode(self.variables, audio_preprocessed)
        tokens = jnp.array([[1]], dtype=jnp.int32)
        output = [1]
        max_len = int((audio.shape[1] / 16000) * 6) + 10
        for i in range(max_len):
            logits = self.jit_decode(self.variables, tokens, context)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)[0]
            output.append(int(next_token))
            if next_token == 2: break
            tokens = jnp.concatenate([tokens, next_token[None, None]], axis=-1)
        return self.tokenizer.decode(output)

def get_gpu_info():
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "used": info.used // 1024**2,
            "total": info.total // 1024**2,
            "load": util.gpu
        }
    except:
        return None

def select_devices_gui():
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, f"{dev['name']} ({dev['hostapi']})"))

    selected_ids = []

    root = tk.Tk()
    root.title("Moonshine Mic Selection")
    root.geometry("400x500")

    label = tk.Label(root, text="Select Microphones to use (Multiple allowed):", pady=10)
    label.pack()

    # Create a frame for the listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(expand=True, fill='both', padx=20, pady=10)

    scrollbar = tk.Scrollbar(frame, orient="vertical")
    listbox = tk.Listbox(frame, selectmode="multiple", yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", expand=True, fill="both")

    for id, name in input_devices:
        listbox.insert(tk.END, f"[{id}] {name}")

    def on_confirm():
        indices = listbox.curselection()
        if not indices:
            messagebox.showwarning("Warning", "Please select at least one microphone!")
            return
        for i in indices:
            selected_ids.append(input_devices[i][0])
        root.destroy()

    btn = tk.Button(root, text="Start Transcribing", command=on_confirm, pady=10, bg="#4CAF50", fg="white")
    btn.pack(pady=20)

    root.mainloop()
    return selected_ids

def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="main", ratio=3),
        Layout(name="stats", ratio=1),
    )
    return layout

def run_mic(model_name="tiny", device_ids=None):
    pynvml.nvmlInit()
    console = Console()
    
    # Use GUI if no devices provided and not listing
    if not device_ids:
        device_ids = select_devices_gui()
        if not device_ids: # User closed window without selection
            return

    with console.status("[bold green]Initializing Moonshine JAX (GPU)..."):
        inference = MoonshineInference(model_name)

    samplerate = 16000
    chunk_duration = 5.0
    chunk_samples = int(samplerate * chunk_duration)
    # Per-device state
    streams = []
    buffers = {d: [] for d in device_ids}
    volumes = {d: 0 for d in device_ids}
    names = {d: sd.query_devices(d)['name'] for d in device_ids}
    transcriptions = []
    last_latency = 0.0

    def make_callback(d_id):
        def callback(indata, frames, time_info, status):
            if indata.size > 0:
                volumes[d_id] = np.sqrt(np.mean(indata**2))
            buffers[d_id].extend(indata.flatten().tolist())
        return callback

    layout = make_layout()

    def update_display():
        layout["header"].update(Panel(f"[bold blue]Moonshine JAX Multi-Mic Transcriber[/bold blue] | Model: [green]{model_name}[/green]", style="white on black"))

        gpu = get_gpu_info()
        gpu_str = f"GPU Load: {gpu['load']}% \nMem: {gpu['used']}/{gpu['total']} MB" if gpu else "GPU: N/A"
        latency_str = f"Inference: [bold yellow]{last_latency:.3f}s[/bold yellow]"

        stats_content = [gpu_str, latency_str, "\n[bold]Microphones[/bold]"]

        for d in device_ids:
            vol_width = 15
            filled = int(min(volumes[d] * 10, 1) * vol_width)
            meter = "[" + "█" * filled + " " * (vol_width - filled) + "]"
            short_name = (names[d][:20] + '..') if len(names[d]) > 20 else names[d]
            stats_content.append(f"{short_name}\n{meter}")
            
        layout["stats"].update(Panel("\n".join(stats_content), title="System"))
        
        text_display = "\n".join(transcriptions[-15:])
        layout["main"].update(Panel(text_display, title="Transcriptions", border_style="green"))
        layout["footer"].update(Panel("Press [bold red]Ctrl+C[/bold red] to Exit", style="white on black"))

    for d in device_ids:
        s = sd.InputStream(samplerate=samplerate, channels=1, device=d, callback=make_callback(d))
        s.start()
        streams.append(s)

    try:
        with Live(layout, refresh_per_second=10, screen=True):
            while True:
                update_display()
                for d in device_ids:
                    if len(buffers[d]) >= chunk_samples:
                        audio_segment = np.array(buffers[d][:chunk_samples])
                        del buffers[d][:chunk_samples]
                        
                        jax_audio = jnp.array(audio_segment[None, :, None])
                        
                        start_time = time.time()
                        text = inference.transcribe_audio(jax_audio)
                        last_latency = time.time() - start_time
                        
                        if text.strip():
                            timestamp = time.strftime('%H:%M:%S')
                            mic_label = names[d][:10]
                            transcriptions.append(f"[bold cyan]{timestamp}[/bold cyan] [[yellow]{mic_label}[/yellow]]: {text} [dim]({last_latency:.2f}s)[/dim]")
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            s.stop()
            s.close()
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Moonshine JAX Inference")
    parser.add_argument("--model", type=str, default="tiny", choices=["tiny", "base"], help="Model size")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--devices", type=str, help="Comma-separated list of device IDs to use")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI selection")
    args = parser.parse_args()

    if args.list_devices:
        from sounddevice import query_devices
        devices = query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"ID {i}: {dev['name']}")
    else:
        dev_ids = None
        if args.devices:
            dev_ids = [int(x.strip()) for x in args.devices.split(",")]
        elif args.no_gui:
            dev_ids = [sd.default.device[0]]
            
        run_mic(args.model, dev_ids)
