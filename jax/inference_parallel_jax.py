import os
import jax
import jax.numpy as jnp
from jax import pmap
import flax
from flax.serialization import from_bytes
import numpy as np
import librosa
from jax_moonshine.models.moonshine import Moonshine
from functools import partial

# Set XLA flags to use all available cores (default to 16 for this system)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

class MoonshineParallelInference:
    def __init__(self, model_name="tiny"):
        if model_name == "tiny":
            self.config = {'dim': 288, 'inner_dim': 288, 'n_head': 8, 'enc_n_layers': 6, 'dec_n_layers': 6}
        else:
            self.config = {'dim': 416, 'inner_dim': 416, 'n_head': 8, 'enc_n_layers': 8, 'dec_n_layers': 8}
            
        self.num_devices = jax.local_device_count()
        self.model = Moonshine(**self.config)
        
        # Adjust weight loading path based on repo structure
        weights_path = f"weights/moonshine_{model_name}.msgpack"
        if not os.path.exists(weights_path):
            # Try parent directory if running from jax/
            weights_path = f"../weights/moonshine_{model_name}.msgpack"

        with open(weights_path, "rb") as f:
            data = f.read()
        
        # Init variables
        variables = self.model.init(jax.random.PRNGKey(0), jnp.zeros((1, 16000, 1)))
        params = from_bytes(variables['params'], data)
        
        # Replicate parameters for parallel processing
        self.replicated_params = flax.jax_utils.replicate(params)
        
        # Define parallel encoding function
        @partial(pmap, axis_name='batch')
        def _parallel_encode(p, audio_batch):
            # audio_batch shape: (samples_per_device, length, 1)
            x = self.model.apply({'params': p}, audio_batch, method=self.model.preprocess)
            context = self.model.apply({'params': p}, x, method=self.model.encode)
            return context
            
        self.jit_parallel_encode = _parallel_encode

    def preprocess_audio(self, file_path, target_sr=16000, duration_sec=10):
        audio, _ = librosa.load(file_path, sr=target_sr)
        max_len = target_sr * duration_sec 
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            audio = np.pad(audio, (0, max_len - len(audio)))
        return audio.reshape(-1, 1)

    def encode_batch(self, audio_files):
        audio_data = [self.preprocess_audio(f) for f in audio_files]
        
        # Pad batch to be divisible by device count
        original_count = len(audio_data)
        while len(audio_data) % self.num_devices != 0:
            audio_data.append(np.zeros_like(audio_data[0]))
            
        audio_batch = np.stack(audio_data)
        samples_per_device = len(audio_batch) // self.num_devices
        audio_batch = audio_batch.reshape((self.num_devices, samples_per_device, -1, 1))
        
        print(f"Distributing {original_count} files across {self.num_devices} JAX cores...")
        contexts = self.jit_parallel_encode(self.replicated_params, audio_batch)
        return contexts

if __name__ == "__main__":
    # Test with local files if they exist
    inf = MoonshineParallelInference("tiny")
    print(f"Parallel Inference Initialized with {inf.num_devices} devices.")
