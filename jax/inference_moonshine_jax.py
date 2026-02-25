import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import serialization
import numpy as np
import librosa
import tokenizers
import os
from jax_moonshine.models.moonshine import Moonshine
from einops import rearrange

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
            
        self.tokenizer = tokenizers.Tokenizer.from_file("weights/moonshine/tokenizer.json")
        
        # JIT compile the sub-methods for generation
        self.jit_preprocess = jax.jit(lambda params, audio: self.model.apply(params, audio, method=self.model.preprocess))
        self.jit_encode = jax.jit(lambda params, x: self.model.apply(params, x, method=self.model.encode))
        self.jit_decode = jax.jit(lambda params, tokens, context: self.model.apply(params, tokens, context, method=self.model.decode))

    def transcribe(self, audio_path):
        # 1. Load Audio
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = jnp.array(audio[None, :, None]) # [B, L, 1]
        
        # 2. Preprocess
        print("Preprocessing audio...")
        audio_preprocessed = self.jit_preprocess(self.variables, audio)
        
        # 3. Encode
        print("Encoding...")
        context = self.jit_encode(self.variables, audio_preprocessed)
        
        # 4. Generate Tokens
        print("Generating tokens...")
        tokens = jnp.array([[1]], dtype=jnp.int32) # [B, L]
        output = [1]
        
        max_len = int((audio.shape[1] / 16000) * 6)
        
        for i in range(max_len):
            # For simplicity, we run the full decoder each time here.
            logits = self.jit_decode(self.variables, tokens, context)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)[0]
            output.append(int(next_token))
            print(f"Token {i}: {next_token} ('{self.tokenizer.decode([int(next_token)])}')")
            if next_token == 2: # EOS
                break
            tokens = jnp.concatenate([tokens, next_token[None, None]], axis=-1)
            
        # 5. Decode text
        return self.tokenizer.decode(output)

def run_test():
    audio_path = "moonshine/test-assets/beckett.wav"
    if not os.path.exists(audio_path):
        print(f"Audio not found at {audio_path}")
        return
        
    inference = MoonshineInference("base")
    text = inference.transcribe(audio_path)
    print("\nTranscription:")
    print(text)

if __name__ == "__main__":
    run_test()
