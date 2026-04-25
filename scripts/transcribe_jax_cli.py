import sys, os, time, queue
import numpy as np
import sounddevice as sd
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes
from jax_moonshine.models.moonshine import Moonshine
import tokenizers

def load_weights(model, path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f: data = f.read()
    variables = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 16000, 1)))
    return from_bytes(variables['params'], data)

def main():
    repo_root = "/home/kaiser/gemini_project2/kaiser_moonshine_repo"
    weights_path = "/home/kaiser/gemini_project2/weights/moonshine_tiny.msgpack"
    tokenizer_path = os.path.join(repo_root, "weights/moonshine/tokenizer.json")

    model = Moonshine(dim=288, inner_dim=288, n_head=8, enc_n_layers=6, dec_n_layers=6)
    params = load_weights(model, weights_path)
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    
    @jax.jit
    def encode(p, audio):
        x = model.apply({'params': p}, audio, method=model.preprocess)
        return model.encode(x)

    @jax.jit
    def decode_step(p, tokens, context):
        return model.apply({'params': p}, tokens, context, method=model.decode)

    samplerate = 44100
    q = queue.Queue()
    all_audio = []
    
    def callback(indata, frames, time, status):
        q.put(indata.copy().flatten())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        sys.stderr.write("Listening (JAX Backend)... Speak now.\n")
        last_activity = time.time()
        start_time = time.time()
        while True:
            while not q.empty():
                chunk = q.get_nowait()
                all_audio.extend(chunk)
                if np.sqrt(np.mean(chunk**2)) > 0.01: last_activity = time.time()
            if (len(all_audio) > 0 and time.time() - last_activity > 1.5) or (time.time() - start_time > 20):
                break
            time.sleep(0.1)

    # Transcription
    audio_jax = jnp.array(all_audio).reshape(1, -1, 1)
    context = encode(params, audio_jax)
    
    tokens = jnp.array([[1]], dtype=jnp.int32)
    output = []
    for _ in range(100):
        logits = decode_step(params, tokens, context)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)[0]
        if next_token == 2: break
        output.append(int(next_token))
        tokens = jnp.concatenate([tokens, next_token[None, None]], axis=-1)
    
    print(tokenizer.decode(output))

if __name__ == "__main__":
    main()
