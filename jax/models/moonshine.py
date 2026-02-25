import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, List, Any
from einops import rearrange

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")

def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-3] if t.ndim == 4 else t.shape[-2]
    current_freqs = freqs[-seq_len:, :]
    if t.ndim == 4:
        current_freqs = rearrange(current_freqs, "l d -> 1 l 1 d")
    else:
        current_freqs = rearrange(current_freqs, "l d -> 1 l d")
    t_rot, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot * jnp.cos(current_freqs) + rotate_half(t_rot) * jnp.sin(current_freqs)
    return jnp.concatenate((t_rot, t_unrotated), axis=-1)

class RotaryEmbedding(nn.Module):
    dim: int
    base: int = 10000
    @nn.compact
    def __call__(self, t):
        inv_freq = self.variable('params', 'inv_freq', 
                                 lambda: 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)))
        freqs = jnp.einsum("i, j -> i j", t.astype(jnp.float32), inv_freq.value)
        freqs = jnp.stack((freqs, freqs), axis=-1)
        return rearrange(freqs, "... d r -> ... (d r)")

class AudioPreprocessor(nn.Module):
    dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.dim, kernel_size=(127,), strides=(64,), padding="VALID", use_bias=False, name="conv1")(x)
        x = jax.nn.tanh(x)
        x = nn.GroupNorm(num_groups=1, epsilon=1e-5, name="group_norm")(x)
        x = nn.Conv(features=2 * self.dim, kernel_size=(7,), strides=(3,), padding="VALID", name="conv2")(x)
        x = jax.nn.gelu(x, approximate=True)
        x = nn.Conv(features=self.dim, kernel_size=(3,), strides=(2,), padding="VALID", name="conv3")(x)
        x = jax.nn.gelu(x, approximate=True)
        return x

class FFLinearGelu(nn.Module):
    dim: int
    ff_mult: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.dim * self.ff_mult, name="dense_0")(x)
        x = jax.nn.gelu(x, approximate=True)
        x = nn.Dense(features=self.dim, name="dense_1")(x)
        return x

class FFSwiGLU(nn.Module):
    dim: int
    ff_mult: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.dim * self.ff_mult * 2, name="dense_0")(x)
        x, gate = jnp.split(x, 2, axis=-1)
        x = x * jax.nn.silu(gate)
        x = nn.Dense(features=self.dim, name="dense_1")(x)
        return x

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    use_bias: bool = False
    use_rope: bool = False
    @nn.compact
    def __call__(self, query, key, value, rot_pos_emb=None, mask=None):
        B, L, D = query.shape
        q_kernel = self.param('query_kernel', nn.initializers.lecun_normal(), (D, self.num_heads, self.head_dim))
        k_kernel = self.param('key_kernel', nn.initializers.lecun_normal(), (key.shape[-1], self.num_heads, self.head_dim))
        v_kernel = self.param('value_kernel', nn.initializers.lecun_normal(), (value.shape[-1], self.num_heads, self.head_dim))
        
        q = jnp.einsum("b l d, d h k -> b l h k", query, q_kernel)
        k = jnp.einsum("b l d, d h k -> b l h k", key, k_kernel)
        v = jnp.einsum("b l d, d h k -> b l h k", value, v_kernel)
        
        if self.use_rope and rot_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rot_pos_emb)
            k = apply_rotary_pos_emb(k, rot_pos_emb)
        
        logits = jnp.einsum("b q h d, b k h d -> b h q k", q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("b h q k, b k h d -> b q h d", attn, v)
        
        o_kernel = self.param('out_kernel', nn.initializers.lecun_normal(), (self.num_heads, self.head_dim, D))
        return jnp.einsum("b q h d, h d e -> b q e", out, o_kernel)

class EncoderLayer(nn.Module):
    dim: int
    inner_dim: int
    n_head: int
    ff_mult: int
    ff_swiglu: bool
    @nn.compact
    def __call__(self, x, rot_pos_emb, mask=None):
        _x = x
        x = nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="norm1")(x)
        x = MultiHeadAttention(num_heads=self.n_head, head_dim=self.inner_dim // self.n_head, use_rope=True, name="attention")(x, x, x, rot_pos_emb, mask=mask)
        x = x + _x
        _x = x
        x = nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="norm2")(x)
        ff = FFSwiGLU(self.dim, self.ff_mult, name="ff") if self.ff_swiglu else FFLinearGelu(self.dim, self.ff_mult, name="ff")
        return ff(x) + _x

class Encoder(nn.Module):
    n_layers: int
    dim: int
    inner_dim: int
    n_head: int
    ff_mult: int = 4
    ff_swiglu: bool = False
    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape
        rot_pos_emb_layer = RotaryEmbedding(max(self.inner_dim // self.n_head // 2, 32), name="rot_pos_emb")
        pos_emb = rot_pos_emb_layer(jnp.arange(L))
        for i in range(self.n_layers):
            x = EncoderLayer(self.dim, self.inner_dim, self.n_head, self.ff_mult, self.ff_swiglu, name=f"layer_{i}")(x, pos_emb)
        return nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="final_norm")(x)

class DecoderLayer(nn.Module):
    dim: int
    inner_dim: int
    n_head: int
    ff_mult: int
    ff_swiglu: bool
    @nn.compact
    def __call__(self, x, context, rot_pos_emb, self_attn_mask=None, cross_attn_mask=None):
        _x = x
        x = nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="norm1")(x)
        x = MultiHeadAttention(num_heads=self.n_head, head_dim=self.inner_dim // self.n_head, use_rope=True, name="self_attention")(x, x, x, rot_pos_emb, mask=self_attn_mask)
        x = x + _x
        _x = x
        x = nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="norm2")(x)
        x = MultiHeadAttention(num_heads=self.n_head, head_dim=self.inner_dim // self.n_head, use_rope=False, name="cross_attention")(x, context, context, mask=cross_attn_mask)
        x = x + _x
        _x = x
        x = nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="norm3")(x)
        ff = FFSwiGLU(self.dim, self.ff_mult, name="ff") if self.ff_swiglu else FFLinearGelu(self.dim, self.ff_mult, name="ff")
        return ff(x) + _x

class Decoder(nn.Module):
    n_layers: int
    dim: int
    inner_dim: int
    n_head: int
    vocab_size: int
    ff_mult: int = 4
    ff_swiglu: bool = True
    @nn.compact
    def __call__(self, tokens, context):
        B, L = tokens.shape
        embedding_layer = nn.Embed(num_embeddings=self.vocab_size, features=self.dim, name="embedding")
        x = embedding_layer(tokens)
        rot_pos_emb_layer = RotaryEmbedding(max(self.inner_dim // self.n_head // 2, 32), name="rot_pos_emb")
        pos_emb = rot_pos_emb_layer(jnp.arange(L))
        mask = rearrange(jnp.tril(jnp.ones((L, L))), "q k -> 1 1 q k")
        for i in range(self.n_layers):
            x = DecoderLayer(self.dim, self.inner_dim, self.n_head, self.ff_mult, self.ff_swiglu, name=f"layer_{i}")(x, context, pos_emb, self_attn_mask=mask)
        x = nn.LayerNorm(use_scale=True, use_bias=False, epsilon=1e-5, name="post_norm")(x)
        emb_w = embedding_layer.variables['params']['embedding']
        return jnp.matmul(x, emb_w.T)

class Moonshine(nn.Module):
    dim: int
    inner_dim: int
    n_head: int
    enc_n_layers: int
    dec_n_layers: int
    enc_ff_mult: int = 4
    dec_ff_mult: int = 4
    enc_ff_swiglu: bool = False
    dec_ff_swiglu: bool = True
    vocab_size: int = 32768
    def setup(self):
        self.preprocessor = AudioPreprocessor(self.dim, name="preprocessor")
        self.encoder = Encoder(self.enc_n_layers, self.dim, self.inner_dim, self.n_head, self.enc_ff_mult, self.enc_ff_swiglu, name="encoder")
        self.decoder = Decoder(self.dec_n_layers, self.dim, self.inner_dim, self.n_head, self.vocab_size, self.dec_ff_mult, self.dec_ff_swiglu, name="decoder")
    def __call__(self, audio):
        x = self.preprocessor(audio)
        context = self.encoder(x)
        return self.decoder(jnp.zeros((audio.shape[0], 1), dtype=jnp.int32), context)
    def preprocess(self, audio): return self.preprocessor(audio)
    def encode(self, x): return self.encoder(x)
    def decode(self, tokens, context): return self.decoder(tokens, context)
