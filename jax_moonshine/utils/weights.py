import h5py
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from jax_moonshine.models.moonshine import Moonshine

def load_h5_weights(path):
    weights = {}
    with h5py.File(path, 'r') as f:
        def visit_fn(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)
        f.visititems(visit_fn)
    return weights

def convert_moonshine(model_name="tiny"):
    if model_name == "tiny":
        dim, inner_dim, n_head, enc_n_layers, dec_n_layers = 288, 288, 8, 6, 6
    else:
        dim, inner_dim, n_head, enc_n_layers, dec_n_layers = 416, 416, 8, 8, 8

    model = Moonshine(
        dim=dim, inner_dim=inner_dim, n_head=n_head,
        enc_n_layers=enc_n_layers, dec_n_layers=dec_n_layers
    )
    
    key = jax.random.PRNGKey(0)
    audio = jnp.zeros((1, 16000, 1))
    variables = model.init(key, audio)
    params = variables['params']
    
    base_path = f"weights/moonshine/{model_name}"
    pre_h5 = load_h5_weights(f"{base_path}/preprocessor.weights.h5")
    enc_h5 = load_h5_weights(f"{base_path}/encoder.weights.h5")
    dec_h5 = load_h5_weights(f"{base_path}/decoder.weights.h5")
    
    new_params = jax.tree_util.tree_map(lambda x: x, params)

    # 1. Preprocessor
    new_params['preprocessor']['conv1']['kernel'] = pre_h5['layers/sequential/layers/conv1d/vars/0']
    new_params['preprocessor']['group_norm']['scale'] = pre_h5['layers/sequential/layers/group_normalization/vars/0']
    new_params['preprocessor']['group_norm']['bias'] = pre_h5['layers/sequential/layers/group_normalization/vars/1']
    new_params['preprocessor']['conv2']['kernel'] = pre_h5['layers/sequential/layers/conv1d_1/vars/0']
    new_params['preprocessor']['conv2']['bias'] = pre_h5['layers/sequential/layers/conv1d_1/vars/1']
    new_params['preprocessor']['conv3']['kernel'] = pre_h5['layers/sequential/layers/conv1d_2/vars/0']
    new_params['preprocessor']['conv3']['bias'] = pre_h5['layers/sequential/layers/conv1d_2/vars/1']

    # 2. Encoder
    new_params['encoder']['rot_pos_emb']['inv_freq'] = enc_h5['layers/rotary_embedding/vars/0']
    for i in range(enc_n_layers):
        layer_name = f"layer_{i}"
        prefix = f"layers/functional_{i}" if i > 0 else "layers/functional"
        new_params['encoder'][layer_name]['norm1']['scale'] = enc_h5[f"{prefix}/layers/layer_normalization/vars/0"]
        new_params['encoder'][layer_name]['norm2']['scale'] = enc_h5[f"{prefix}/layers/layer_normalization_1/vars/0"]
        
        new_params['encoder'][layer_name]['attention']['query_kernel'] = enc_h5[f"{prefix}/layers/mha_with_rope/query_dense/vars/0"]
        new_params['encoder'][layer_name]['attention']['key_kernel'] = enc_h5[f"{prefix}/layers/mha_with_rope/key_dense/vars/0"]
        new_params['encoder'][layer_name]['attention']['value_kernel'] = enc_h5[f"{prefix}/layers/mha_with_rope/value_dense/vars/0"]
        new_params['encoder'][layer_name]['attention']['out_kernel'] = enc_h5[f"{prefix}/layers/mha_with_rope/output_dense/vars/0"]
        
        new_params['encoder'][layer_name]['ff']['dense_0']['kernel'] = enc_h5[f"{prefix}/layers/functional/layers/sequential/layers/dense/vars/0"]
        new_params['encoder'][layer_name]['ff']['dense_0']['bias'] = enc_h5[f"{prefix}/layers/functional/layers/sequential/layers/dense/vars/1"]
        new_params['encoder'][layer_name]['ff']['dense_1']['kernel'] = enc_h5[f"{prefix}/layers/functional/layers/sequential/layers/dense_1/vars/0"]
        new_params['encoder'][layer_name]['ff']['dense_1']['bias'] = enc_h5[f"{prefix}/layers/functional/layers/sequential/layers/dense_1/vars/1"]
    new_params['encoder']['final_norm']['scale'] = enc_h5['layers/layer_normalization/vars/0']

    # 3. Decoder
    new_params['decoder']['embedding']['embedding'] = dec_h5['layers/reversible_embedding/vars/0']
    new_params['decoder']['rot_pos_emb']['inv_freq'] = dec_h5['layers/rotary_embedding/vars/0']
    for i in range(dec_n_layers):
        layer_name = f"layer_{i}"
        prefix = f"layers/functional_{i}" if i > 0 else "layers/functional"
        new_params['decoder'][layer_name]['norm1']['scale'] = dec_h5[f"{prefix}/layers/layer_normalization/vars/0"]
        new_params['decoder'][layer_name]['norm2']['scale'] = dec_h5[f"{prefix}/layers/layer_normalization_1/vars/0"]
        new_params['decoder'][layer_name]['norm3']['scale'] = dec_h5[f"{prefix}/layers/layer_normalization_2/vars/0"]
        
        new_params['decoder'][layer_name]['self_attention']['query_kernel'] = dec_h5[f"{prefix}/layers/mha_causal_with_rope/query_dense/vars/0"]
        new_params['decoder'][layer_name]['self_attention']['key_kernel'] = dec_h5[f"{prefix}/layers/mha_causal_with_rope/key_dense/vars/0"]
        new_params['decoder'][layer_name]['self_attention']['value_kernel'] = dec_h5[f"{prefix}/layers/mha_causal_with_rope/value_dense/vars/0"]
        new_params['decoder'][layer_name]['self_attention']['out_kernel'] = dec_h5[f"{prefix}/layers/mha_causal_with_rope/output_dense/vars/0"]
        
        new_params['decoder'][layer_name]['cross_attention']['query_kernel'] = dec_h5[f"{prefix}/layers/mha_precomputed_kv/query_dense/vars/0"]
        new_params['decoder'][layer_name]['cross_attention']['key_kernel'] = dec_h5[f"{prefix}/layers/mha_precomputed_kv/key_dense/vars/0"]
        new_params['decoder'][layer_name]['cross_attention']['value_kernel'] = dec_h5[f"{prefix}/layers/mha_precomputed_kv/value_dense/vars/0"]
        new_params['decoder'][layer_name]['cross_attention']['out_kernel'] = dec_h5[f"{prefix}/layers/mha_precomputed_kv/output_dense/vars/0"]
        
        new_params['decoder'][layer_name]['ff']['dense_0']['kernel'] = dec_h5[f"{prefix}/layers/functional/layers/dense/vars/0"]
        new_params['decoder'][layer_name]['ff']['dense_0']['bias'] = dec_h5[f"{prefix}/layers/functional/layers/dense/vars/1"]
        new_params['decoder'][layer_name]['ff']['dense_1']['kernel'] = dec_h5[f"{prefix}/layers/functional/layers/dense_1/vars/0"]
        new_params['decoder'][layer_name]['ff']['dense_1']['bias'] = dec_h5[f"{prefix}/layers/functional/layers/dense_1/vars/1"]
    new_params['decoder']['post_norm']['scale'] = dec_h5['layers/layer_normalization/vars/0']

    output_path = f"weights/moonshine_{model_name}.msgpack"
    with open(output_path, "wb") as f:
        f.write(serialization.to_bytes(new_params))
    print(f"Successfully converted Moonshine-{model_name} to {output_path}")

if __name__ == "__main__":
    convert_moonshine("tiny")
    convert_moonshine("base")
