"""Verify Rust AZ inference matches PyTorch output."""
import sys, os, numpy as np, torch
sys.path.insert(0, '.')
from penguinchess.ai.alphazero_net import AlphaZeroResNet2M, AlphaZeroResNet1M, AlphaZeroResNet3M
from penguinchess.rust_ffi import ffi_az_create, get_engine
from penguinchess.rust_core import RustCore
from alphazero.train import export_resnet_to_rust, OBS_DIM, _encode_flat_obs

torch.manual_seed(42)
np.random.seed(42)

def test_named(name, NetClass):
    print(f'\n{"="*60}')
    print(f'Testing {name}...')
    print(f'{"="*60}')
    
    net = NetClass()
    net.eval()
    
    # Generate 3 random obs
    obs = torch.randn(1, OBS_DIM)
    
    # PyTorch forward
    with torch.no_grad():
        py_logits, py_value = net(obs)
    
    # Export to Rust
    sd = {k: v for k, v in net.state_dict().items()}
    
    if 'resnet' in NetClass.arch_name:
        layer_info, weights, biases, pi, v1, v2 = export_resnet_to_rust(sd)
        arch_str = 'resnet'
    else:
        from alphazero.train import export_mlp_to_rust
        layer_info, weights, biases, pi, v1, v2 = export_mlp_to_rust(sd)
        arch_str = 'mlp'
    
    handle = ffi_az_create(arch=arch_str, layer_info=layer_info, weights=weights, biases=biases, policy_idx=pi, value1_idx=v1, value2_idx=v2)
    
    # Rust forward via low-level FFI (separate logits/values buffers)
    lib = get_engine()._lib
    from ctypes import POINTER, c_float, c_int32
    
    rust_obs = obs[0].numpy().astype(np.float32)
    obs_ptr = rust_obs.ctypes.data_as(POINTER(c_float))
    
    logits_out = (c_float * 60)()
    value_out = (c_float * 1)()
    
    rc = lib.ffi_az_evaluate(c_int32(handle._handle), obs_ptr, c_int32(1), logits_out, value_out)
    assert rc == 0, f"Rust evaluate returned {rc}"
    
    rust_logits = np.array([logits_out[i] for i in range(60)])
    rust_value = float(value_out[0])
    
    # Compare
    py_p = torch.softmax(py_logits[0], dim=0).numpy()
    rust_p = np.exp(rust_logits - np.max(rust_logits))
    rust_p /= rust_p.sum()
    
    logit_mse = np.mean((py_logits[0].numpy() - rust_logits) ** 2)
    logit_max_diff = np.max(np.abs(py_logits[0].numpy() - rust_logits))
    kl_div = np.sum(py_p * np.log(py_p / (rust_p + 1e-10) + 1e-10))
    value_diff = abs(float(py_value[0, 0]) - rust_value)
    
    print(f'  Logit MSE:     {logit_mse:.6f}')
    print(f'  Logit max diff: {logit_max_diff:.6f}')
    print(f'  Policy KL:     {kl_div:.8f}')
    print(f'  Value diff:    {value_diff:.6f}')
    print(f'  Py Value:      {py_value[0,0].item():.4f}')
    print(f'  Rust Value:    {rust_value:.4f}')
    
    # Test with real game state
    from penguinchess.rust_ffi import get_engine as ge
    e = ge()
    core = RustCore(engine=e, seed=42).reset(seed=42)
    for _ in range(6):
        leg = core.get_legal_actions()
        if leg: core.step(leg[0])
    flat_obs = _encode_flat_obs(core)
    obs2 = torch.from_numpy(flat_obs).float().unsqueeze(0)
    
    with torch.no_grad():
        py2_logits, py2_value = net(obs2)
    
    rust_obs2 = flat_obs.astype(np.float32)
    obs_ptr2 = rust_obs2.ctypes.data_as(POINTER(c_float))
    logits_out2 = (c_float * 60)()
    value_out2 = (c_float * 1)()
    lib.ffi_az_evaluate(c_int32(handle._handle), obs_ptr2, c_int32(1), logits_out2, value_out2)
    
    rust2_logits = np.array([logits_out2[i] for i in range(60)])
    rust2_value = float(value_out2[0])
    
    logit_mse2 = np.mean((py2_logits[0].numpy() - rust2_logits) ** 2)
    value_diff2 = abs(float(py2_value[0, 0]) - rust2_value)
    print(f'  [real obs] Logit MSE: {logit_mse2:.6f}')
    print(f'  [real obs] Value diff: {value_diff2:.6f}')
    
    PASS = logit_max_diff < 0.1 and value_diff < 0.1 and logit_mse2 < 0.1
    print(f'  {"PASS" if PASS else "FAIL"}')
    
    handle.free()
    core.close()
    return PASS

print('Testing AlphaZeroResNet1M...', flush=True)
test_named('AlphaZeroResNet1M', AlphaZeroResNet1M)
print('Testing AlphaZeroResNet2M...', flush=True)
test_named('AlphaZeroResNet2M', AlphaZeroResNet2M)
print('Testing AlphaZeroResNet3M...', flush=True)
test_named('AlphaZeroResNet3M', AlphaZeroResNet3M)
