import jax

import os
import subprocess
import sys
from jaxtyping import Array
from jax import shard_map, P
import jax.numpy as jnp
from functools import partial
from jax.experimental.pallas.ops.tpu

def set_XLA_flags_gpu():
    flags = os.environ.get("XLA_FLAGS", "")
    flags += (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    os.environ["XLA_FLAGS"] = flags


def simulate_CPU_devices(device_count: int = 8):
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_force_host_platform_device_count={device_count}"
    os.environ["XLA_FLAGS"] = flags
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def install_package(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


def maybe_shard(x: Array, pspec: jax.P):
    if pspec is not None:
        return jax.lax.with_sharding_constraint(x, pspec)
    return x

simulate_CPU_devices()

mesh = jax.make_mesh((8,), ("seq",), devices = jax.devices())

@partial(
    shard_map,
    in_specs = (P("seq"),P("seq"),P("seq")),
    out_specs = (P("seq")),
    mesh = mesh
)
def shard_attention_just_P(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
) -> jax.Array:
    axis_size = jax.lax.psum(1, axis_name = "seq")
    axis_index = jax.lax.axis_index("seq")
    kv_size = axis_size * k.shape[0]


    def f(carry, x):
        q, k = carry 
        local_score = q @ k.T # (2, 2)
        k = jax.lax.ppermute(k, axis_name = "seq", perm =[(j, (j+1) % axis_size) for j in range(axis_size)])
        return (q, k), local_score

    # Scan over ring rotations to collect local scores in ring order starting
    # from the current device's axis index. Then roll so columns are in global
    # order [0..axis_size-1] for every device before concatenation.
    (_, _), y = jax.lax.scan(f, (q, k), length = axis_size)
    # y has shape (axis_size, local_q, local_k). Reorder along the scan axis
    # so that index 0 corresponds to global K-block 0 for all devices.
    # The sequence observed by device i is [k_i, k_{i-1}, ..., k_{i-(n-1)}].
    # We want [k_0, k_1, ..., k_{n-1}]. Build indices: t = (i - p) mod n.
    n = y.shape[0]
    indices = (axis_index - jnp.arange(n)) % n
    y = jnp.take(y, indices, axis = 0)
    # Concatenate per-step local scores along the last axis.
    y = jnp.transpose(y, (1, 0, 2)).reshape(q.shape[0], -1)
    return y

@partial(
    shard_map,
    in_specs = (P("seq"),P("seq"),P("seq")),
    out_specs = (P("seq")),
    mesh = mesh
)
def shard_attention_with_v(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
) -> jax.Array:
    axis_size = jax.lax.psum(1, axis_name = "seq")
    axis_index = jax.lax.axis_index("seq")
    kv_size = axis_size * k.shape[0]
    local_kv_size = k.shape[0]
    local_q_size = q.shape[0]
    acc = jnp.zeros((local_q_size, q.shape[1])) # (t/n, hidden_size)
    s = jnp.zeros((local_q_size, 1)) # (t/n, 1)
    jax.debug.inspect_array_sharding(s, callback = print)
    prev_max = jnp.full((local_q_size, 1), -1e7)# (t/n, )

    s         = jax.lax.pvary(s, ('seq',))
    acc       = jax.lax.pvary(acc, ('seq',))
    prev_max  = jax.lax.pvary(prev_max, ('seq',))

    def f(carry, x):
        q, k, v, (s, acc, prev_max_logits) = carry 
        local_score = q @ k.T # (t/n, s/n)
        new_max_logits = jnp.maximum(prev_max_logits, jnp.max(local_score, axis = 1, keepdims = True))  # (t/n, 1)

        s = s * jnp.exp(prev_max_logits - new_max_logits) + jnp.sum(jnp.exp(local_score - new_max_logits), axis = 1, keepdims = True)  # (t/n, 1)
        acc = acc * jnp.exp(prev_max_logits - new_max_logits) + jnp.exp(local_score  - new_max_logits)  @ v

        prev_max_logits = new_max_logits
        k = jax.lax.ppermute(k, axis_name = "seq", perm =[(j, (j+1) % axis_size) for j in range(axis_size)])
        v = jax.lax.ppermute(v, axis_name = "seq", perm =[(j, (j+1) % axis_size) for j in range(axis_size)])

        return (q, k, v, (s, acc, prev_max_logits)), local_score

    (q, k, v, (s, acc, prev_max)), _ = jax.lax.scan(f, (q, k, v, (s, acc, prev_max)), length = axis_size)
    # (t/n, hidden_size) / (t/n, )
    out = acc / s

    return out


def main1():
    key = jax.random.key(10)
    key, qkey, kkey, vkey = jax.random.split(key, 4)
    q = jax.random.normal(qkey, (16, 12))
    k = jax.random.normal(kkey, (16, 12))
    v = jax.random.normal(vkey, (16, 12))

    shard_attention_just_P = shard_attention_just_P (q, k, v)

    full_attention = jnp.einsum("th,sh-> ts", q, k)

    if not jnp.allclose(shard_attention_just_P, full_attention, rtol=1e-5, atol=1e-5):
        diff = jnp.max(jnp.abs(shard_attention_just_P - full_attention))
        print("Max abs diff:", diff)
        raise AssertionError("not same")

def main2():
    key = jax.random.key(10)
    key, qkey, kkey, vkey = jax.random.split(key, 4)
    q = jax.random.normal(qkey, (16, 12))
    k = jax.random.normal(kkey, (16, 12))
    v = jax.random.normal(vkey, (16, 12))

    ring_attn = shard_attention_with_v(q, k, v)

    pp = jnp.einsum("th,sh-> ts", q, k)
    attn = jnp.einsum("ts, sh-> th", jax.nn.softmax(pp, axis = 1), v)

    if not jnp.allclose(ring_attn, attn, rtol=1e-5, atol=1e-5):
        diff = jnp.max(jnp.abs(ring_attn - attn))
        print("Max abs diff:", diff)
        raise AssertionError("not same")

if __name__ == "__main__":
    main2()
    


