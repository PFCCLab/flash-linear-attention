# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import sys

import numpy as np
import paddle
import pytest

paddle.enable_compat(scope={"fla", "triton"}, silent=True)

from kda_eager_reference import apply_gate, assert_close, clone_inputs, normalize, recurrent_kda  # noqa: E402

from fla.ops.kda import chunk_fwd as kda_chunk_fwd  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402
from fla.ops.kda.gate import fused_kda_gate  # noqa: E402


def _random_arrays(
    *,
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "q": rng.standard_normal([B, T, H, K], dtype=np.float32),
        "k": rng.standard_normal([B, T, H, K], dtype=np.float32),
        "v": rng.standard_normal([B, T, HV, V], dtype=np.float32),
        "g": rng.standard_normal([B, T, HV, K], dtype=np.float32),
        "beta": rng.standard_normal([B, T, HV], dtype=np.float32),
        "A_log": rng.uniform(-0.5, 0.5, [HV]).astype(np.float32),
        "dt_bias": rng.uniform(-1.0, 1.0, [HV * K]).astype(np.float32),
        "h0": rng.standard_normal([B, HV, K, V], dtype=np.float32),
        "do": rng.standard_normal([B, T, HV, V], dtype=np.float32),
        "dht": rng.standard_normal([B, HV, K, V], dtype=np.float32),
    }


def _run_dense(
    inputs: dict[str, paddle.Tensor],
    *,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    lower_bound = -5.0 if safe_gate else None
    output, final_state = chunk_kda(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"] if use_gate_in_kernel else None,
        dt_bias=inputs["dt_bias"] if use_gate_in_kernel else None,
        initial_state=inputs["h0"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
    )
    loss = (
        output.astype("float32") * inputs["do"]
    ).sum() + (
        final_state * inputs["dht"]
    ).sum()
    loss.backward()
    return output, final_state


def _run_dense_reference(
    inputs: dict[str, paddle.Tensor],
    *,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    safe_gate: bool,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    lower_bound = -5.0 if safe_gate else None
    gate = (
        apply_gate(inputs["g"], inputs["A_log"], inputs["dt_bias"], lower_bound)
        if use_gate_in_kernel
        else inputs["g"]
    )
    beta = paddle.nn.functional.sigmoid(inputs["beta"]) if use_beta_sigmoid_in_kernel else inputs["beta"]
    output, final_state = recurrent_kda(
        q=normalize(inputs["q"]),
        k=normalize(inputs["k"]),
        v=inputs["v"],
        g=gate,
        beta=beta,
        initial_state=inputs["h0"],
        output_final_state=True,
    )
    loss = (
        output.astype("float32") * inputs["do"]
    ).sum() + (
        final_state * inputs["dht"]
    ).sum()
    loss.backward()
    return output, final_state


def test_kda_dependency_closure():
    loaded = {name for name in sys.modules if name == "fla" or name.startswith("fla.")}
    allowed_op_roots = {"attnres", "backends", "common", "cp", "gla", "kda", "utils"}
    loaded_op_roots = {
        name.split(".")[2]
        for name in loaded
        if name.startswith("fla.ops.") and len(name.split(".")) > 2
    }

    assert "fla._paddle" not in loaded
    assert "fla._framework" not in loaded
    assert not any(name.startswith("fla.layers") for name in loaded)
    assert not any(name.startswith("fla.models") for name in loaded)
    assert "fla.ops.gla.chunk" in loaded
    assert kda_chunk_fwd.chunk_gla_fwd_o_gk.__module__ == "fla.ops.gla.chunk"
    assert loaded_op_roots <= allowed_op_roots


@pytest.mark.parametrize("disable_recompute", [False, True])
def test_chunk_kda_dense_forward_backward(disable_recompute: bool):
    arrays = _random_arrays(B=1, T=64, H=1, HV=2, K=64, V=48, seed=42)
    dtypes = {
        "q": "bfloat16",
        "k": "bfloat16",
        "v": "bfloat16",
        "g": "bfloat16",
        "beta": "bfloat16",
        "A_log": "float32",
        "dt_bias": "float32",
        "h0": "float32",
        "do": "float32",
        "dht": "float32",
    }
    tri_inputs = clone_inputs(arrays, dtypes)
    ref_inputs = clone_inputs(arrays, dtypes)

    tri, tri_ht = _run_dense(
        tri_inputs,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        disable_recompute=disable_recompute,
    )
    ref, ref_ht = _run_dense_reference(
        ref_inputs,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
    )

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    for name, tolerance in {
        "q": 0.008,
        "k": 0.008,
        "v": 0.008,
        "g": 0.02,
        "beta": 0.02,
        "A_log": 0.02,
        "dt_bias": 0.008,
        "h0": 0.008,
    }.items():
        assert_close(f"d{name}", ref_inputs[name].grad, tri_inputs[name].grad, tolerance)


@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_kda_gate_backward(has_bias: bool):
    arrays = _random_arrays(B=1, T=37, H=2, HV=2, K=64, V=32, seed=123)
    dtypes = {
        "g": "float32",
        "A_log": "float32",
        "dt_bias": "float32",
    }
    tri_inputs = clone_inputs({name: arrays[name] for name in dtypes}, dtypes)
    ref_inputs = clone_inputs({name: arrays[name] for name in dtypes}, dtypes)
    tri_bias = tri_inputs["dt_bias"] if has_bias else None
    ref_bias = ref_inputs["dt_bias"] if has_bias else None

    tri = fused_kda_gate(tri_inputs["g"], tri_inputs["A_log"], tri_bias, lower_bound=-5.0)
    ref = apply_gate(ref_inputs["g"], ref_inputs["A_log"], ref_bias, lower_bound=-5.0)
    dy = paddle.to_tensor(arrays["g"])
    (tri * dy).sum().backward()
    (ref * dy).sum().backward()

    assert_close("gate", ref, tri, 1e-4)
    assert_close("dg", ref_inputs["g"].grad, tri_inputs["g"].grad, 1e-4)
    assert_close("dA", ref_inputs["A_log"].grad, tri_inputs["A_log"].grad, 1e-4)
    if has_bias:
        assert_close("dbias", ref_inputs["dt_bias"].grad, tri_inputs["dt_bias"].grad, 1e-4)


def test_chunk_kda_varlen_backward():
    lengths = [13, 19, 32]
    B, T, H, HV, K, V = 1, sum(lengths), 1, 1, 64, 40
    arrays = _random_arrays(B=B, T=T, H=H, HV=HV, K=K, V=V, seed=2026)
    arrays["g"] = np.minimum(arrays["g"], -0.01).astype(np.float32)
    arrays["beta"] = (1.0 / (1.0 + np.exp(-arrays["beta"]))).astype(np.float32)
    arrays["h0"] = np.random.default_rng(7).standard_normal([len(lengths), HV, K, V], dtype=np.float32)
    arrays["dht"] = np.random.default_rng(8).standard_normal([len(lengths), HV, K, V], dtype=np.float32)
    dtypes = {
        "q": "float16",
        "k": "float16",
        "v": "float16",
        "g": "float32",
        "beta": "float16",
        "A_log": "float32",
        "dt_bias": "float32",
        "h0": "float32",
        "do": "float32",
        "dht": "float32",
    }
    tri_inputs = clone_inputs(arrays, dtypes)
    ref_inputs = clone_inputs(arrays, dtypes)
    cu_values = np.cumsum([0, *lengths]).astype(np.int32)
    cu_seqlens = paddle.to_tensor(cu_values)
    cu_seqlens_cpu = paddle.to_tensor(cu_values, place=paddle.CPUPlace())

    tri, tri_ht = chunk_kda(
        q=normalize(tri_inputs["q"]).astype(tri_inputs["q"].dtype),
        k=normalize(tri_inputs["k"]).astype(tri_inputs["k"].dtype),
        v=tri_inputs["v"],
        g=tri_inputs["g"],
        beta=tri_inputs["beta"],
        initial_state=tri_inputs["h0"],
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        disable_recompute=True,
    )
    ((tri.astype("float32") * tri_inputs["do"]).sum() + (tri_ht * tri_inputs["dht"]).sum()).backward()

    ref_outputs = []
    ref_states = []
    start = 0
    for sequence_idx, length in enumerate(lengths):
        end = start + length
        ref_output, ref_state = recurrent_kda(
            q=normalize(ref_inputs["q"][:, start:end]),
            k=normalize(ref_inputs["k"][:, start:end]),
            v=ref_inputs["v"][:, start:end],
            g=ref_inputs["g"][:, start:end],
            beta=ref_inputs["beta"][:, start:end],
            initial_state=ref_inputs["h0"][sequence_idx:sequence_idx + 1],
            output_final_state=True,
        )
        ref_outputs.append(ref_output)
        ref_states.append(ref_state)
        start = end
    ref = paddle.concat(ref_outputs, axis=1)
    ref_ht = paddle.concat(ref_states, axis=0)
    ((ref.astype("float32") * ref_inputs["do"]).sum() + (ref_ht * ref_inputs["dht"]).sum()).backward()

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    for name, tolerance in {
        "q": 0.008,
        "k": 0.008,
        "v": 0.008,
        "g": 0.02,
        "beta": 0.02,
        "h0": 0.008,
    }.items():
        assert_close(f"d{name}", ref_inputs[name].grad, tri_inputs[name].grad, tolerance)
