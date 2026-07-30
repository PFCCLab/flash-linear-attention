# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import paddle


def make_leaf(array: np.ndarray, dtype: str, stop_gradient: bool = False) -> paddle.Tensor:
    tensor = paddle.to_tensor(array).astype(dtype)
    tensor.stop_gradient = stop_gradient
    return tensor


def clone_inputs(arrays: Mapping[str, np.ndarray], dtypes: Mapping[str, str]) -> dict[str, paddle.Tensor]:
    return {name: make_leaf(array, dtypes[name]) for name, array in arrays.items()}


def normalize(x: paddle.Tensor) -> paddle.Tensor:
    x = x.astype("float32")
    return x * paddle.rsqrt((x.square()).sum(axis=-1, keepdim=True) + 1e-6)


def apply_gate(
    g: paddle.Tensor,
    A_log: paddle.Tensor,
    dt_bias: paddle.Tensor | None,
    lower_bound: float | None,
) -> paddle.Tensor:
    H, _ = g.shape[-2:]
    g = g.astype("float32")
    if dt_bias is not None:
        g = g + dt_bias.reshape([H, -1])
    rate = paddle.exp(A_log.reshape([H, 1]).astype("float32"))
    if lower_bound is None:
        return -rate * paddle.nn.functional.softplus(g)
    return lower_bound * paddle.nn.functional.sigmoid(rate * g)


def recurrent_kda(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float | None = None,
    initial_state: paddle.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor | None]:
    dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    value_heads_per_qk_head = HV // H
    scale = K ** -0.5 if scale is None else scale

    q = paddle.repeat_interleave(q.astype("float32"), value_heads_per_qk_head, axis=2) * scale
    k = paddle.repeat_interleave(k.astype("float32"), value_heads_per_qk_head, axis=2)
    v, g, beta = (value.astype("float32") for value in (v, g, beta))

    state = paddle.zeros([B, HV, K, V], dtype="float32")
    if initial_state is not None:
        state = state + initial_state.astype("float32")

    outputs = []
    for token_idx in range(T):
        q_i, k_i, v_i, g_i, beta_i = (
            q[:, token_idx],
            k[:, token_idx],
            v[:, token_idx],
            g[:, token_idx],
            beta[:, token_idx],
        )
        state = state * paddle.exp(g_i).unsqueeze(-1)
        prediction = (k_i.unsqueeze(-1) * state).sum(axis=-2)
        update = beta_i.unsqueeze(-1).unsqueeze(-1) * k_i.unsqueeze(-1) * (v_i - prediction).unsqueeze(-2)
        state = state + update
        outputs.append((q_i.unsqueeze(-1) * state).sum(axis=-2))

    output = paddle.stack(outputs, axis=1).astype(dtype)
    return output, state if output_final_state else None


def tensor_error(ref: paddle.Tensor, actual: paddle.Tensor) -> tuple[float, float]:
    ref = ref.detach().astype("float32")
    actual = actual.detach().astype("float32")
    diff = ref - actual
    max_abs = diff.abs().max().item()
    rms_error = diff.square().mean().sqrt().item()
    rms_ref = ref.square().mean().sqrt().item()
    return max_abs, rms_error / (rms_ref + 1e-8)


def assert_close(
    name: str,
    ref: paddle.Tensor,
    actual: paddle.Tensor,
    ratio: float,
    abs_atol: float = 1e-6,
) -> None:
    assert list(ref.shape) == list(actual.shape), f"{name}: shape {list(actual.shape)} != {list(ref.shape)}"
    assert ref.dtype == actual.dtype, f"{name}: dtype {actual.dtype} != {ref.dtype}"
    assert bool(paddle.isfinite(ref).all().item()), f"{name}: non-finite reference"
    assert bool(paddle.isfinite(actual).all().item()), f"{name}: non-finite result"
    max_abs, error_ratio = tensor_error(ref, actual)
    if max_abs <= abs_atol:
        return
    assert error_ratio < ratio, f"{name}: max_abs={max_abs:.6f}, error_ratio={error_ratio:.6f}, tolerance={ratio:.6f}"
