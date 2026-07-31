# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import builtins
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import paddle
import pytest

paddle.enable_compat(scope={"fla", "triton"}, silent=True)

import fla  # noqa: E402

paddle.disable_compat()

from fla.modules import FusedRMSNormGated, ShortConvolution  # noqa: E402
from fla.ops.cp import FLACPContext  # noqa: E402
from fla.ops.kda import chunk_kda  # noqa: E402
from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask  # noqa: E402
from fla.utils import tensor_cache  # noqa: E402


@contextmanager
def _assert_no_fla_runtime_imports():
    loaded_modules = {name for name in sys.modules if name == "fla" or name.startswith("fla.")}
    runtime_imports = []
    original_import = builtins.__import__

    def monitored_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        caller = globals_.get("__name__", "") if globals_ else ""
        if caller == "fla" or caller.startswith("fla."):
            runtime_imports.append((caller, name, tuple(fromlist or ()), level))
        return original_import(name, globals_, locals_, fromlist, level)

    builtins.__import__ = monitored_import
    try:
        yield
    finally:
        builtins.__import__ = original_import

    assert runtime_imports == []
    current_modules = {name for name in sys.modules if name == "fla" or name.startswith("fla.")}
    assert current_modules == loaded_modules


@tensor_cache
def _get_unpad_data(attention_mask: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
    indices = paddle.nonzero(attention_mask.flatten()).flatten()
    return indices, prepare_cu_seqlens_from_mask(attention_mask)


def _unpad(x: paddle.Tensor, indices: paddle.Tensor) -> paddle.Tensor:
    return x.reshape([-1, x.shape[-1]])[indices].unsqueeze(0)


def _pad(x: paddle.Tensor, indices: paddle.Tensor, batch_size: int, seq_len: int) -> paddle.Tensor:
    return paddle.scatter_nd(indices.unsqueeze(-1), x.squeeze(0), [batch_size * seq_len, x.shape[-1]]).reshape(
        [batch_size, seq_len, x.shape[-1]]
    )


class _KimiDeltaAttentionTrainingHarness(paddle.nn.Layer):
    """Training-only mirror of the Hugging Face KimiDeltaAttention data flow."""

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        projection_size = self.head_dim * self.num_heads

        self.q_proj = paddle.nn.Linear(self.hidden_size, projection_size, bias_attr=False)
        self.k_proj = paddle.nn.Linear(self.hidden_size, projection_size, bias_attr=False)
        self.v_proj = paddle.nn.Linear(self.hidden_size, projection_size, bias_attr=False)
        self.q_conv1d = ShortConvolution(projection_size, config.linear_attn_config["short_conv_kernel_size"])
        self.k_conv1d = ShortConvolution(projection_size, config.linear_attn_config["short_conv_kernel_size"])
        self.v_conv1d = ShortConvolution(projection_size, config.linear_attn_config["short_conv_kernel_size"])

        self.A_log = self.create_parameter(
            shape=[self.num_heads],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(paddle.log(paddle.uniform([self.num_heads], min=1, max=16))),
        )
        self.f_a_proj = paddle.nn.Linear(self.hidden_size, self.head_dim, bias_attr=False)
        self.f_b_proj = paddle.nn.Linear(self.head_dim, projection_size, bias_attr=False)
        self.dt_bias = self.create_parameter(
            shape=[projection_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-1, 1),
        )
        self.b_proj = paddle.nn.Linear(self.hidden_size, self.num_heads, bias_attr=False)
        self.g_a_proj = paddle.nn.Linear(self.hidden_size, self.head_dim, bias_attr=False)
        self.g_b_proj = paddle.nn.Linear(self.head_dim, projection_size, bias_attr=False)
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = paddle.nn.Linear(projection_size, self.hidden_size, bias_attr=False)
        self.gate_lower_bound = config.linear_attn_config["gate_lower_bound"]

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        indices = None
        cu_seqlens = None
        if attention_mask is not None:
            indices, cu_seqlens = _get_unpad_data(attention_mask)
            hidden_states = _unpad(hidden_states, indices)

        q, _ = self.q_conv1d(self.q_proj(hidden_states), cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(self.k_proj(hidden_states), cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(self.v_proj(hidden_states), cu_seqlens=cu_seqlens)
        g = self.f_b_proj(self.f_a_proj(hidden_states)).reshape([*hidden_states.shape[:-1], self.num_heads, self.head_dim])
        beta = self.b_proj(hidden_states).astype("float32")
        q = q.reshape([*q.shape[:-1], self.num_heads, self.head_dim])
        k = k.reshape([*k.shape[:-1], self.num_heads, self.head_dim])
        v = v.reshape([*v.shape[:-1], self.num_heads, self.head_dim])

        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(
            [*hidden_states.shape[:-1], self.num_heads, self.head_dim]
        )
        o = self.o_norm(o, gate).reshape([*hidden_states.shape[:-1], self.num_heads * self.head_dim])
        o = self.o_proj(o)
        return o if indices is None else _pad(o, indices, batch_size, seq_len)


def _assert_finite_gradients(layer: paddle.nn.Layer, x: paddle.Tensor) -> None:
    assert x.grad is not None
    assert paddle.isfinite(x.grad).all().item()
    for name, parameter in layer.named_parameters():
        assert parameter.grad is not None, name
        assert paddle.isfinite(parameter.grad).all().item(), name


def test_huggingface_kda_training_import_contract():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import paddle

paddle.enable_compat(scope={"fla", "triton"}, silent=True)

import fla

paddle.disable_compat()

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.backends import dispatch as modules_dispatch
from fla.ops.backends import dispatch as ops_dispatch
from fla.ops.kda import chunk_kda
from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask, prepare_split_cu_seqlens
from fla.utils import tensor_cache

assert FusedRMSNormGated is not None
assert ShortConvolution is not None
assert modules_dispatch is ops_dispatch
assert callable(chunk_kda)
assert callable(prepare_cu_seqlens_from_mask)
assert callable(prepare_lens_from_mask)
assert prepare_split_cu_seqlens(batch_size=2, seq_len=5, split_size=3).tolist() == [0, 3, 5, 8, 10]
assert callable(tensor_cache)
assert fla.modules is sys.modules["fla.modules"]
assert fla.modules.conv.cp is sys.modules["fla.modules.conv.cp"]
assert fla.modules.conv.triton is sys.modules["fla.modules.conv.triton"]
assert fla.ops is sys.modules["fla.ops"]
assert fla.ops.cp is sys.modules["fla.ops.cp"]
assert fla.ops.kda is sys.modules["fla.ops.kda"]
assert fla.ops.kda.chunk_kda is chunk_kda
assert fla.ops.utils is sys.modules["fla.ops.utils"]
assert "fla.ops.kda" in sys.modules
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert fla.modules is sys.modules["fla.modules"]
    assert fla.modules.conv.cp is sys.modules["fla.modules.conv.cp"]
    assert fla.modules.conv.triton is sys.modules["fla.modules.conv.triton"]
    assert fla.ops is sys.modules["fla.ops"]
    assert fla.ops.cp is sys.modules["fla.ops.cp"]
    assert fla.ops.kda is sys.modules["fla.ops.kda"]
    assert fla.ops.kda.chunk_kda is chunk_kda
    assert fla.ops.utils is sys.modules["fla.ops.utils"]
    assert ShortConvolution.__module__ == "fla.modules.conv.short_conv"
    assert FusedRMSNormGated.__module__ == "fla.modules.fused_norm_gate"
    assert callable(chunk_kda)
    assert callable(prepare_lens_from_mask)
    assert callable(prepare_cu_seqlens_from_mask)
    assert callable(tensor_cache)


def test_short_convolution_varlen_matches_independent_sequences():
    paddle.seed(42)
    conv = ShortConvolution(64, 4, activation="silu")
    x = paddle.randn([1, 37, 64], dtype="float32")
    x.stop_gradient = False
    x_ref = x.detach().clone()
    x_ref.stop_gradient = False
    weight_ref = conv.weight.detach().clone()
    weight_ref.stop_gradient = False
    cu_seqlens = paddle.to_tensor([0, 13, 37], dtype="int32")

    actual, _ = conv(x, cu_seqlens=cu_seqlens)
    expected = []
    for bos, eos in [(0, 13), (13, 37)]:
        y = paddle.nn.functional.conv1d(
            x_ref[:, bos:eos].transpose([0, 2, 1]),
            weight_ref,
            padding=3,
            groups=64,
        )
        expected.append(paddle.nn.functional.silu(y[:, :, :eos - bos].transpose([0, 2, 1])))
    expected = paddle.concat(expected, axis=1)

    paddle.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    output_gradient = paddle.randn(actual.shape, dtype="float32")
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()
    paddle.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    paddle.testing.assert_close(conv.weight.grad, weight_ref.grad, rtol=1e-5, atol=1e-5)


def test_fused_rms_norm_gated_forward_backward():
    paddle.seed(42)
    x = paddle.randn([2, 17, 2, 64], dtype="float32")
    g = paddle.randn([2, 17, 2, 64], dtype="float32")
    x.stop_gradient = False
    g.stop_gradient = False
    layer = FusedRMSNormGated(64, eps=1e-6, activation="sigmoid")
    x_ref = x.detach().clone()
    x_ref.stop_gradient = False
    g_ref = g.detach().clone()
    g_ref.stop_gradient = False
    weight_ref = layer.weight.detach().clone()
    weight_ref.stop_gradient = False

    actual = layer(x, g)
    expected = x_ref * paddle.rsqrt(x_ref.square().mean(axis=-1, keepdim=True) + 1e-6)
    expected = expected * weight_ref * paddle.nn.functional.sigmoid(g_ref)
    paddle.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    output_gradient = paddle.randn(actual.shape, dtype="float32")
    (actual * output_gradient).sum().backward()
    (expected * output_gradient).sum().backward()
    paddle.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    paddle.testing.assert_close(g.grad, g_ref.grad, rtol=1e-5, atol=1e-5)
    paddle.testing.assert_close(layer.weight.grad, weight_ref.grad, rtol=1e-5, atol=1e-5)


def test_short_convolution_cp_forward_backward_uses_eager_imports():
    paddle.seed(42)
    conv = ShortConvolution(64, 4, activation="silu")
    x = paddle.randn([1, 32, 64], dtype="float32")
    x.stop_gradient = False
    cu_seqlens = paddle.to_tensor([0, 32], dtype="int32")
    cp_context = FLACPContext(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=paddle.to_tensor([0, 32], dtype="int32", place=paddle.CPUPlace()),
        is_first_rank=True,
        conv1d_kernel_size=4,
        pre_num_conv_tokens=0,
    )

    with _assert_no_fla_runtime_imports():
        output, final_state = conv(x, cp_context=cp_context)
        output.square().mean().backward()

    assert final_state is None
    assert paddle.isfinite(output).all().item()
    assert x.grad is not None
    assert paddle.isfinite(x.grad).all().item()
    assert conv.weight.grad is not None
    assert paddle.isfinite(conv.weight.grad).all().item()


@pytest.mark.parametrize("use_padding_mask", [False, True])
def test_kimi_delta_attention_training_forward_backward(use_padding_mask: bool):
    paddle.seed(42)
    config = SimpleNamespace(
        hidden_size=128,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "short_conv_kernel_size": 4,
            "head_dim": 128,
            "num_heads": 1,
            "gate_lower_bound": -5.0,
        },
    )
    with _assert_no_fla_runtime_imports():
        layer = _KimiDeltaAttentionTrainingHarness(config)
        hidden_states = paddle.randn([2 if use_padding_mask else 1, 64, 128], dtype="float32")
        hidden_states.stop_gradient = False
        attention_mask = None
        if use_padding_mask:
            attention_mask = paddle.to_tensor([[1] * 53 + [0] * 11, [1] * 64], dtype="bool")

        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            output = layer(hidden_states, attention_mask)
            loss = output.astype("float32").square().mean()
        loss.backward()

    assert output.shape == hidden_states.shape
    assert paddle.isfinite(output).all().item()
    if attention_mask is not None:
        assert (output[0, 53:] == 0).all().item()
    _assert_finite_gradients(layer, hidden_states)
