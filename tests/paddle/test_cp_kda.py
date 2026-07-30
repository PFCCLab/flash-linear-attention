# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import numpy as np
import paddle
import paddle.distributed as dist
import pytest
from kda_eager_reference import apply_gate, assert_close, clone_inputs, normalize, recurrent_kda


def _cp_worker(world_size: int) -> None:
    paddle.enable_compat(scope={"fla", "triton"}, silent=True)
    dist.init_parallel_env()
    rank = dist.get_rank()
    group = dist.new_group(ranks=list(range(world_size)))

    from fla.ops.cp import build_cp_context
    from fla.ops.kda import chunk_kda

    try:
        lengths = [20, 44]
        B, T, H, HV, K, V = 1, sum(lengths), 1, 1, 64, 32
        local_tokens = T // world_size
        local_start = rank * local_tokens
        local_end = local_start + local_tokens
        rng = np.random.default_rng(20260729)
        arrays = {
            "q": rng.standard_normal([B, T, H, K], dtype=np.float32),
            "k": rng.standard_normal([B, T, H, K], dtype=np.float32),
            "v": rng.standard_normal([B, T, HV, V], dtype=np.float32),
            "g": rng.standard_normal([B, T, HV, K], dtype=np.float32),
            "beta": rng.standard_normal([B, T, HV], dtype=np.float32),
            "A_log": rng.uniform(-0.5, 0.5, [HV]).astype(np.float32),
            "dt_bias": rng.uniform(-1.0, 1.0, [HV * K]).astype(np.float32),
            "do": rng.standard_normal([B, T, HV, V], dtype=np.float32),
        }
        dtypes = {
            "q": "bfloat16",
            "k": "bfloat16",
            "v": "bfloat16",
            "g": "bfloat16",
            "beta": "bfloat16",
            "A_log": "float32",
            "dt_bias": "float32",
            "do": "float32",
        }
        local_arrays = {
            name: (
                array[:, local_start:local_end]
                if name in {"q", "k", "v", "g", "beta", "do"}
                else array
            )
            for name, array in arrays.items()
        }
        tri_inputs = clone_inputs(local_arrays, dtypes)
        ref_inputs = clone_inputs(arrays, dtypes)

        cu_values = np.cumsum([0, *lengths]).astype(np.int32)
        cu_seqlens = paddle.to_tensor(cu_values)
        cu_seqlens_cpu = paddle.to_tensor(cu_values, place=paddle.CPUPlace())
        cp_context = build_cp_context(cu_seqlens, group=group, cu_seqlens_cpu=cu_seqlens_cpu)
        tri, final_state = chunk_kda(
            q=tri_inputs["q"],
            k=tri_inputs["k"],
            v=tri_inputs["v"],
            g=tri_inputs["g"],
            beta=tri_inputs["beta"],
            A_log=tri_inputs["A_log"],
            dt_bias=tri_inputs["dt_bias"],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            cp_context=cp_context,
        )
        assert final_state is None
        (tri.astype("float32") * tri_inputs["do"]).sum().backward()

        ref_gate = apply_gate(ref_inputs["g"], ref_inputs["A_log"], ref_inputs["dt_bias"], lower_bound=-5.0)
        ref_beta = paddle.nn.functional.sigmoid(ref_inputs["beta"])
        ref_outputs = []
        start = 0
        for length in lengths:
            end = start + length
            ref_output, _ = recurrent_kda(
                q=normalize(ref_inputs["q"][:, start:end]),
                k=normalize(ref_inputs["k"][:, start:end]),
                v=ref_inputs["v"][:, start:end],
                g=ref_gate[:, start:end],
                beta=ref_beta[:, start:end],
            )
            ref_outputs.append(ref_output)
            start = end
        ref = paddle.concat(ref_outputs, axis=1)
        (ref.astype("float32") * ref_inputs["do"]).sum().backward()

        assert_close("o", ref[:, local_start:local_end], tri, 0.008)
        for name, tolerance in {
            "q": 0.01,
            "k": 0.01,
            "v": 0.01,
            "g": 0.025,
            "beta": 0.025,
        }.items():
            assert_close(
                f"d{name}",
                ref_inputs[name].grad[:, local_start:local_end],
                tri_inputs[name].grad,
                tolerance,
            )

        for name, tolerance, abs_atol in (("A_log", 0.025, 1e-6), ("dt_bias", 0.01, 1e-3)):
            dist.all_reduce(tri_inputs[name].grad, group=group)
            assert_close(f"d{name}", ref_inputs[name].grad, tri_inputs[name].grad, tolerance, abs_atol=abs_atol)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(paddle.device.cuda.device_count() < 2, reason="Paddle KDA CP requires at least two GPUs")
def test_chunk_kda_context_parallel(monkeypatch: pytest.MonkeyPatch):
    cloud_prefixes = (
        "PADDLE_CLUSTER_",
        "PADDLE_CURRENT_ENDPOINT",
        "PADDLE_IS_LOCAL",
        "PADDLE_NUM_GRADIENT_SERVERS",
        "PADDLE_TRAINER",
        "PADDLE_TRAINERS",
        "PADDLE_TRAINING_ROLE",
        "PADDLE_WORKERS_IP_PORT_LIST",
        "POD_",
        "TRAINER_",
        "TRAINERS",
    )
    for name in tuple(os.environ):
        if name.startswith(cloud_prefixes):
            monkeypatch.delenv(name)
    dist.spawn(_cp_worker, args=(2,), nprocs=2)
