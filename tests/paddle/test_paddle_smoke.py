import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np


def run_python(code: str, env=None) -> None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=full_env,
    )


def test_import():
    run_python(
        """
        import paddle

        paddle.enable_compat(scope={"fla"})

        import fla
        import fla.layers

        assert fla is not None
        assert fla.layers is not None
        """
    )


def test_precision(tmp_path: Path):
    paddle_fla_root = os.environ.get("PADDLE_FLA_ROOT")
    torch_fla_root = os.environ.get("TORCH_FLA_ROOT")

    assert paddle_fla_root, "PADDLE_FLA_ROOT must point to the Paddle PR checkout"
    assert torch_fla_root, "TORCH_FLA_ROOT must point to the Torch reference checkout"

    shared_npz = tmp_path / "shared_input_and_state.npz"
    torch_npz  = tmp_path / "torch_out.npz"
    paddle_npz = tmp_path / "paddle_out.npz"

    torch_env  = {"PYTHONPATH": torch_fla_root}
    paddle_env = {"PYTHONPATH": paddle_fla_root}

    # Step 1: Torch forward — save weights and output as float32 numpy.
    run_python(
        f"""
        import sys
        # Remove paddle_fla editable-install path so torch fla takes precedence.
        sys.path = [p for p in sys.path if "paddle_fla" not in p]

        import numpy as np
        import torch

        from fla.layers import LinearAttention

        torch.manual_seed(2026)

        hidden_size = 64
        num_heads   = 4
        batch_size  = 2
        seq_len     = 128

        layer = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mode="chunk",
        ).cuda().to(torch.bfloat16)
        layer.eval()

        x = torch.randn(
            batch_size, seq_len, hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            y = layer(x)
        if isinstance(y, (tuple, list)):
            y = y[0]

        state = {{
            k: v.detach().cpu().float().numpy()
            for k, v in layer.state_dict().items()
        }}

        np.savez(
            "{shared_npz}",
            x=x.detach().cpu().float().numpy(),
            **{{"state__" + k: v for k, v in state.items()}},
        )

        np.savez("{torch_npz}", y=y.detach().cpu().float().numpy())
        """,
        env=torch_env,
    )

    # Step 2: Paddle forward — load same weights and input, save output.
    run_python(
        f"""
        import numpy as np
        import paddle

        paddle.enable_compat(scope={{"fla"}})

        from fla.layers import LinearAttention

        data = np.load("{shared_npz}")

        hidden_size = 64
        num_heads   = 4

        layer = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mode="chunk",
        )
        layer.eval()

        # paddle.compat.nn.Linear weight layout matches torch.nn.Linear: [out, in].
        # No transpose needed.
        state = {{}}
        for key in data.files:
            if not key.startswith("state__"):
                continue
            name = key[len("state__"):]
            state[name] = paddle.to_tensor(data[key])

        layer.set_state_dict(state)
        # Cast to bfloat16 after weight loading (matches test_paddle.py pattern).
        layer.to(dtype="bfloat16")

        x = paddle.to_tensor(data["x"]).cast("bfloat16")

        with paddle.no_grad():
            y = layer(x)
        if isinstance(y, (tuple, list)):
            y = y[0]

        assert list(y.shape) == list(data["x"].shape)
        assert bool(paddle.isfinite(y.cast("float32")).all())

        np.savez("{paddle_npz}", y=y.cast("float32").numpy())
        """,
        env=paddle_env,
    )

    # Step 3: Compare forward outputs.
    # bfloat16 has ~3 decimal digits of precision; 1e-2 tolerance is appropriate.
    torch_data  = np.load(torch_npz)
    paddle_data = np.load(paddle_npz)

    y_paddle = paddle_data["y"]
    y_torch  = torch_data["y"]

    abs_err = np.abs(y_paddle - y_torch)
    denom   = np.maximum(np.maximum(np.abs(y_paddle), np.abs(y_torch)), 1e-12)
    rel_err = abs_err / denom

    print("\n===== Precision (LinearAttention bf16 forward) =====")
    print(f"  torch  mean : {y_torch.mean():.6f}")
    print(f"  paddle mean : {y_paddle.mean():.6f}")
    print(f"  max abs err : {abs_err.max():.6f}")
    print(f"  mean abs err: {abs_err.mean():.6f}")
    print(f"  max rel err : {rel_err.max():.6f}")
    print(f"  mean rel err: {rel_err.mean():.6f}")

    np.testing.assert_allclose(y_paddle, y_torch, rtol=1e-2, atol=1e-2)
    print("  ✅ PASS")