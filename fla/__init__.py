# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from pkgutil import extend_path

import paddle

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.2"


def _torch_compat_empty(*args, **kwargs):
    if kwargs.get("device") == "cuda":
        del kwargs["device"]
    return paddle.empty(*args, **kwargs)


paddle.compat.proxy._extend_torch_proxy_overrides(
    {
        "torch.empty": paddle.compat.proxy.RawOverriddenAttribute(
            _torch_compat_empty
        ),
    }
)

from fla import modules, ops  # noqa: E402

__all__ = ["modules", "ops"]
