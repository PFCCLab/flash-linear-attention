# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.2"

from fla import modules, ops  # noqa: E402

__all__ = ["modules", "ops"]
