# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Generic backend dispatch system for FLA operations."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from functools import cache
from typing import ClassVar, TypeVar

from fla.utils import find_spec_cached

F = TypeVar('F', bound=Callable)


class BaseBackend:
    """Base class for operation-specific backends.

    Attributes:
        backend_type (str, Optional):
            Identifier for the backend type, used to distinguish different backend implementations.
            Default: `"base"`.
        package_name (str, Optional):
            Name of the external package required by the backend.
            `None` indicates no external dependency. Default: `None`.
        env_var (str, Optional):
            Environment variable name that controls whether the backend is enabled.
            `None` means always enabled. Default: `None`.
        default_enable (bool, Optional):
            Whether the backend is enabled by default when `env_var` is not set.
            Set to `False` to require explicit user opt-in. Default: `True`.
        priority (int, Optional):
            Backend priority. Lower values indicate higher priority. Default: 5.
    """

    backend_type: ClassVar[str] = "base"
    package_name: ClassVar[str | None] = None
    env_var: ClassVar[str | None] = None
    default_enable: ClassVar[bool] = True
    # Lower number = higher priority, default is 5
    priority: ClassVar[int] = 5

    @classmethod
    def is_available(cls) -> bool:
        if cls.package_name is None:
            return True
        return find_spec_cached(cls.package_name) is not None

    @classmethod
    def is_enabled(cls) -> bool:
        if cls.env_var is None:
            return True
        default_value = "1" if cls.default_enable else "0"
        return os.environ.get(cls.env_var, default_value) != "0"

    @classmethod
    @cache
    def can_use(cls) -> bool:
        return cls.is_available() and cls.is_enabled()

    def verify(self, func_name: str, *args, **kwargs) -> tuple[bool, str | None]:
        """Check if backend can handle the function call."""
        verifier_name = f"{func_name}_verifier"
        verifier = getattr(self, verifier_name, None)
        if verifier is None:
            return True, None

        try:
            return verifier(*args, **kwargs)
        except Exception as e:
            return False, str(e)


_OPERATION_BACKEND_MODULES: dict[str, str] = {
    'modules': 'fla.modules.backends',
}


class BackendRegistry:
    """Per-operation backend registry."""

    _registries: ClassVar[dict[str, BackendRegistry]] = {}
    _initialized: ClassVar[set[str]] = set()
    _init_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self._backends: dict[str, BaseBackend] = {}
        self._active: BaseBackend | None = None
        self._lock = threading.RLock()
        self._logged: set[str] = set()
        BackendRegistry._registries[operation_name] = self

    def register(self, backend: BaseBackend) -> None:
        """Register a backend."""
        with self._lock:
            self._backends[backend.backend_type] = backend
            # Update active backend based on priority
            self._update_active_backend()

    def _get_sorted_backends(self) -> list[BaseBackend]:
        """Get backends sorted by priority (lower number = higher priority).

        Backends with the same priority are sorted by registration order.
        """
        return sorted(
            self._backends.values(),
            key=lambda b: (b.priority, list(self._backends.values()).index(b))
        )

    def _update_active_backend(self) -> None:
        """Update active backend based on priority."""
        for backend in self._get_sorted_backends():
            if backend.can_use():
                self._active = backend
                return

    def get_active(self) -> BaseBackend | None:
        """Get active backend."""
        return self._active

    @classmethod
    def ensure_initialized(cls, operation: str) -> None:
        """Lazy-load backends on first use."""
        if operation in cls._initialized:
            return

        with cls._init_lock:
            if operation in cls._initialized:
                return

            # Import backend module to trigger registration
            module_path = _OPERATION_BACKEND_MODULES.get(
                operation,
                f'fla.ops.{operation}.backends',
            )
            with contextlib.suppress(ImportError):
                __import__(module_path, fromlist=[''])

            cls._initialized.add(operation)


def dispatch(operation: str):
    """Return the default implementation without optional backend routing.

    This branch supports KDA training on NVIDIA GPUs through the default Triton implementation.
    The optional KDA backends are:

    - TileLang is deferred for now.
      TODO: consider supporting it if performance is better than the Triton implementation.
    - FlashKDA, which is inference-only and has no training backward.
    - Triton-Ascend, which only targets Huawei NPUs.

    The module registry similarly contains only Triton-Ascend overrides.
    None of these backends are currently required for the supported CUDA training paths.
    Those paths include dense and variable-length forward/backward and context parallelism.
    Loading them would only expand the unsupported dependency and import surface.

    The upstream dispatch wrapper depends on ``torch.compiler.disable``, which Paddle compat does not provide.
    """
    def decorator(func: F) -> F:
        return func
    return decorator


__all__ = ['BackendRegistry', 'BaseBackend', 'dispatch']
