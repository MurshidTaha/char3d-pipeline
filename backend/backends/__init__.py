"""
Registry + on-demand loader manager for the four backends.

Per spec: don't keep all four resident in VRAM simultaneously. Load the
selected backend on demand, unload after job completion. This manager
enforces that — it guarantees at most one backend is loaded at a time.
"""

from .charactergen import CharacterGenBackend
from .triposr import TripoSRBackend
from .instantmesh import InstantMeshBackend
from .trellis import TRELLISBackend

_REGISTRY = {
    "charactergen": CharacterGenBackend,
    "triposr": TripoSRBackend,
    "instantmesh": InstantMeshBackend,
    "trellis": TRELLISBackend,
}


class BackendManager:
    """Singleton-style manager — one instance shared across the FastAPI app/worker."""

    def __init__(self):
        self._instances: dict[str, object] = {}
        self._current_loaded: str | None = None

    def get(self, backend_name: str):
        if backend_name not in _REGISTRY:
            raise ValueError(f"Unknown backend '{backend_name}'. Valid: {sorted(_REGISTRY.keys())}")

        # Unload whatever else is currently loaded (only one fits on a T4 at a time)
        if self._current_loaded and self._current_loaded != backend_name:
            self.unload_current()

        if backend_name not in self._instances:
            self._instances[backend_name] = _REGISTRY[backend_name]()

        instance = self._instances[backend_name]
        instance.load()
        self._current_loaded = backend_name
        return instance

    def unload_current(self):
        if self._current_loaded and self._current_loaded in self._instances:
            self._instances[self._current_loaded].unload()
        self._current_loaded = None

    def unload_all(self):
        for inst in self._instances.values():
            inst.unload()
        self._current_loaded = None


backend_manager = BackendManager()
