"""
Backend routing configuration for the Character-to-3D tool.

Kept separate from main.py so the routing table can be re-tuned without
redeploying — open-weight model quality shifts fast (per spec, Open Decisions).
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "routing_table.json")

DEFAULT_ROUTING_TABLE = {
    "human": {
        "anime": "charactergen",
        "stylized": "charactergen",
        "photoreal": "trellis",
    },
    "animal": {
        "*": "instantmesh",
    },
    "fantasy_creature": {
        "*": "instantmesh",
    },
    "stylized_humanoid": {
        "*": "charactergen",
    },
    "robot_mechanical": {
        "*": "trellis",
    },
}

FALLBACK_BACKEND = "triposr"

BACKEND_INFO = {
    "charactergen": {
        "vram_gb": 16,
        "reason": "Anime/stylized style detected → CharacterGen (pose-canonicalized, purpose-built for your cast)",
        "license": "Apache 2.0 (weights: zjpshadow/CharacterGen on HuggingFace) — commercial use cleared. "
                    "One checkpoint flagged 'scanned unsafe' on the HF model card — almost certainly a "
                    "pickle-serialization false positive on an older PyTorch checkpoint. Load with "
                    "torch.load(..., weights_only=True) rather than trusting it blindly. No clear "
                    "repo-level LICENSE on the inference code repo — check before forking/redistributing "
                    "the code itself; doesn't affect rights over generated outputs.",
        "stages": ["2d_pose_canonicalization_multiview", "3d_sparse_view_reconstruction"],
    },
    "triposr": {
        "vram_gb": 7,
        "reason": "Fast draft/preview pass — not for final assets",
        "license": "Check current HuggingFace model card before commercial use.",
        "stages": ["single_pass"],
    },
    "instantmesh": {
        "vram_gb": 16,
        "reason": "Organic non-anime shape (animal/creature) → InstantMesh",
        "license": "Check current HuggingFace model card before commercial use.",
        "stages": ["single_pass"],
    },
    "trellis": {
        "vram_gb": 16,
        "reason": "Hard-surface / photoreal-human / mechanical → TRELLIS",
        "license": "Check current HuggingFace model card before commercial use.",
        "stages": ["single_pass"],
    },
}

VALID_BACKENDS = set(BACKEND_INFO.keys())
VALID_ORGANISM_TYPES = {"human", "animal", "fantasy_creature", "stylized_humanoid", "robot_mechanical"}
VALID_STYLE_MODES = {"photoreal", "stylized", "anime"}


def load_routing_table() -> dict:
    """Load routing table from disk if present, else fall back to the default and write it out."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    save_routing_table(DEFAULT_ROUTING_TABLE)
    return DEFAULT_ROUTING_TABLE


def save_routing_table(table: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(table, f, indent=2)


def select_backend(organism_type: str, style_mode: str, backend_override: str | None = None) -> tuple[str, str]:
    """
    Returns (backend_name, reason_string).
    Explicit override always wins, per spec.
    """
    if backend_override:
        if backend_override not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend_override '{backend_override}'. Valid: {sorted(VALID_BACKENDS)}")
        return backend_override, f"User override → {backend_override}"

    table = load_routing_table()
    organism_rules = table.get(organism_type, {})

    backend = organism_rules.get(style_mode) or organism_rules.get("*")
    if backend:
        reason = BACKEND_INFO.get(backend, {}).get("reason", f"Routed via ({organism_type}, {style_mode})")
        return backend, reason

    return FALLBACK_BACKEND, f"No routing match for ({organism_type}, {style_mode}) → fast fallback"
