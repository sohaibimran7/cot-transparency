"""Single source of truth for training types, biases, models.

Loads from `experiments.toml` (preferred) and `model_registry.json` (legacy,
back-compat). Frozen dataclasses; no module-level mutation.

All callers should access state through `REGISTRY` rather than re-importing
constants from `visualize_results.py`. The legacy module re-exports its dicts
from this registry for back-compat with code that hasn't been migrated yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class TrainingTypeInfo:
    """One training type — one bar color, one display name, one bias-set."""
    key: str                       # canonical key (e.g. "rlct_da_aw0")
    display_name: str              # legend label
    color: str                     # bar facecolor
    hatch: str = ""
    edgecolor: str = "black"
    method: str = "other"          # "base" | "bct" | "rlct" | "vft" | "other"
    data_scale: str = "da"         # "da" | "dawfs" — drives panel grouping
    is_control: bool = False
    control_for: str | None = None  # if this is a control, the trained type
                                    # it pairs with (drives panel placement).
                                    # Explicit in TOML; auto-filled from the
                                    # `_ctrl`/`_control` suffix as fallback.
    training_biases: frozenset[str] = field(default_factory=frozenset)
    aggregate_group: str | None = None
    dir_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelInfo:
    key: str                       # "llama" | "gpt" | "gpt-oss-20b" | …
    display_name: str
    dir_prefix: str                # e.g. "llama-"
    prompt_styles: tuple[str, ...] = ("no_cot",)


@dataclass(frozen=True)
class BiasInfo:
    key: str
    display_name: str
    publication_label: str         # may include \n


@dataclass
class Registry:
    """All registry data. One immutable instance is built at import time."""
    training_types: dict[str, TrainingTypeInfo] = field(default_factory=dict)
    models: dict[str, ModelInfo] = field(default_factory=dict)
    biases: dict[str, BiasInfo] = field(default_factory=dict)
    dir_to_training_type: dict[str, str | None] = field(default_factory=dict)
    training_type_order: list[str] = field(default_factory=list)
    aggregate_order: list[str] = field(default_factory=list)
    aggregate_names: dict[str, str] = field(default_factory=dict)


def _classify_method(key: str) -> tuple[str, str, bool]:
    """Heuristic: derive (method, data_scale, is_control) from a training_type key."""
    is_control = "control" in key or "ctrl" in key or key.endswith("_ctrl")
    if "da_wfs" in key or "dawfs" in key:
        scale = "dawfs"
    else:
        scale = "da"
    if key == "base" or key.startswith("base"):
        return "base", scale, False
    if "bct" in key:
        return "bct", scale, is_control
    if "rlct" in key or key.startswith("rl_") or key.startswith("rl-"):
        return "rlct", scale, is_control
    if "vft" in key:
        return "vft", scale, is_control
    return "other", scale, is_control


def _model_family_for_prefix(prefix: str) -> str:
    """Strip trailing dash to get the model family key."""
    return prefix.rstrip("-")


def _build_registry(toml_path: Path, json_path: Path) -> Registry:
    reg = Registry()

    # ── 1. Load experiments.toml — the new canonical source ─────────────────
    if toml_path.exists():
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)
        for key, info in (toml_data.get("biases") or {}).items():
            reg.biases[key] = BiasInfo(
                key=key,
                display_name=info["display_name"],
                publication_label=info.get("publication_label", info["display_name"]),
            )
        for key, info in (toml_data.get("models") or {}).items():
            reg.models[key] = ModelInfo(
                key=key,
                display_name=info["display_name"],
                dir_prefix=info["dir_prefix"],
                prompt_styles=tuple(info.get("prompt_styles", ["no_cot"])),
            )
        for key, info in (toml_data.get("training_types") or {}).items():
            method, scale, is_control = _classify_method(key)
            method = info.get("method", method)
            scale = info.get("data_scale", scale)
            is_control = info.get("is_control", is_control)
            reg.training_types[key] = TrainingTypeInfo(
                key=key,
                display_name=info.get("display_name", key),
                color=info.get("color", "#888888"),
                hatch=info.get("hatch", ""),
                edgecolor=info.get("edgecolor", "black"),
                method=method,
                data_scale=scale,
                is_control=is_control,
                control_for=info.get("control_for"),
                training_biases=frozenset(info.get("training_biases", [])),
                aggregate_group=info.get("aggregate_group"),
                dir_aliases=tuple(info.get("dir_aliases", [])),
            )
            for alias in info.get("dir_aliases", []):
                reg.dir_to_training_type[alias] = key
        reg.training_type_order = list(toml_data.get("ordering", {})
                                       .get("training_types", []))
        reg.aggregate_order = list(toml_data.get("ordering", {})
                                   .get("aggregates", []))
        reg.aggregate_names = dict(toml_data.get("aggregate_names") or {})

    # ── 2. Merge legacy model_registry.json — extends with new entries ──────
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        # Model prefixes → ModelInfo if not already known
        for prefix in data.get("model_prefixes", []):
            family = _model_family_for_prefix(prefix)
            if family not in reg.models:
                # Legacy: assume cot+no_cot for llama, no_cot otherwise.
                styles = ("cot", "no_cot") if family == "llama" else ("no_cot",)
                reg.models[family] = ModelInfo(
                    key=family,
                    display_name=family.upper(),
                    dir_prefix=prefix,
                    prompt_styles=styles,
                )
        for dir_alias, info in data.get("models", {}).items():
            tt_key = info.get("training_type", dir_alias.replace("-", "_"))
            reg.dir_to_training_type[dir_alias] = tt_key
            if tt_key not in reg.training_types:
                method, scale, is_control = _classify_method(tt_key)
                reg.training_types[tt_key] = TrainingTypeInfo(
                    key=tt_key,
                    display_name=info.get("display_name", tt_key),
                    color=info.get("color", "#888888"),
                    hatch=info.get("hatch", ""),
                    edgecolor=info.get("edgecolor", "black"),
                    method=method,
                    data_scale=scale,
                    is_control=is_control,
                    training_biases=frozenset(info.get("training_biases", [])),
                    dir_aliases=(dir_alias,),
                )
            if tt_key not in reg.training_type_order:
                reg.training_type_order.append(tt_key)

    # ── 3. Auto-fill control_for from suffix as a convenience fallback ──────
    # Entries that already set control_for explicitly are left untouched. For
    # the rest, if the key ends in `_ctrl` or `_control` and the stripped name
    # matches an existing training_type, link them. This is the only place
    # control linkage is derived from a name suffix; downstream code reads
    # `info.control_for` directly and doesn't slice strings.
    _CONTROL_SUFFIXES = ("_ctrl", "_control")
    for key, info in list(reg.training_types.items()):
        if info.control_for is not None:
            continue
        if not info.is_control:
            continue
        for suffix in _CONTROL_SUFFIXES:
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                if base in reg.training_types:
                    reg.training_types[key] = TrainingTypeInfo(
                        **{**info.__dict__, "control_for": base}
                    )
                break

    return reg


def training_type_info(key: str) -> TrainingTypeInfo:
    info = REGISTRY.training_types.get(key)
    if info is None:
        # Synthesize a fallback so unknown training types render with neutral grey
        method, scale, is_control = _classify_method(key)
        return TrainingTypeInfo(
            key=key, display_name=key, color="#888888",
            method=method, data_scale=scale, is_control=is_control,
        )
    return info


def model_info(key: str) -> ModelInfo:
    info = REGISTRY.models.get(key)
    if info is None:
        return ModelInfo(key=key, display_name=key.upper(), dir_prefix=f"{key}-")
    return info


_PKG_DIR = Path(__file__).parent.parent
REGISTRY = _build_registry(
    toml_path=_PKG_DIR / "experiments.toml",
    json_path=_PKG_DIR / "model_registry.json",
)
