"""Single source of truth for training types, biases, models.

Loads from `experiments.toml` for global config and from `viz_registration:`
blocks in `scripts/tinker_training/experiment_configs/*.yaml` for
per-experiment training types. Frozen dataclasses; no module-level mutation.
"""
from __future__ import annotations

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


def _register_yaml_viz(reg: Registry, vr: dict) -> None:
    """Register a single `viz_registration:` block into `reg` in place.

    Mints two training_type entries from one block: the trained variant
    (keyed on `dir_suffix`) and an optional control (`{dir_suffix}-ctrl`)
    when `control_color` is supplied. Both share `training_biases` from
    the block; the control's `control_for` points back at the trained.
    Skips silently when the trained key already exists in `reg`.
    """
    dir_suffix = vr.get("dir_suffix")
    if not isinstance(dir_suffix, str) or not dir_suffix:
        return
    if dir_suffix in reg.training_types:
        return  # explicit TOML/JSON entry takes precedence

    method, scale, _ = _classify_method(dir_suffix)
    biases = frozenset(vr.get("training_biases") or [])
    reg.training_types[dir_suffix] = TrainingTypeInfo(
        key=dir_suffix,
        display_name=vr.get("display_name", dir_suffix),
        color=vr.get("color", "#888888"),
        hatch=vr.get("hatch", ""),
        edgecolor=vr.get("edgecolor", "black"),
        method=method,
        data_scale=scale,
        is_control=False,
        training_biases=biases,
        dir_aliases=(dir_suffix,),
    )
    reg.dir_to_training_type[dir_suffix] = dir_suffix
    if dir_suffix not in reg.training_type_order:
        reg.training_type_order.append(dir_suffix)

    # Control variant: only created when YAML supplies a control_color,
    # signalling the experiment runs a paired control.
    ctrl_color = vr.get("control_color")
    if ctrl_color:
        ctrl_key = f"{dir_suffix}-ctrl"
        if ctrl_key not in reg.training_types:
            reg.training_types[ctrl_key] = TrainingTypeInfo(
                key=ctrl_key,
                display_name=f"{vr.get('display_name', dir_suffix)} Control",
                color=ctrl_color,
                hatch="",
                edgecolor=vr.get("color", "black"),
                method=method,
                data_scale=scale,
                is_control=True,
                control_for=dir_suffix,
                training_biases=frozenset(),  # controls don't carry trained biases
                dir_aliases=(ctrl_key,),
            )
            reg.dir_to_training_type[ctrl_key] = ctrl_key
            if ctrl_key not in reg.training_type_order:
                reg.training_type_order.append(ctrl_key)


def _build_registry(toml_path: Path) -> Registry:
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

    # ── 2. Scan experiment-config YAMLs for viz_registration blocks ─────────
    # Each block self-describes a training run's display style. Registering
    # new methods becomes "edit the experiment YAML"; no separate registry
    # bookkeeping. Existing TOML entries win.
    yaml_dir = _PKG_DIR.parent / "scripts" / "tinker_training" / "experiment_configs"
    if yaml_dir.exists():
        try:
            import yaml  # PyYAML
        except ImportError:
            yaml = None
        if yaml is not None:
            for yml_path in sorted(yaml_dir.glob("*.yaml")):
                try:
                    with open(yml_path) as f:
                        cfg = yaml.safe_load(f) or {}
                except (yaml.YAMLError, OSError):
                    continue
                vr = cfg.get("viz_registration")
                if not isinstance(vr, dict):
                    continue
                _register_yaml_viz(reg, vr)

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
REGISTRY = _build_registry(toml_path=_PKG_DIR / "experiments.toml")
