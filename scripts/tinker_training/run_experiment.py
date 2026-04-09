"""
End-to-end experiment runner for consistency training.

Chains data generation, training, evaluation, and analysis stages
using a YAML config file. Supports stage skipping and resumption.

Usage:
    # Full pipeline
    python scripts/tinker_training/run_experiment.py experiments/my_experiment.yaml

    # Skip to evaluation (uses checkpoint from state or config)
    python scripts/tinker_training/run_experiment.py experiments/my_experiment.yaml --start-from evaluation

    # Run only specific stages
    python scripts/tinker_training/run_experiment.py experiments/my_experiment.yaml --stages evaluation,analysis

    # Override checkpoint for eval-only run
    python scripts/tinker_training/run_experiment.py experiments/my_experiment.yaml \
        --start-from evaluation --checkpoint "tinker://..."

    # Preview commands without executing
    python scripts/tinker_training/run_experiment.py experiments/my_experiment.yaml --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent.parent.parent

STAGES = ["data_generation", "data_preparation", "training", "evaluation", "analysis"]

# Thread-safe output with optional per-experiment prefix
_print_lock = threading.Lock()
_thread_local = threading.local()


def _write(text: str) -> None:
    """Write text to stdout with optional thread-local experiment prefix."""
    prefix = getattr(_thread_local, "prefix", "")
    with _print_lock:
        if prefix:
            for line in text.splitlines(keepends=True):
                if line.strip():
                    sys.stdout.write(f"[{prefix}] {line}")
                else:
                    sys.stdout.write(line)
        else:
            sys.stdout.write(text)
        sys.stdout.flush()


def _print(*args, **kwargs) -> None:
    """Thread-safe print with optional experiment prefix."""
    import io
    buf = io.StringIO()
    kwargs.pop("file", None)  # Always write to our _write
    print(*args, file=buf, **kwargs)
    _write(buf.getvalue())

STAGE_ALIASES = {
    "data_gen": "data_generation",
    "datagen": "data_generation",
    "data_prep": "data_preparation",
    "dataprep": "data_preparation",
    "train": "training",
    "eval": "evaluation",
    "analyze": "analysis",
    "viz": "analysis",
}


def resolve_stage(name: str) -> str:
    """Resolve a stage name or alias to the canonical stage name."""
    lower = name.lower().strip()
    return STAGE_ALIASES.get(lower, lower)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load and validate a YAML experiment config."""
    with open(path) as f:
        config = yaml.safe_load(f)

    if not config.get("name"):
        raise ValueError("Config must have a 'name' field")
    if not config.get("model"):
        raise ValueError("Config must have a 'model' field")

    return config


def config_hash(config: dict) -> str:
    """Compute a stable hash of the config for change detection."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def sanitize_model_name(model: str) -> str:
    """Convert model name to filesystem-safe directory name.

    Matches the convention in generate_bct_from_test.py.
    """
    name = model.split("/")[-1].lower()
    return name.replace(".", "-")


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def state_dir(config: dict) -> Path:
    """Return the state directory for an experiment."""
    return _PROJECT_ROOT / "experiments" / config["name"]


def state_path(config: dict) -> Path:
    return state_dir(config) / "state.json"


def load_state(config: dict) -> dict:
    """Load existing pipeline state, or return a fresh skeleton."""
    p = state_path(config)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "experiment": config["name"],
        "config_hash": config_hash(config),
        "stages": {},
    }


def save_state(config: dict, st: dict) -> None:
    """Atomically write pipeline state."""
    d = state_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=d, suffix=".json", delete=False
    )
    json.dump(st, tmp, indent=2, default=str)
    tmp.close()
    os.replace(tmp.name, state_path(config))


def stage_status(st: dict, stage: str) -> str:
    return st.get("stages", {}).get(stage, {}).get("status", "pending")


def stage_outputs(st: dict, stage: str) -> dict:
    return st.get("stages", {}).get(stage, {}).get("outputs", {})


def mark_stage(st: dict, stage: str, status: str, **extra) -> None:
    if stage not in st.setdefault("stages", {}):
        st["stages"][stage] = {}
    st["stages"][stage]["status"] = status
    ts_key = "started_at" if status == "running" else "completed_at"
    if status in ("running", "completed", "failed"):
        st["stages"][stage][ts_key] = datetime.now(timezone.utc).isoformat()
    for k, v in extra.items():
        st["stages"][stage][k] = v


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _subprocess_env() -> dict[str, str]:
    """Build environment for subprocesses with project root on PYTHONPATH."""
    env = os.environ.copy()
    root = str(_PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}:{existing}" if existing else root
    return env


def run_stage_command(
    cmd: list[str],
    stage_name: str,
    config: dict,
    dry_run: bool = False,
) -> tuple[int, str]:
    """Run a subprocess, tee output to terminal and capture it.

    Returns (return_code, captured_stdout).
    """
    cmd_str = " \\\n    ".join(cmd)
    _print(f"\n{'='*60}")
    _print(f"  Stage: {stage_name}")
    _print(f"{'='*60}")
    _print(f"Command:\n  {cmd_str}\n")

    if dry_run:
        _print("  [dry-run] Skipping execution")
        return 0, ""

    # Log output to file as well
    log_dir = state_dir(config) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage_name}.log"

    captured: list[str] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env=_subprocess_env(),
    )

    with open(log_file, "w") as lf:
        for line in proc.stdout:
            _write(line)
            lf.write(line)
            captured.append(line)

    proc.wait()
    return proc.returncode, "".join(captured)


def run_parallel_commands(
    commands: list[tuple[list[str], str]],
    stage_name: str,
    config: dict,
    dry_run: bool = False,
) -> list[tuple[int, str]]:
    """Run multiple commands in parallel with labeled output.

    Args:
        commands: List of (cmd, label) tuples.
        stage_name: Stage name for headers and log files.
        config: Experiment config (for log directory).
        dry_run: Print commands without executing.

    Returns:
        List of (return_code, captured_output) per command.
    """
    # Print all commands
    for cmd, label in commands:
        cmd_str = " \\\n    ".join(cmd)
        _print(f"\n{'='*60}")
        _print(f"  {stage_name} [{label}]")
        _print(f"{'='*60}")
        _print(f"Command:\n  {cmd_str}\n")

    if dry_run:
        for _, label in commands:
            _print(f"  [dry-run] Skipping {label}")
        return [(0, "") for _ in commands]

    if len(commands) == 1:
        # Single command — run without label prefix for cleaner output
        cmd, label = commands[0]
        log_dir = state_dir(config) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{stage_name}_{label}.log"

        captured: list[str] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_PROJECT_ROOT), env=_subprocess_env(),
        )
        with open(log_file, "w") as lf:
            for line in proc.stdout:
                _write(line)
                lf.write(line)
                captured.append(line)
        proc.wait()
        return [(proc.returncode, "".join(captured))]

    # Multiple commands — run in parallel with label prefixes
    log_dir = state_dir(config) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    captures: list[list[str]] = []
    log_files = []
    for cmd, label in commands:
        lf = open(log_dir / f"{stage_name}_{label}.log", "w")
        log_files.append(lf)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_PROJECT_ROOT), env=_subprocess_env(),
        )
        procs.append(proc)
        captures.append([])

    # Capture parent thread's prefix so child reader threads inherit it
    parent_prefix = getattr(_thread_local, "prefix", "")

    def _reader(proc, label, capture, log_file):
        _thread_local.prefix = parent_prefix
        for line in proc.stdout:
            tagged = f"[{label}] {line}"
            _write(tagged)
            log_file.write(line)
            capture.append(line)

    threads = []
    for i, (proc, (_, label)) in enumerate(zip(procs, commands)):
        t = threading.Thread(target=_reader, args=(proc, label, captures[i], log_files[i]))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    results = []
    for i, proc in enumerate(procs):
        proc.wait()
        log_files[i].close()
        results.append((proc.returncode, "".join(captures[i])))

    return results


# ---------------------------------------------------------------------------
# Stage: data_generation
# ---------------------------------------------------------------------------

def _get_data_gen_steps(config: dict) -> list[dict]:
    """Normalize data_generation config to a list of step dicts."""
    dg = config.get("data_generation")
    if dg is None:
        return []
    if isinstance(dg, list):
        return dg
    return [dg]  # Single dict → list of one


def _build_single_data_gen_cmd(config: dict, step: dict) -> list[str]:
    """Build command for a single data generation step.

    Generic: passes all args as CLI flags without knowing the script's internals.
    List values become positional args after the flag. Bool true values become bare flags.
    """
    script = step.get("script", "generate_bct_from_test")
    args = step.get("args", {})

    # Resolve script path: bare names map to scripts/tinker_training/
    if "/" not in script and not script.endswith(".py"):
        script_path = f"scripts/tinker_training/{script}.py"
    else:
        script_path = script

    cmd = [sys.executable, script_path]

    # Add --model unless disabled
    if step.get("pass_model", True):
        cmd += ["--model", config["model"]]

    # Pass all args generically as CLI flags
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            cmd += [flag] + [str(v) for v in value]
        else:
            cmd += [flag, str(value)]

    return cmd


def parse_data_gen_output(output: str) -> dict:
    """Parse data generation stdout for output file paths.

    Uses generic patterns to capture any JSONL file paths from "Saved N samples to ..."
    and directory paths from "Output dir[s]:" lines.
    """
    results = {}
    saved_files: list[str] = []

    for line in output.splitlines():
        # Capture any "Saved N samples to <path>" lines
        m = re.search(r"Saved \d+ samples to (.+\.jsonl)", line)
        if m:
            path = m.group(1).strip()
            saved_files.append(path)
            # Derive key from filename: bct_cot.jsonl -> bct_path, control_cot.jsonl -> control_path
            stem = Path(path).stem  # e.g. "bct_cot", "control_cot", "vft_cot"
            key = stem.replace("_cot", "_path").replace("_non_cot", "_non_cot_path")
            results[key] = path

        # Capture directory paths from common output patterns
        m = re.search(r"(?:Output dir|Control|Train):\s*(.+)", line)
        if m:
            dir_path = m.group(1).strip()
            if "Control:" in line and "control_dir" not in results:
                results["control_dir"] = dir_path
            elif "Train:" in line and "train_dir" not in results:
                results["train_dir"] = dir_path
            elif "Output dir" in line:
                results.setdefault("output_dir", dir_path)

    # Store all saved files for generic access
    if saved_files:
        results["data_files"] = saved_files

    return results


# ---------------------------------------------------------------------------
# Stage: data_preparation
# ---------------------------------------------------------------------------

def build_data_prep_cmd(config: dict, st: dict) -> list[str] | None:
    """Build command for data preparation (optional stage)."""
    dp = config.get("data_preparation", {})
    if not dp.get("enabled", False):
        return None

    args = dp.get("args", {})
    cmd = [sys.executable, "scripts/prepare_datasets.py"]

    # Resolve inputs - "auto:*" references use data_gen outputs
    dg_outputs = stage_outputs(st, "data_generation")
    raw_inputs = args.get("inputs", [])
    for inp in raw_inputs:
        if inp.startswith("auto:"):
            # auto:bct_cot, auto:instruct:half etc.
            parts = inp.split(":")
            key = parts[1]  # e.g. bct_cot, instruct
            limit = parts[2] if len(parts) > 2 else None
            # Derive path from data_gen output dirs
            train_dir = dg_outputs.get("train_dir")
            if not train_dir:
                path = f"<auto:{key}>"
            else:
                path = f"{train_dir}/{key}.jsonl"
            spec = f"{path}:{limit}" if limit else path
            cmd += ["--input", spec]
        else:
            cmd += ["--input", inp]

    if "output" in args:
        cmd += ["--output", str(args["output"])]
    if args.get("clean"):
        cmd.append("--clean")
    if args.get("shuffle"):
        cmd.append("--shuffle")
    if "seed" in args:
        cmd += ["--seed", str(args["seed"])]

    return cmd


# ---------------------------------------------------------------------------
# Stage: training
# ---------------------------------------------------------------------------

def _resolve_training_data(config: dict, st: dict, is_control: bool) -> list[str]:
    """Auto-resolve training data files from data_generation outputs.

    Uses generic output keys: any *_path ending in .jsonl from data_gen,
    filtered by control vs non-control convention (control_* files for control).
    """
    dp_outputs = stage_outputs(st, "data_preparation")
    dg_outputs = stage_outputs(st, "data_generation")

    if dp_outputs.get("output_path") and not is_control:
        return [dp_outputs["output_path"]]

    # Use data_files list if available (all saved JSONL paths from data gen)
    all_files = dg_outputs.get("data_files", [])
    if all_files:
        if is_control:
            files = [f for f in all_files if "control" in Path(f).stem]
        else:
            files = [f for f in all_files if "control" not in Path(f).stem]
        if files:
            return files

    # Fallback: use individual keys
    files = []
    if is_control:
        if dg_outputs.get("control_path"):
            files.append(dg_outputs["control_path"])
    else:
        # Collect all non-control *_path keys
        for key, val in dg_outputs.items():
            if key.endswith("_path") and "control" not in key and isinstance(val, str) and val.endswith(".jsonl"):
                files.append(val)

    if files:
        return files

    tag = "control-data" if is_control else "data-generation"
    return [f"<auto:resolved-from-{tag}>"]


def build_training_cmd(
    config: dict, st: dict, is_control: bool = False, seed: int | None = None,
) -> list[str]:
    """Build the command for a single training run.

    Args:
        is_control: If True, build the control variant (SFT uses control data, RL adds --control).
        seed: If set, append -s{seed} to run_name and pass --seed to the training script.
    """
    tr = config.get("training", {})
    method = tr.get("method", "sft")
    args = tr.get("args", {})

    run_name = str(args.get("run_name", "default"))
    if seed is not None:
        run_name = f"{run_name}-s{seed}"
    if is_control:
        run_name = f"{run_name}-ctrl"

    if method == "sft":
        cmd = [sys.executable, "scripts/tinker_training/train_sft.py"]
        cmd += ["--model", config["model"]]

        # Data files - resolve from prior stages or explicit config
        data_files = args.get("data", [])
        if not data_files or data_files == "auto":
            data_files = _resolve_training_data(config, st, is_control)

        if isinstance(data_files, str):
            data_files = [data_files]
        if data_files:
            cmd += ["--data"] + [str(f) for f in data_files]

        # Interleave: explicit flag or data_mode config
        data_mode = tr.get("data_mode", "concat")
        if args.get("interleave") or data_mode == "interleave":
            cmd.append("--interleave")

        # Naming
        cmd += ["--experiment-name", str(args.get("experiment_name", "sft_experiment"))]
        cmd += ["--run-name", run_name]

        # Hyperparams
        for key in ("lr", "batch_size", "epochs", "lora_rank", "save_every", "skip_near_final"):
            if key in args:
                cmd += [f"--{key.replace('_', '-')}", str(args[key])]

        if seed is not None:
            cmd += ["--seed", str(seed)]

        if args.get("save_state"):
            cmd.append("--save-state")

        if tr.get("resume_from") or tr.get("checkpoint"):
            resume = tr.get("resume_from") or tr.get("checkpoint")
            cmd += ["--resume-from", str(resume)]

        # Always skip confirmation
        cmd.append("-y")

    elif method == "rl":
        cmd = [sys.executable, "scripts/tinker_training/train_rl.py"]
        cmd += ["--model", config["model"]]

        # Required args
        cmd += ["--experiment-name", str(args.get("experiment_name", "rl_experiment"))]
        cmd += ["--run-name", run_name]

        if "bias_types" in args:
            bt = args["bias_types"]
            cmd += ["--bias-types", bt if isinstance(bt, str) else ",".join(bt)]
        if "datasets" in args:
            ds = args["datasets"]
            cmd += ["--datasets", ds if isinstance(ds, str) else ",".join(ds)]

        # Hyperparams
        for key in (
            "n_datapoints", "lr", "lr_schedule", "lora_rank", "kl_coef",
            "anchor_weight", "anchor_model", "loss_fn",
            "n_ref_rollouts", "n_train_rollouts", "n_consistency_rollouts",
            "n_anchor_rollouts", "temperature", "max_new_tokens",
            "n_epochs", "batch_size", "gradient_accumulation_steps",
            "refresh_every", "checkpoint_every",
        ):
            if key in args:
                cmd += [f"--{key.replace('_', '-')}", str(args[key])]

        if seed is not None:
            cmd += ["--seed", str(seed)]

        if args.get("save_state"):
            cmd.append("--save-state")
        if is_control or args.get("control"):
            cmd.append("--control")

        if tr.get("resume_from") or tr.get("checkpoint"):
            resume = tr.get("resume_from") or tr.get("checkpoint")
            cmd += ["--resume-from", str(resume)]
            if args.get("resume_with_optimizer"):
                cmd.append("--resume-with-optimizer")

        # Always skip confirmation
        cmd.append("-y")

    else:
        raise ValueError(f"Unknown training method: {method}")

    return cmd


def build_training_cmds(config: dict, st: dict) -> list[tuple[list[str], str]]:
    """Build training commands. Returns list of (cmd, label) tuples.

    Supports include_control (parallel main + control), control_only, and
    multi-seed (training.seeds list). Seeds and control are crossed:
    e.g. seeds=[0,42] + include_control produces 4 parallel runs.
    For data_mode=sequential, returns commands for each data step (run serially by caller).
    """
    tr = config.get("training", {})
    control_only = tr.get("control_only", False)
    include_control = tr.get("include_control", False) or control_only
    seeds = tr.get("seeds")  # None or list[int]

    cmds = []
    seed_list = seeds if seeds else [None]

    for seed in seed_list:
        if not control_only:
            label = f"main-s{seed}" if seed is not None else "main"
            cmds.append((build_training_cmd(config, st, is_control=False, seed=seed), label))
        if include_control:
            label = f"control-s{seed}" if seed is not None else "control"
            cmds.append((build_training_cmd(config, st, is_control=True, seed=seed), label))
    return cmds


def build_sequential_training_cmds(
    config: dict, st: dict, data_paths: list[str],
) -> list[tuple[list[str], str]]:
    """Build sequential training commands for data_mode=sequential.

    Each data path gets its own training run, resuming from the previous checkpoint.
    Returns list of (cmd, label) to run IN ORDER (not parallel).
    """
    tr = config.get("training", {})
    args = tr.get("args", {})
    base_run_name = str(args.get("run_name", "default"))

    cmds = []
    for i, data_path in enumerate(data_paths):
        step_name = Path(data_path).stem  # e.g. "vft_cot", "bct_cot"
        run_name = f"{base_run_name}-{step_name}" if len(data_paths) > 1 else base_run_name

        cmd = [sys.executable, "scripts/tinker_training/train_sft.py"]
        cmd += ["--model", config["model"]]
        cmd += ["--data", data_path]
        cmd += ["--experiment-name", str(args.get("experiment_name", "sft_experiment"))]
        cmd += ["--run-name", run_name]

        for key in ("lr", "batch_size", "epochs", "lora_rank", "save_every", "skip_near_final"):
            if key in args:
                cmd += [f"--{key.replace('_', '-')}", str(args[key])]

        if args.get("save_state"):
            cmd.append("--save-state")

        # First step: use resume_from from config. Subsequent: filled in at runtime.
        if i == 0:
            resume = tr.get("resume_from") or tr.get("checkpoint")
            if resume:
                cmd += ["--resume-from", str(resume)]
        else:
            # Placeholder — caller will replace with previous step's checkpoint
            cmd += ["--resume-from", "<auto:previous-checkpoint>"]

        cmd.append("-y")
        cmds.append((cmd, f"seq-{step_name}"))

    return cmds


def parse_training_output(output: str) -> dict:
    """Parse training stdout for checkpoint path."""
    results = {}
    for line in output.splitlines():
        m = re.search(r"Final checkpoint:\s*(.+)", line)
        if m:
            results["checkpoint"] = m.group(1).strip()
    return results


# ---------------------------------------------------------------------------
# Stage: evaluation
# ---------------------------------------------------------------------------

def _eval_name(config: dict, suffix: str = "") -> str:
    """Build eval model name with model prefix and optional suffix."""
    tr_args = config.get("training", {}).get("args", {})
    ev_args = config.get("evaluation", {}).get("args", {})
    name = ev_args.get("name") or tr_args.get("run_name") or "model"
    model_prefix = sanitize_model_name(config["model"]).split("-")[0]
    if not name.lower().startswith(model_prefix):
        name = f"{model_prefix}-{name}"
    return f"{name}{suffix}"


def _build_single_eval_cmd(
    config: dict, args: dict, checkpoint: str | None, eval_name: str,
) -> list[str]:
    """Build a single evaluation command."""
    cmd = [sys.executable, "-m", "sycophancy_eval_inspect.run_tinker_evals"]

    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]

    cmd += ["--base-model", config["model"]]
    cmd += ["--name", eval_name]

    if "prompt_styles" in args:
        ps = args["prompt_styles"]
        cmd += ["--prompt-styles", ps if isinstance(ps, str) else ",".join(ps)]
    if "bias_types" in args:
        bt = args["bias_types"]
        cmd += ["--bias-types", bt if isinstance(bt, str) else ",".join(bt)]
    if "datasets" in args:
        ds = args["datasets"]
        cmd += ["--datasets", ds if isinstance(ds, str) else ",".join(ds)]
    if "log_dir" in args:
        cmd += ["--log-dir", str(args["log_dir"])]
    if "max_tokens" in args:
        cmd += ["--max-tokens", str(args["max_tokens"])]
    if "limit" in args:
        cmd += ["--limit", str(args["limit"])]
    if "hash_file" in args:
        cmd += ["--hash-file", str(args["hash_file"])]
    else:
        # Auto-detect existing hash file to avoid FileExistsError
        log_dir = args.get("log_dir", "sycophancy_eval_inspect/logs/tinker_evals")
        hash_path = Path(log_dir) / "common_hashes.json"
        if hash_path.exists():
            cmd += ["--hash-file", str(hash_path)]
    if args.get("biased_only"):
        cmd.append("--biased-only")
    if args.get("unbiased_only"):
        cmd.append("--unbiased-only")

    return cmd


def build_eval_cmds(
    config: dict, st: dict, checkpoint_override: str | None,
) -> list[tuple[list[str], str]]:
    """Build evaluation commands. Returns list of (cmd, label) tuples.

    Supports include_base (also eval base model) and base_only.
    Automatically includes control checkpoint eval if one was trained.
    """
    ev = config.get("evaluation", {})
    tr = config.get("training", {})
    args = ev.get("args", {})
    base_only = ev.get("base_only", False)
    include_base = ev.get("include_base", False) or base_only

    cmds = []

    # Base model eval (no checkpoint)
    if include_base:
        model_prefix = sanitize_model_name(config["model"]).split("-")[0]
        base_name = f"{model_prefix}-base"
        cmd = _build_single_eval_cmd(config, args, checkpoint=None, eval_name=base_name)
        cmds.append((cmd, "base"))

    if not base_only:
        training_outputs = stage_outputs(st, "training")
        seeds = tr.get("seeds")

        if seeds and not checkpoint_override:
            # Multi-seed: eval each seed's checkpoint
            for seed in seeds:
                main_key = f"main-s{seed}_checkpoint"
                ckpt = training_outputs.get(main_key)
                if ckpt:
                    cmd = _build_single_eval_cmd(
                        config, args, checkpoint=ckpt,
                        eval_name=_eval_name(config, suffix=f"-s{seed}"),
                    )
                    cmds.append((cmd, f"main-s{seed}"))

                ctrl_key = f"control-s{seed}_checkpoint"
                ctrl_ckpt = training_outputs.get(ctrl_key)
                if ctrl_ckpt:
                    cmd = _build_single_eval_cmd(
                        config, args, checkpoint=ctrl_ckpt,
                        eval_name=_eval_name(config, suffix=f"-s{seed}-ctrl"),
                    )
                    cmds.append((cmd, f"control-s{seed}"))
        else:
            # Single-seed: existing logic
            checkpoint = (
                checkpoint_override
                or training_outputs.get("checkpoint")
                or tr.get("checkpoint")
            )
            if checkpoint:
                cmd = _build_single_eval_cmd(
                    config, args, checkpoint=checkpoint, eval_name=_eval_name(config),
                )
                cmds.append((cmd, "main"))

            # Control checkpoint (auto-detected from training outputs)
            control_checkpoint = training_outputs.get("control_checkpoint")
            if control_checkpoint:
                cmd = _build_single_eval_cmd(
                    config, args, checkpoint=control_checkpoint,
                    eval_name=_eval_name(config, suffix="-ctrl"),
                )
                cmds.append((cmd, "control"))

    return cmds


# ---------------------------------------------------------------------------
# Stage: analysis
# ---------------------------------------------------------------------------

def build_analysis_cmd(config: dict, st: dict) -> list[str]:
    """Build the command for analysis/visualization."""
    an = config.get("analysis", {})
    args = an.get("args", {})

    cmd = [sys.executable, "-m", "sycophancy_eval_inspect.visualize_results"]

    # Resolve log dir
    ev_args = config.get("evaluation", {}).get("args", {})
    log_dir = args.get("log_dir") or ev_args.get("log_dir", "sycophancy_eval_inspect/logs/tinker_evals")
    cmd += ["--log-dir", str(log_dir)]

    if args.get("bir"):
        cmd.append("--bir")
    if args.get("ba"):
        cmd.append("--ba")
    if args.get("plot"):
        cmd.append("--plot")
    if "output_dir" in args:
        cmd += ["--output-dir", str(args["output_dir"])]
    if "model" in args:
        models = args["model"] if isinstance(args["model"], list) else [args["model"]]
        for m in models:
            cmd += ["--model", m]
    if "prompt_style" in args:
        styles = args["prompt_style"] if isinstance(args["prompt_style"], list) else [args["prompt_style"]]
        for s in styles:
            cmd += ["--prompt-style", s]
    if args.get("summary"):
        cmd.append("--summary")
    if args.get("common_questions"):
        cmd.append("--common-questions")
    if args.get("variance_across"):
        cmd += ["--variance-across", str(args["variance_across"])]

    return cmd


# ---------------------------------------------------------------------------
# Model registry integration
# ---------------------------------------------------------------------------

def _lighten_color(hex_color: str, factor: float = 0.4) -> str:
    """Lighten a hex color by mixing with white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def update_model_registry(config: dict) -> None:
    """Update model_registry.json with viz_registration from config."""
    viz = config.get("viz_registration")
    if not viz:
        return

    registry_path = _PROJECT_ROOT / "sycophancy_eval_inspect" / "model_registry.json"

    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"model_prefixes": [], "models": {}}

    # Ensure model prefix is registered
    model_prefix = sanitize_model_name(config["model"]).split("-")[0] + "-"
    if model_prefix not in registry.get("model_prefixes", []):
        registry.setdefault("model_prefixes", []).append(model_prefix)

    suffix = viz.get("dir_suffix")
    if suffix:
        training_type = viz.get("training_type", suffix.replace("-", "_"))
        registry["models"][suffix] = {
            "training_type": training_type,
            "display_name": viz.get("display_name", training_type),
            "color": viz.get("color", "#888888"),
            "hatch": viz.get("hatch", ""),
            "edgecolor": viz.get("edgecolor", "black"),
            "training_biases": viz.get("training_biases", []),
        }

        # Auto-register control variant if include_control is set
        tr = config.get("training", {})
        if tr.get("include_control") or tr.get("control_only"):
            ctrl_suffix = f"{suffix}-ctrl"
            if ctrl_suffix not in registry["models"]:
                ctrl_color = viz.get("control_color", _lighten_color(viz.get("color", "#888888")))
                registry["models"][ctrl_suffix] = {
                    "training_type": f"{training_type}_ctrl",
                    "display_name": f"{viz.get('display_name', training_type)} Control",
                    "color": ctrl_color,
                    "hatch": "//",
                    "edgecolor": "black",
                    "training_biases": [],
                }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    _print(f"Updated model registry: {registry_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(config: dict, stages_to_run: list[str], st: dict, checkpoint_override: str | None) -> None:
    """Validate that required inputs exist for each stage to run."""
    tr = config.get("training", {})
    tr_args = tr.get("args", {})

    seeds = tr.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            raise ValueError("training.seeds must be a list of integers")
        if tr.get("data_mode") == "sequential":
            raise ValueError("Multi-seed with data_mode=sequential is not yet supported")

    if "training" in stages_to_run:
        method = tr.get("method", "sft")
        if method == "sft":
            data = tr_args.get("data")
            has_auto_data = (
                stage_outputs(st, "data_generation").get("bct_path")
                or stage_outputs(st, "data_generation").get("train_dir")
                or stage_outputs(st, "data_preparation").get("output_path")
            )
            # Only error if data_generation isn't also going to run
            if not data and not has_auto_data and "data_generation" not in stages_to_run:
                raise ValueError(
                    "Training (sft) requires data files. Either:\n"
                    "  - Include data_generation in stages to run\n"
                    "  - Set training.args.data in config\n"
                    "  - Run a prior pipeline that produced data_generation outputs"
                )
        elif method == "rl":
            if not tr_args.get("bias_types"):
                raise ValueError("Training (rl) requires training.args.bias_types")

    if "evaluation" in stages_to_run:
        ev = config.get("evaluation", {})
        base_only = ev.get("base_only", False)

        if not base_only:
            has_checkpoint = (
                checkpoint_override
                or stage_outputs(st, "training").get("checkpoint")
                or stage_outputs(st, "training").get("control_checkpoint")
                or tr.get("checkpoint")
            )
            # Only error if training isn't also going to run
            if not has_checkpoint and "training" not in stages_to_run:
                raise ValueError(
                    "Evaluation requires a checkpoint. Either:\n"
                    "  - Include training in stages to run\n"
                    "  - Set training.checkpoint in config\n"
                    "  - Pass --checkpoint on the command line\n"
                    "  - Set evaluation.base_only: true for base model eval only"
                )

    if "analysis" in stages_to_run:
        ev_args = config.get("evaluation", {}).get("args", {})
        an_args = config.get("analysis", {}).get("args", {})
        log_dir = an_args.get("log_dir") or ev_args.get("log_dir")
        if log_dir and not Path(log_dir).exists() and "evaluation" not in stages_to_run:
            _print(f"  Warning: eval log dir '{log_dir}' does not exist yet")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _resolve_stages(config: dict, args) -> list[str]:
    """Determine which stages to run for a config, applying skips."""
    if args.stages:
        stages = [resolve_stage(s) for s in args.stages.split(",")]
    elif args.start_from:
        start = resolve_stage(args.start_from)
        idx = STAGES.index(start)
        stages = STAGES[idx:]
    else:
        stages = list(STAGES)

    # Skip data_generation if not configured
    if "data_generation" not in config and "data_generation" in stages:
        stages.remove("data_generation")

    # Skip data_preparation if not enabled
    dp = config.get("data_preparation", {})
    if not dp.get("enabled", False) and "data_preparation" in stages:
        stages.remove("data_preparation")

    return stages


def run_pipeline(
    config: dict,
    stages_to_run: list[str],
    checkpoint_override: str | None,
    force: bool,
    dry_run: bool,
) -> bool:
    """Run a single experiment pipeline (all stages except analysis).

    Returns True on success, False on failure.
    """
    st = load_state(config)

    # Check for config changes
    current_hash = config_hash(config)
    if st.get("config_hash") and st["config_hash"] != current_hash:
        if not force:
            _print(
                f"\nWarning: Config '{config['name']}' has changed since last run "
                f"(was {st['config_hash']}, now {current_hash})."
            )
            _print("Use --force to proceed anyway.")
            return False
    st["config_hash"] = current_hash

    for stage in stages_to_run:
        if stage == "analysis":
            continue  # Deferred to caller

        if stage_status(st, stage) == "completed" and not force:
            _print(f"Skipping {stage} (already completed, use --force to re-run)")
            continue

        mark_stage(st, stage, "running")
        save_state(config, st)

        try:
            if stage == "data_generation":
                steps = _get_data_gen_steps(config)
                cmds = [
                    (_build_single_data_gen_cmd(config, step), step.get("script", "datagen"))
                    for step in steps
                ]
                if len(cmds) == 1:
                    rc, output = run_stage_command(cmds[0][0], stage, config, dry_run)
                    outputs = parse_data_gen_output(output) if not dry_run else {}
                else:
                    results = run_parallel_commands(cmds, stage, config, dry_run)
                    outputs = {}
                    rc = 0
                    if not dry_run:
                        for (ret, output), (_, label) in zip(results, cmds):
                            outputs.update(parse_data_gen_output(output))
                            if ret != 0:
                                rc = ret

            elif stage == "data_preparation":
                cmd = build_data_prep_cmd(config, st)
                if cmd is None:
                    mark_stage(st, stage, "skipped")
                    save_state(config, st)
                    continue
                rc, output = run_stage_command(cmd, stage, config, dry_run)
                outputs = {}

            elif stage == "training":
                tr = config.get("training", {})
                data_mode = tr.get("data_mode", "concat")

                if data_mode == "sequential":
                    # Sequential: train on each data file in order, chaining checkpoints
                    dg_outputs = stage_outputs(st, "data_generation")
                    data_paths = _resolve_training_data(config, st, is_control=False)
                    seq_cmds = build_sequential_training_cmds(config, st, data_paths)

                    outputs = {}
                    rc = 0
                    prev_checkpoint = None
                    for seq_cmd, label in seq_cmds:
                        # Replace placeholder with actual previous checkpoint
                        if prev_checkpoint and "<auto:previous-checkpoint>" in seq_cmd:
                            idx = seq_cmd.index("<auto:previous-checkpoint>")
                            seq_cmd[idx] = prev_checkpoint

                        ret, out = run_stage_command(seq_cmd, f"{stage} [{label}]", config, dry_run)
                        if not dry_run:
                            parsed = parse_training_output(out)
                            if parsed.get("checkpoint"):
                                prev_checkpoint = parsed["checkpoint"]
                            if ret != 0:
                                rc = ret
                                break

                    if prev_checkpoint:
                        outputs["checkpoint"] = prev_checkpoint

                    # Sequential control variant (if enabled)
                    if tr.get("include_control") or tr.get("control_only"):
                        ctrl_paths = _resolve_training_data(config, st, is_control=True)
                        ctrl_cmds = build_sequential_training_cmds(config, st, ctrl_paths)
                        prev_ctrl_ckpt = None
                        for seq_cmd, label in ctrl_cmds:
                            # Adjust run name for control
                            if "--run-name" in seq_cmd:
                                rn_idx = seq_cmd.index("--run-name") + 1
                                if not seq_cmd[rn_idx].endswith("-ctrl"):
                                    seq_cmd[rn_idx] = f"{seq_cmd[rn_idx]}-ctrl"
                            if prev_ctrl_ckpt and "<auto:previous-checkpoint>" in seq_cmd:
                                idx = seq_cmd.index("<auto:previous-checkpoint>")
                                seq_cmd[idx] = prev_ctrl_ckpt
                            ret, out = run_stage_command(
                                seq_cmd, f"{stage} [{label}-ctrl]", config, dry_run,
                            )
                            if not dry_run:
                                parsed = parse_training_output(out)
                                if parsed.get("checkpoint"):
                                    prev_ctrl_ckpt = parsed["checkpoint"]
                                if ret != 0:
                                    rc = ret
                                    break
                        if prev_ctrl_ckpt:
                            outputs["control_checkpoint"] = prev_ctrl_ckpt
                else:
                    # concat or interleave: build commands and run in parallel
                    cmds = build_training_cmds(config, st)
                    results = run_parallel_commands(cmds, stage, config, dry_run)

                    outputs = {}
                    rc = 0
                    if not dry_run:
                        for (ret, output), (_, label) in zip(results, cmds):
                            parsed = parse_training_output(output)
                            if parsed.get("checkpoint"):
                                if label == "main":
                                    outputs["checkpoint"] = parsed["checkpoint"]
                                elif label == "control":
                                    outputs["control_checkpoint"] = parsed["checkpoint"]
                                else:
                                    # Multi-seed: label is "main-s{N}" or "control-s{N}"
                                    outputs[f"{label}_checkpoint"] = parsed["checkpoint"]
                                    # Also update primary keys for backward compat
                                    if label.startswith("main"):
                                        outputs["checkpoint"] = parsed["checkpoint"]
                                    elif label.startswith("control"):
                                        outputs["control_checkpoint"] = parsed["checkpoint"]
                            if ret != 0:
                                rc = ret

            elif stage == "evaluation":
                cmds = build_eval_cmds(config, st, checkpoint_override)
                if not cmds:
                    mark_stage(st, stage, "skipped", outputs={})
                    save_state(config, st)
                    _print(f"  Skipping {stage} (no checkpoints or base eval to run)")
                    continue
                results = run_parallel_commands(cmds, stage, config, dry_run)

                outputs = {}
                rc = 0
                if not dry_run:
                    for (ret, _), (_, label) in zip(results, cmds):
                        if ret != 0:
                            rc = ret

            else:
                _print(f"Unknown stage: {stage}")
                continue

            if dry_run:
                mark_stage(st, stage, "pending")
            elif rc == 0:
                mark_stage(st, stage, "completed", outputs=outputs)
                _print(f"\n  {stage} completed successfully")
                if outputs:
                    for k, v in outputs.items():
                        _print(f"    {k}: {v}")
            else:
                mark_stage(st, stage, "failed", error=f"Exit code {rc}")
                save_state(config, st)
                _print(f"\n  {stage} FAILED (exit code {rc})")
                _print(f"  Check logs: {state_dir(config)}/logs/{stage}_*.log")
                _print(f"  Fix the issue and re-run with: --start-from {stage}")
                return False

        except Exception as e:
            mark_stage(st, stage, "failed", error=str(e))
            save_state(config, st)
            _print(f"\n  {stage} FAILED: {e}")
            return False

        save_state(config, st)

    # Save config alongside state for reference
    config_copy = state_dir(config) / "config.yaml"
    with open(config_copy, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return True


def _build_combined_analysis_cmd(configs: list[dict]) -> list[str]:
    """Build analysis command combining log dirs from all configs."""
    # Collect unique log dirs
    log_dirs: list[str] = []
    for config in configs:
        ev_args = config.get("evaluation", {}).get("args", {})
        an_args = config.get("analysis", {}).get("args", {})
        log_dir = an_args.get("log_dir") or ev_args.get("log_dir", "sycophancy_eval_inspect/logs/tinker_evals")
        if str(log_dir) not in log_dirs:
            log_dirs.append(str(log_dir))

    # Merge analysis args from first config that has them
    args = {}
    for config in configs:
        an = config.get("analysis", {}).get("args", {})
        if an:
            args = an
            break

    cmd = [sys.executable, "-m", "sycophancy_eval_inspect.visualize_results"]
    for ld in log_dirs:
        cmd += ["--log-dir", ld]

    if args.get("bir"):
        cmd.append("--bir")
    if args.get("ba"):
        cmd.append("--ba")
    if args.get("plot"):
        cmd.append("--plot")
    if "output_dir" in args:
        cmd += ["--output-dir", str(args["output_dir"])]
    if "model" in args:
        models = args["model"] if isinstance(args["model"], list) else [args["model"]]
        for m in models:
            cmd += ["--model", m]
    if "prompt_style" in args:
        styles = args["prompt_style"] if isinstance(args["prompt_style"], list) else [args["prompt_style"]]
        for s in styles:
            cmd += ["--prompt-style", s]
    if args.get("summary"):
        cmd.append("--summary")
    if args.get("common_questions"):
        cmd.append("--common-questions")
    if args.get("variance_across"):
        cmd += ["--variance-across", str(args["variance_across"])]

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end consistency training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "configs", type=Path, nargs="+",
        help="YAML experiment config file(s). Multiple configs run sequentially, with analysis deferred to the end.",
    )
    parser.add_argument(
        "--start-from",
        metavar="STAGE",
        help="Skip all stages before this one (e.g. evaluation, train, eval)",
    )
    parser.add_argument(
        "--stages",
        metavar="STAGE,STAGE,...",
        help="Run only these stages (comma-separated)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Override checkpoint path for evaluation stage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run already-completed stages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Validate stage names early
    if args.stages:
        for s in args.stages.split(","):
            if resolve_stage(s) not in STAGES:
                parser.error(f"Unknown stage: {s}. Valid: {', '.join(STAGES)}")
    if args.start_from and resolve_stage(args.start_from) not in STAGES:
        parser.error(f"Unknown stage: {args.start_from}. Valid: {', '.join(STAGES)}")

    # Load all configs
    configs = []
    for config_path in args.configs:
        config = load_config(config_path)
        configs.append(config)

    multi = len(configs) > 1
    any_has_analysis = False

    # Validate all configs and resolve stages upfront
    all_stages = []
    for config in configs:
        stages = _resolve_stages(config, args)
        validate_config(config, stages, load_state(config), args.checkpoint)
        if "analysis" in stages:
            any_has_analysis = True
        all_stages.append(stages)

    if not multi:
        # Single config — run directly (no threading overhead)
        config = configs[0]
        stages = all_stages[0]
        _print(f"Experiment: {config['name']}")
        _print(f"Model: {config['model']}")
        _print(f"\nStages to run: {', '.join(s for s in stages if s != 'analysis')}")
        _print()

        ok = run_pipeline(config, stages, args.checkpoint, args.force, args.dry_run)
        if not ok:
            sys.exit(1)

        _print(f"\n  Pipeline complete: {config['name']}")
        _print(f"  State: {state_path(config)}")
    else:
        # Multiple configs — run pipelines in parallel threads
        for i, (config, stages) in enumerate(zip(configs, all_stages)):
            _print(f"  [{config['name']}] Model: {config['model']}")
            non_analysis = [s for s in stages if s != "analysis"]
            _print(f"  [{config['name']}] Stages: {', '.join(non_analysis)}")
        _print(f"\nRunning {len(configs)} experiment pipelines in parallel...")
        if any_has_analysis:
            _print("  (analysis deferred to end)")
        _print()

        results: dict[str, bool] = {}
        errors: dict[str, str] = {}

        def _run_one(config, stages):
            try:
                _thread_local.prefix = config["name"]
                ok = run_pipeline(config, stages, args.checkpoint, args.force, args.dry_run)
                results[config["name"]] = ok
            except Exception as e:
                results[config["name"]] = False
                errors[config["name"]] = str(e)

        threads = []
        for config, stages in zip(configs, all_stages):
            t = threading.Thread(target=_run_one, args=(config, stages), name=config["name"])
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Check results
        failed = [name for name, ok in results.items() if not ok]
        if failed:
            for name in failed:
                err = errors.get(name, "see logs")
                _print(f"\n  {name} FAILED: {err}")
            sys.exit(1)

    # Run combined analysis at the end
    if any_has_analysis:
        if multi:
            _print(f"\n{'#'*60}")
            _print(f"  Combined Analysis ({len(configs)} experiments)")
            _print(f"{'#'*60}")

        # Update model registry for all configs
        for config in configs:
            update_model_registry(config)

        cmd = _build_combined_analysis_cmd(configs)
        # Use first config for log directory (analysis is shared)
        rc, _ = run_stage_command(cmd, "analysis", configs[0], args.dry_run)

        if not args.dry_run and rc != 0:
            _print(f"\n  Analysis FAILED (exit code {rc})")
            sys.exit(1)

        # Mark analysis completed in all configs
        for config in configs:
            st = load_state(config)
            if not args.dry_run:
                mark_stage(st, "analysis", "completed")
            save_state(config, st)

    _print(f"\n{'='*60}")
    if multi:
        names = ", ".join(c["name"] for c in configs)
        _print(f"  All experiments complete: {names}")
    else:
        _print(f"  Experiment complete: {configs[0]['name']}")
        _print(f"  State: {state_path(configs[0])}")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
