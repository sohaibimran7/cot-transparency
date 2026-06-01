# Environment setup

The current workflow uses **`uv`** (not the `pyenv`/`pip` flow in the old
`README_FOR_CODE.md`, which describes the legacy `cot_transparency` stage_one pipeline).

## Quick start

```bash
# 1. Create the venv (Python 3.11+; the working venv is 3.12).
uv venv

# 2. Install deps, EXCLUDING grugstream (it fails to resolve/build here and
#    nothing in the active Tinker/eval path imports it).
grep -v '^grugstream' requirements.txt | uv pip install -r /dev/stdin

# 3. inspect-ai is NOT in requirements.txt — install it separately.
uv pip install inspect-ai

# 4. One-time NLTK data (used by some formatters).
uv run python -c "import nltk; nltk.download('punkt')"
```

Run things with `uv run <cmd>` or activate the venv (`source .venv/bin/activate`).

## Gotchas

- **grugstream**: listed in `requirements.txt` but excluded above — see step 2.
- **inspect-ai**: the sycophancy eval stack (`sycophancy_eval_inspect/`) depends on it,
  but it is installed separately (step 3), not pinned in `requirements.txt`.
- **`zsh: command not found: conda`** printed before shell commands is harmless — it
  comes from a stale conda init in the shell environment, not from this repo. It does
  not affect anything; ignore it (or remove the conda hook from your shell profile).
- **No `uv.lock`** is checked in; `requirements.txt` is the dependency source of truth.

## API keys

Put credentials in a `.env` at the repo root (loaded via `python-dotenv`). Needed:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and the Tinker API key. (`.env` is gitignored
and read-protected — do not commit it.)

## Tests

```bash
pytest                 # offline subset (default; excludes network/gpu/tinker markers)
pytest -m tinker       # tests that need the live Tinker service / a checkpoint
pytest -m ""           # everything
```

See `[tool.pytest.ini_options]` in `pyproject.toml` for the marker definitions.

## Lint / type-check

```bash
make check             # black + ruff + pyright via pre-commit (see .pre-commit-config.yaml)
```
