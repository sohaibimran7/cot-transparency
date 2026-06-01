"""Side-by-side playground for BCT/RLCT checkpoints.

Launch:
    python scripts/playground/app.py            # local
    python scripts/playground/app.py --share    # public Gradio URL

UI:
  * Choose 1–4 panels.
  * Each panel has a model dropdown (base / BCT / RLCT / control, all seeds).
  * One shared input box dispatches to every selected panel in parallel.
  * Temperature, max-tokens, and system prompt are shared.
  * Each panel keeps its own multi-turn chat history; "Clear" wipes one panel.

Sampling clients are cached by checkpoint URL — swapping dropdowns reuses
warm clients within a session.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402

from cot_transparency.data_models.messages import (  # noqa: E402
    StrictChatMessage,
    StrictMessageRole,
)
from scripts.playground.registry import ModelEntry, build_registry  # noqa: E402


# Import inference module directly to bypass the broken-init chain that
# scripts/tinker_training/sample_from_checkpoint.py already works around.
def _import_module_directly(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_inference = _import_module_directly(
    REPO_ROOT / "cot_transparency" / "apis" / "tinker" / "inference.py",
    "tinker_inference_direct",
)
TinkerSamplingClient = _inference.TinkerSamplingClient
SamplingConfig = _inference.SamplingConfig


MAX_PANELS = 4

ROLE_MAP = {
    "user": StrictMessageRole.user,
    "assistant": StrictMessageRole.assistant,
    "system": StrictMessageRole.system,
}


@dataclass
class ClientCache:
    """Reuse one TinkerSamplingClient per (model, checkpoint) pair."""

    _clients: dict[tuple[str, str | None], TinkerSamplingClient]
    _lock: Lock

    @classmethod
    def make(cls) -> "ClientCache":
        return cls(_clients={}, _lock=Lock())

    def get(self, model: str, checkpoint: str | None) -> TinkerSamplingClient:
        key = (model, checkpoint)
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = TinkerSamplingClient(model=model, checkpoint=checkpoint)
                client.setup()
                self._clients[key] = client
            return client


def _to_strict(messages: list[dict[str, str]]) -> list[StrictChatMessage]:
    return [
        StrictChatMessage(role=ROLE_MAP[m["role"]], content=m["content"])
        for m in messages
        if m.get("content")
    ]


def _build_app(registry: list[ModelEntry]) -> gr.Blocks:
    by_label = {e.label: e for e in registry}
    labels = [e.label for e in registry]

    cache = ClientCache.make()
    executor = ThreadPoolExecutor(max_workers=MAX_PANELS)

    def _default_label(idx: int) -> str:
        # Pre-fill with first N entries when available so the UI lands populated.
        return labels[idx] if idx < len(labels) else labels[0]

    def sample_one(
        label: str,
        history: list[dict[str, str]],
        user_msg: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> list[dict[str, str]]:
        if not label or not user_msg.strip():
            return history
        entry = by_label[label]
        client = cache.get(entry.model, entry.checkpoint)

        msgs: list[dict[str, str]] = []
        if system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.extend(history)
        msgs.append({"role": "user", "content": user_msg})

        cfg = SamplingConfig(max_tokens=int(max_tokens), temperature=float(temperature))
        try:
            results = client.sample(_to_strict(msgs), n_samples=1, config=cfg)
            reply = results[0].text if results else "(empty)"
        except Exception as exc:  # surface errors in the chat instead of crashing
            reply = f"[error: {type(exc).__name__}: {exc}]"

        return history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]

    with gr.Blocks(title="Consistency-Training Playground", fill_width=True) as demo:
        gr.Markdown(
            "## Consistency-Training Playground\n"
            "Send one prompt to up to 4 checkpoints in parallel. "
            "Pick a model per panel; histories are kept per-panel for multi-turn chats."
        )

        with gr.Row():
            with gr.Column(scale=1):
                n_panels = gr.Radio(
                    choices=[1, 2, 3, 4],
                    value=2,
                    label="Panels",
                    interactive=True,
                )
                system_prompt = gr.Textbox(
                    label="System prompt (shared)",
                    placeholder="Optional — e.g. 'You are a helpful assistant.'",
                    lines=2,
                )
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max tokens")
                clear_all = gr.Button("Clear all chats")
            with gr.Column(scale=4):
                dropdowns: list[gr.Dropdown] = []
                chatbots: list[gr.Chatbot] = []
                clear_buttons: list[gr.Button] = []
                panel_cols: list[gr.Column] = []

                # Build 4 fixed panels, hide the trailing ones based on `n_panels`.
                with gr.Row(equal_height=True):
                    for i in range(MAX_PANELS):
                        col = gr.Column(visible=(i < 2), min_width=280)
                        with col:
                            dd = gr.Dropdown(
                                choices=labels,
                                value=_default_label(i),
                                label=f"Panel {i + 1}",
                                interactive=True,
                                filterable=True,
                            )
                            cb = gr.Chatbot(
                                height=520,
                                label=None,
                            )
                            cl = gr.Button("Clear panel", size="sm")
                        panel_cols.append(col)
                        dropdowns.append(dd)
                        chatbots.append(cb)
                        clear_buttons.append(cl)

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type a message and press Enter…",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)

        # Panel visibility wiring.
        def _set_visibility(n: int):
            return [gr.update(visible=(i < n)) for i in range(MAX_PANELS)]

        n_panels.change(_set_visibility, inputs=n_panels, outputs=panel_cols)

        # Per-panel clear.
        for i, btn in enumerate(clear_buttons):
            btn.click(lambda: [], outputs=chatbots[i])

        # Clear-all.
        clear_all.click(lambda: [[] for _ in chatbots], outputs=chatbots)

        # Send: dispatch to all visible panels in parallel.
        def on_send(
            user_msg: str,
            n: int,
            system_prompt_val: str,
            temperature_val: float,
            max_tokens_val: int,
            *panel_args,
        ):
            # panel_args is (dd_1..dd_N, history_1..history_N) flattened.
            labels_in = list(panel_args[:MAX_PANELS])
            histories = list(panel_args[MAX_PANELS:])
            if not user_msg.strip():
                return histories + [""]

            futures = []
            for i in range(MAX_PANELS):
                if i < n:
                    futures.append(
                        executor.submit(
                            sample_one,
                            labels_in[i],
                            histories[i],
                            user_msg,
                            system_prompt_val,
                            temperature_val,
                            max_tokens_val,
                        )
                    )
                else:
                    futures.append(None)

            new_histories = []
            for i, fut in enumerate(futures):
                new_histories.append(fut.result() if fut else histories[i])
            return new_histories + [""]  # clear the input box

        send_inputs = [msg, n_panels, system_prompt, temperature, max_tokens, *dropdowns, *chatbots]
        send_outputs = [*chatbots, msg]
        send.click(on_send, inputs=send_inputs, outputs=send_outputs)
        msg.submit(on_send, inputs=send_inputs, outputs=send_outputs)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Expose a public Gradio URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    registry = build_registry()
    if not registry:
        print("No checkpoints found under artifacts/runs/. Nothing to launch.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(registry)} model entries.")
    demo = _build_app(registry)
    demo.queue(default_concurrency_limit=MAX_PANELS).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
