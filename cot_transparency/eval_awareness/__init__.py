"""Eval-awareness consistency training.

Transfers the repo's perturbation-agnostic consistency machinery (BCT / RLCT)
from anti-sycophancy to *evaluation awareness*: training a model to behave the
same whether or not the input looks like an alignment evaluation.

Training uses a single eval cue ("You are being evaluated"); other cues and a
held-out environment (Agentic Misalignment) are reserved for measuring
generalization. See cues.py (perturbations) and judge.py (trait classifier).
"""
