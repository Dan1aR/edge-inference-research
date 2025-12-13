from __future__ import annotations

from typing import Iterable

from src.bf16_accum import patch_yolos_self_attention_bf16


_DEF_WARN_ONCE = False


def apply_attention_patch(model) -> None:
    """Apply the existing YOLOS self-attention bf16-accum forward patch.

    The underlying implementation lives in :mod:`src.bf16_accum`. This helper wraps
    the call so training scripts have a single entrypoint.
    """

    global _DEF_WARN_ONCE
    if not _DEF_WARN_ONCE:
        print("[patch] enabling bf16-accum attention forward")
        _DEF_WARN_ONCE = True
    patch_yolos_self_attention_bf16(model)
