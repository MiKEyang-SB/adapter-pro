"""
fa3_fp32_patch.py

Replace FA2's flash_attn_func / flash_attn_varlen_func with FlashAttention-3 (Hopper)
versions that use fp32_accumulate=True.

This fixes BF16 training instability (arXiv:2510.04212) by ensuring the O = P @ V
accumulation inside the fused flash attention kernel uses FP32 precision.

Requires:
  - H100/H800 GPU (Hopper, sm_90)
  - flash-attn package compiled with Hopper/FA3 support

Usage:
    from prismatic.util.fa3_fp32_patch import patch_fa3_fp32_accumulate
    patch_fa3_fp32_accumulate()  # Call BEFORE model instantiation
"""

import warnings
import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_fa3_fp32_accumulate():
    """
    Monkey-patch HuggingFace Qwen2's flash attention calls to use FA3 with fp32_accumulate=True.

    FA3 has a different API than FA2:
      - No dropout_p (FA3 doesn't support attention dropout)
      - No alibi_slopes, no return_attn_probs
      - Added fp32_accumulate parameter
      - Returns (out, softmax_lse) tuple instead of just out

    This creates thin wrappers that accept FA2's API, call FA3, and return in FA2's format.
    Must be called BEFORE model instantiation (before AutoModelForCausalLM.from_config).
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # --- Import FA3 ---
    _fa3_func = None
    _fa3_varlen_func = None

    # Try multiple known import paths for FA3
    for module_path in [
        "flash_attn.hopper.flash_attn_interface",
        "flash_attn_hopper.flash_attn_interface",
        "flash_attn_hopper",
    ]:
        try:
            mod = __import__(module_path, fromlist=["flash_attn_func", "flash_attn_varlen_func"])
            _fa3_func = getattr(mod, "flash_attn_func", None)
            _fa3_varlen_func = getattr(mod, "flash_attn_varlen_func", None)
            if _fa3_func is not None:
                print(f"[FA3 Patch] Loaded FA3 from: {module_path}")
                break
        except ImportError:
            continue

    if _fa3_func is None:
        raise ImportError(
            "FlashAttention-3 not found. Tried:\n"
            "  - flash_attn.hopper.flash_attn_interface\n"
            "  - flash_attn_hopper.flash_attn_interface\n"
            "  - flash_attn_hopper\n"
            "Install flash-attn with Hopper support. FA3 requires H100/H800 (sm_90)."
        )

    # --- Wrappers: FA2 API → FA3 + fp32_accumulate=True ---

    def fa3_flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        """Drop-in replacement for FA2 flash_attn_func, calls FA3 with fp32_accumulate=True."""
        if dropout_p != 0.0:
            warnings.warn(
                f"FA3 does not support attention dropout (got dropout_p={dropout_p}), ignoring.",
                stacklevel=2,
            )
        result = _fa3_func(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            fp32_accumulate=True,
        )
        # FA3 returns (out, softmax_lse), FA2 returns just out
        if isinstance(result, tuple):
            return result[0]
        return result

    def fa3_flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        """Drop-in replacement for FA2 flash_attn_varlen_func, calls FA3 with fp32_accumulate=True."""
        if dropout_p != 0.0:
            warnings.warn(
                f"FA3 does not support attention dropout (got dropout_p={dropout_p}), ignoring.",
                stacklevel=2,
            )
        result = _fa3_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            fp32_accumulate=True,
        )
        # FA3 returns (out, softmax_lse), FA2 returns just out
        if isinstance(result, tuple):
            return result[0]
        return result

    # --- Apply patch: replace module-level references in HF's Qwen2 ---
    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
    qwen2_module.flash_attn_func = fa3_flash_attn_func
    qwen2_module.flash_attn_varlen_func = fa3_flash_attn_varlen_func

    print("[FA3 Patch] Replaced flash_attn_func/varlen_func → FA3 with fp32_accumulate=True")
