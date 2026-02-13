"""
fa3_fp32_patch.py

Replace FA2's flash_attn_func / flash_attn_varlen_func with FlashAttention-3 (Hopper)
versions that use fp32_accumulate=True.

This fixes BF16 training instability by ensuring the O = P @ V accumulation inside
the fused flash attention kernel uses FP32 precision.

Requires:
  - H100/H800 GPU (Hopper, sm_90)
  - FlashAttention-3 installed and importable

Usage:
    from prismatic.util.fa3_fp32_patch import patch_fa3_fp32_accumulate
    patch_fa3_fp32_accumulate()  # Call BEFORE model instantiation
"""

import sys
import types
import importlib.machinery
import importlib.metadata
import warnings
import logging

logger = logging.getLogger(__name__)
_PATCHED = False
_FAKE_FA2_VERSION = "2.6.3"

# ==============================================================================
# Early fake flash_attn registration (runs at import time of this module).
#
# Problem: `from prismatic.util.fa3_fp32_patch import ...` triggers
#   prismatic/__init__.py → prismatic/models/materialize.py → import transformers
# so transformers is imported BEFORE patch_fa3_fp32_accumulate() is called.
#
# transformers caches `_is_package_available("flash_attn")` at import time via
# importlib.util.find_spec() and importlib.metadata.version(), both of which fail
# when only flash_attn_3 (not flash_attn) is installed.
#
# Fix: register a fake flash_attn module and patch metadata.version() at module
# level so the checks pass even if transformers imports first.
# ==============================================================================

# Step 1: fake module in sys.modules → makes find_spec("flash_attn") succeed
if "flash_attn" not in sys.modules:
    _fake_fa = types.ModuleType("flash_attn")
    _fake_fa.__version__ = _FAKE_FA2_VERSION
    _fake_fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    sys.modules["flash_attn"] = _fake_fa

# Step 2: patch importlib.metadata.version → makes version checks succeed
_original_metadata_version = importlib.metadata.version


def _patched_metadata_version(name):
    if name == "flash_attn":
        return _FAKE_FA2_VERSION
    return _original_metadata_version(name)


importlib.metadata.version = _patched_metadata_version


# ==============================================================================


def _try_import(module_path: str):
    try:
        mod = __import__(module_path, fromlist=["flash_attn_func", "flash_attn_varlen_func"])
        fa3_func = getattr(mod, "flash_attn_func", None)
        fa3_varlen = getattr(mod, "flash_attn_varlen_func", None)
        if fa3_func is not None:
            return mod, fa3_func, fa3_varlen
        return None, None, None
    except Exception:
        return None, None, None


def patch_fa3_fp32_accumulate():
    """
    Monkey-patch HuggingFace Qwen2's flash attention calls to use FA3 with fp32_accumulate=True.

    FA3 has a different API than FA2:
      - No dropout_p (FA3 doesn't support attention dropout)
      - No alibi_slopes, no return_attn_probs
      - Added fp32_accumulate parameter
      - Returns (out, softmax_lse) tuple instead of just out

    This creates thin wrappers that accept FA2's API, call FA3, and return in FA2's format.
    Must be called BEFORE model instantiation.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # --- Step 3: Fix cached variables in transformers if already imported ---
    try:
        import transformers.utils.import_utils as _hf_import_utils
        if hasattr(_hf_import_utils, '_flash_attn_available'):
            _hf_import_utils._flash_attn_available = True
    except ImportError:
        pass

    # --- Import FA3 ---
    _fa3_func = None
    _fa3_varlen_func = None
    loaded_from = None

    # Try multiple known import paths for FA3
    for module_path in [
        "flash_attn.hopper.flash_attn_interface",      # some flash-attn installs
        "flash_attn_hopper.flash_attn_interface",      # alt naming
        "flash_attn_hopper",                           # alt naming
        "flash_attn_interface",                        # <-- your current FA3 exposes this
        "flash_attn_3.flash_attn_interface",           # some wheels put it here
    ]:
        mod, f, fv = _try_import(module_path)
        if f is not None:
            _fa3_func = f
            _fa3_varlen_func = fv
            loaded_from = module_path
            print(f"[FA3 Patch] Loaded FA3 from: {module_path}")
            break

    if _fa3_func is None:
        raise ImportError(
            "FlashAttention-3 not found via known import paths.\n"
            "Tried:\n"
            "  - flash_attn.hopper.flash_attn_interface\n"
            "  - flash_attn_hopper.flash_attn_interface\n"
            "  - flash_attn_hopper\n"
            "  - flash_attn_interface\n"
            "  - flash_attn_3.flash_attn_interface\n"
            "But none provided flash_attn_func.\n"
            "Make sure FA3 is installed and importable in the current python env."
        )

    if _fa3_varlen_func is None:
        print("[FA3 Patch] Warning: flash_attn_varlen_func not found; varlen wrapper may fail.")

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
        if _fa3_varlen_func is None:
            raise RuntimeError("flash_attn_varlen_func is not available in the loaded FA3 module.")

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

        if isinstance(result, tuple):
            return result[0]
        return result

    # --- Apply patch: replace module-level references in HF's Qwen2 ---
    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
    qwen2_module.flash_attn_func = fa3_flash_attn_func
    qwen2_module.flash_attn_varlen_func = fa3_flash_attn_varlen_func

    print(f"[FA3 Patch] Replaced Qwen2 flash_attn_func/varlen_func → FA3(fp32_accumulate=True), source={loaded_from}")
