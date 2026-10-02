"""Liger-kernel fused elementwise ops for training (shared import block).

One try-import for the functional RMSNorm / SwiGLU / modulated-RMSNorm
bindings, one loud print when the bindings are unavailable, one env knob
(LIGER_FUSIONS=0, or the legacy KREA2_LIGER_FUSIONS=0). Models decide
WHERE to route (which norm feeds which fused op is a per-model measured
decision); this module only owns the imports.

Numerics: liger is a rounding-position change vs the eager chain (fp32
single-store rounding vs one rounding per eager op, ~1 bf16 ulp) -- fine
for training, not bitwise; verify models end-to-end.
"""

import os

LIGER_OK = (
    os.environ.get("LIGER_FUSIONS", "1") != "0"
    and os.environ.get("KREA2_LIGER_FUSIONS", "1") != "0"
)

liger_rms_norm = liger_swiglu = LigerModNorm = None
if LIGER_OK:
    try:
        from liger_kernel.functional import rms_norm as liger_rms_norm
        from liger_kernel.functional import swiglu as liger_swiglu
        from liger_kernel.ops.modulated_rms_norm import (
            LigerModulatedRMSNormFunction as LigerModNorm,
        )
    except Exception as _e:
        LIGER_OK = False
        liger_rms_norm = liger_swiglu = LigerModNorm = None
        print(
            f"[liger] fusion bindings unavailable ({_e}); "
            f"models fall back to eager elementwise chains"
        )
