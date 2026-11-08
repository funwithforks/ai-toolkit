"""Convert an ai-toolkit-saved MiniMax-H3 Flash LoRA to the ComfyUI loader layout.

The toolkit's module tree stores the factored q/k/v projections of a flash
checkpoint at ``blocks.N.attn.{q,k,v}_proj.{down,up}``. ComfyUI's HyperFlow
INT8 loader nests them under the SplitQKV wrapper it installs:
``blocks.N.attn.qkv_proj.{q,k,v}_proj.{down,up}``. ComfyUI binds LoRA keys by
exact module path, so the q/k/v adapters of every factored block would
silently drop on load (204 of 664 tensors on a 300-step rank-8 file). This
script rewrites those keys into the loader's layout; every other tensor —
int8-block adapters, out_proj/fc1/fc2 factors, the token refiner — passes
through unchanged.

Usage:
    python scripts/convert_flash_lora_to_comfy.py INPUT [INPUT ...] [-o OUTPUT]

Without ``-o`` each input is written next to itself as
``<stem>_comfy.safetensors``. The source files are never modified.
"""

import argparse
import json
import re
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# diffusion_model.blocks.N.attn.<q|k|v>_proj.<down|up>.lora_X.weight on a
# factored block (the factors live directly under the projection)
QKV_KEY = re.compile(
    r"^(diffusion_model\.blocks\.\d+\.attn\.)([qkv])_proj\.((?:down|up)\.lora_[AB]\.weight)$"
)


def convert(path: str, output: str) -> None:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        metadata = json.loads(f.read(n)).get("__metadata__")

    sd = load_file(path)
    out, renamed, rename_map = {}, 0, {}
    for key, tensor in sd.items():
        match = QKV_KEY.match(key)
        if match:
            new_key = f"{match.group(1)}qkv_proj.{match.group(2)}_proj.{match.group(3)}"
            out[new_key] = tensor.contiguous()
            rename_map[new_key] = key
            renamed += 1
        else:
            out[key] = tensor.contiguous()
    save_file(out, output, metadata=metadata)

    # verify: nothing dropped, no un-nested factored key left, shapes preserved
    new = safe_open(output, framework="pt")
    new_keys = list(new.keys())
    src = safe_open(path, framework="pt")
    assert len(new_keys) == len(sd), f"{output}: tensor count changed"
    leftover = [
        k for k in new_keys
        if re.match(r"^diffusion_model\.blocks\.\d+\.attn\.[qkv]_proj\.(down|up)\.", k)
    ]
    assert not leftover, f"{output}: {len(leftover)} keys were not renamed"
    for key in new_keys:
        original = rename_map.get(key, key)
        assert new.get_slice(key).get_shape() == src.get_slice(original).get_shape(), key
    print(f"{path} -> {output} ({renamed} keys renamed, {len(out)} total, verified)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="ai-toolkit flash LoRA safetensors")
    parser.add_argument(
        "-o", "--output", help="output path (single input only; "
        "default: <stem>_comfy.safetensors next to each input)"
    )
    args = parser.parse_args()
    if args.output:
        if len(args.inputs) != 1:
            parser.error("-o works with exactly one input")
        convert(args.inputs[0], args.output)
        return
    for path in args.inputs:
        stem = Path(path).stem
        convert(path, str(Path(path).with_name(f"{stem}_comfy.safetensors")))


if __name__ == "__main__":
    main()
