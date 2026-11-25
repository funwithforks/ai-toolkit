"""Write exact-token page IDs in the order fixed by the native Sol³ routing decisions."""
import threading
import weakref

import torch
import triton
import triton.language as tl

from ..workspace import active_call, count_launch


@triton.jit
def emit(flags, bits, ptr, offsets, pages, tokens, blocks, active, skip, heads,
         first_head, full_offsets: tl.constexpr):
    block = tl.program_id(0) * 16 + tl.arange(0, 16)
    row = tl.program_id(1)
    head = row // active
    q = row % active + skip
    source = (head.to(tl.int64) * blocks + q) * blocks
    valid = block < blocks
    flag = tl.load(flags + source + block, valid, -2)
    word = tl.load(bits + source + block, valid & (flag > 0), 0).to(tl.uint64)
    offset_row = source if full_offsets else row.to(tl.int64) * blocks
    offset = tl.load(offsets + offset_row + block, valid, 0)
    lane = tl.arange(0, 64)
    one = tl.full((64,), 1, tl.uint64) << lane.to(tl.uint64)
    key = block[:, None] * 64 + lane[None, :]
    selected = ((flag == -1)[:, None] | ((word[:, None] & one[None, :]) != 0)) & (key < tokens)
    rank = tl.inline_asm_elementwise("popc.b64 $0, $1;", "=r,l",
              [word[:, None] & (one - 1)[None, :]], dtype=tl.int32, is_pure=True, pack=1)
    rank = tl.where((flag == -1)[:, None], lane[None, :], rank)
    dest = tl.load(ptr + row) + offset[:, None] + rank
    tl.store(pages + dest, key * heads + first_head + head, selected)


_mutex = threading.RLock()
_cache = weakref.WeakKeyDictionary()


def _make_slot(q, slot, vectorize_plan, plan_info_scalars):
    original=slot["wrapper"]
    with _mutex:
        cached=_cache.get(original)
        if cached is not None and cached["ptr"] is slot["ptr"] and cached["pages"] is slot["pages"]:
            return cached
        import flashinfer
        if (getattr(original,"_backend",None)!="fa2" or getattr(original,"_kv_layout",None)!="NHD"
                or slot["ptr"].dtype!=torch.int32 or slot["pages"].dtype!=torch.int32):
            return slot
        wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(slot["float_workspace"],kv_layout="NHD",backend="fa2")
        wrapper=vectorize_plan(wrapper)
        groups=slot["ptr"].numel()-1
        if groups<=0:return slot
        # A one-page planning extent makes splitting a request unnecessary. The
        # resulting nonpartitioned kernel obtains the real length from ptr[row+1]-ptr[row].
        planned=torch.arange(groups+1,dtype=torch.int32,device="cpu")
        wrapper.plan(slot["qo"],planned,slot["pages"],slot["last_cpu"],1,1,q.shape[-1],1,causal=False,
                     q_data_type=q.dtype,kv_data_type=q.dtype,use_fp16_qk_reduction=False)
        info=plan_info_scalars(wrapper._plan_info)
        if len(info)!=15 or info[14]!=0:return slot
        wrapper._paged_kv_indptr_buf=slot["ptr"]
        wrapper._paged_kv_indices_buf=slot["pages"]
        cached=dict(slot,wrapper=wrapper,static_plan=True,plan_info=info)
        _cache[original]=cached
        return cached


def repair_workspace(ws, q, vectorize_plan, plan_info_scalars):
    call = active_call()
    if call is None or not (call.public_qkv or call.profile.repair_plan):
        return ws
    if all(s["static_plan"] for s in ws["slots"].values()):
        return ws
    slots = {}
    for width, slot in ws["slots"].items():
        fixed = slot if slot["static_plan"] else _make_slot(q, slot, vectorize_plan, plan_info_scalars)
        slots[width] = fixed
        if fixed is not slot:
            count_launch("plan_repairs")
    return dict(ws, slots=slots)


def launch_pages(kernel, grid, *args, **options):
    call = active_call()
    if call is None or not call.optimized or not call.profile.ordered_pages:
        return kernel[grid](*args, **options)
    flags, bits = args[:2]
    tokens, blocks, active, skip, heads = args[5:10]
    full_offsets = call.model == "h3"
    if active <= 0 or blocks != triton.cdiv(tokens, 64) or (full_offsets and args[12]):
        return kernel[grid](*args, **options)
    ptr, offsets, pages = args[2:5]
    if full_offsets:
        ptr, offsets = offsets, ptr
    rows = flags.numel() // (blocks * blocks) * active
    result = emit[(triton.cdiv(blocks, 16), rows)](
        flags, bits, ptr, offsets, pages, tokens, blocks, active, skip, heads,
        args[11], full_offsets, num_warps=4,
    )
    count_launch("h3_pages" if full_offsets else "pages")
    return result


def run_paged(wrapper, *args, **kwargs):
    result = wrapper.run(*args, **kwargs)
    count_launch("paged")
    return result
