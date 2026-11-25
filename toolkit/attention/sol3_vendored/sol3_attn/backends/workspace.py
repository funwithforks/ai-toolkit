"""Plan bounded scratch and serialize reuse of each provider's CUDA workspace."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
import sys

from ..profiles import LONG_SEQUENCE, PROFILES


GIB = 1024**3
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class Plan:
    head_chunk: int
    mu_score_chunk: int
    minimum_fits: bool


def choose_plan(tokens, heads, dimension, available_bytes, *, preferred_heads=4,
                preferred_mu=4, budget_limit=12 * GIB, reserve_bytes=2 * GIB, use_score_cache=True):
    blocks = (tokens + 63) // 64
    csr = blocks * tokens * 4 + tokens * dimension * 4 + blocks * blocks * 4 + blocks * 16
    scores = blocks * blocks * 12 + blocks * tokens * 4 if use_score_cache else 0
    budget = max(0, min(budget_limit, (available_bytes - reserve_bytes) // 2))
    count = min(heads, preferred_heads, max(1, INT32_MAX // (blocks * tokens)))
    score_count = min(heads, preferred_mu)

    def scratch():
        return (count + heads % count) * csr + score_count * scores + 256 * 1024**2

    while scratch() > budget and (count > 1 or score_count > 1):
        if score_count > 1 and (score_count >= count or count == 1):
            score_count = max(1, score_count // 2)
        else:
            count = max(1, count // 2)
    return Plan(count, score_count, blocks * tokens <= INT32_MAX and scratch() <= budget)


def available_memory(q):
    import torch

    free, _total = torch.cuda.mem_get_info(q.device)
    return free + torch.cuda.memory_reserved(q.device) - torch.cuda.memory_allocated(q.device)


@dataclass
class Call:
    model: str
    optimized: bool
    counters: dict | None
    native: object
    dense: object
    public_qkv: bool = False
    stage_h3_scores: bool = False
    h3_score_heads: int = 4

    @property
    def profile(self):
        return PROFILES[self.model]


_call = ContextVar("sol3_call")


def active_call():
    return _call.get(None)


def count_launch(name):
    call = active_call()
    if call is not None and call.counters is not None:
        call.counters[name] = call.counters.get(name, 0) + 1


def native_attention(q, k, v, **kwargs):
    call = active_call()
    if call is None or call.native is None:
        raise ValueError("The native attention override is only available inside a model adapter")
    return call.native(q, k, v, **kwargs)


def dense_prefix(q, k, v, *, softmax_scale=None):
    call = active_call()
    if call is not None and call.dense is not None:
        return call.dense(q, k, v)
    from flash_attn import flash_attn_func

    return flash_attn_func(q, k, v, softmax_scale=softmax_scale)


class Workspace:
    def __init__(self, model, provider_prefix):
        self.model = model
        self.provider_prefix = provider_prefix
        self.lock = RLock()
        self.stream = None
        self.plan_key = None
        self.plan = None

    def plan_for(self, q, *, preferred_heads=4, reserve_bytes=2 * GIB):
        key = (q.shape, q.stride(), q.device, q.dtype, preferred_heads)
        if key != self.plan_key:
            self.plan = choose_plan(
                *map(int, q.shape[1:]), available_memory(q),
                preferred_heads=preferred_heads, preferred_mu=preferred_heads, reserve_bytes=reserve_bytes,
            )
            self.plan_key = key
        return self.plan

    @contextmanager
    def call(self, q, counters, *, native=None, dense=None, public_qkv=False):
        import torch

        with self.lock:
            stream = torch.cuda.current_stream(q.device)
            if self.stream is not None and self.stream != stream:
                self.stream.synchronize()
            self.stream = stream
            state = Call(self.model, q.shape[1] >= LONG_SEQUENCE, counters, native, dense, public_qkv)
            token = _call.set(state)
            try:
                yield state
            finally:
                _call.reset(token)

    def release(self):
        with self.lock:
            if self.stream is not None:
                self.stream.synchronize()
            self.stream = self.plan_key = self.plan = None
            count = 0
            for name, module in tuple(sys.modules.items()):
                if not name.startswith(self.provider_prefix):
                    continue
                for attr, value in vars(module).items():
                    if isinstance(value, dict) and (attr.endswith("CACHE") or attr in
                            ("LAST", "LAST_KERNELS", "PARTIAL", "BUFFERS", "ROUTE_BUFFERS", "EXTRA")):
                        count += len(value)
                        value.clear()
            return count
