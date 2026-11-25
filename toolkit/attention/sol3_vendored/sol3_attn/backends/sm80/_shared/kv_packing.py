"""Input-independent lossless head-major KV packing and parallel CSR construction."""
import triton
import triton.language as tl
@triton.jit
def lengths_offsets(Q,FLAGS,LENGTHS,OFFSETS,QP,T:tl.constexpr,N:tl.constexpr,ACTIVE:tl.constexpr,SKIP:tl.constexpr,H:tl.constexpr,D:tl.constexpr,HEAD_BASE:tl.constexpr,NN:tl.constexpr):
 row=tl.program_id(0);lh=row//ACTIVE;qi=row%ACTIVE+SKIP;js=tl.arange(0,NN)
 flag=tl.load(FLAGS+(lh*N+qi)*N+js,js<N,-2);sizes=tl.minimum(64,tl.maximum(T-js*64,0))
 counts=tl.where(js<N,tl.where(flag==-1,sizes,tl.maximum(flag,0)),0)
 offsets=tl.cumsum(counts,0)-counts
 tl.store(OFFSETS+row*N+js,offsets,js<N);tl.store(LENGTHS+row,tl.sum(counts,0))
 qs=qi*64+tl.arange(0,64);ds=tl.arange(0,D)
 values=tl.load(Q+(qs[:,None].to(tl.int64)*H+lh+HEAD_BASE)*D+ds[None,:],(qs<T)[:,None],0)
 tl.store(QP+(lh*(T-SKIP*64)+qs[:,None]-SKIP*64).to(tl.int64)*D+ds[None,:],values,(qs<T)[:,None])
@triton.jit
def indices_parallel(FLAGS,BITS,PTR,OFFSETS,INDEX,T:tl.constexpr,N:tl.constexpr,ACTIVE:tl.constexpr,SKIP:tl.constexpr,H:tl.constexpr,HEAD_BASE:tl.constexpr,PACKED:tl.constexpr,CHUNK:tl.constexpr=16):
 row=tl.program_id(0);tile=tl.program_id(1);bh=row//ACTIVE;qi=row%ACTIVE+SKIP
 js=tile*CHUNK+tl.arange(0,CHUNK);ts=tl.arange(0,64)
 flag=tl.load(FLAGS+(bh*N+qi)*N+js,js<N,-2);word=tl.load(BITS+(bh*N+qi)*N+js,js<N,0)
 keys=js[:,None]*64+ts[None,:];selected=((((word[:,None]>>ts[None,:].to(tl.uint64))&1)!=0)&(flag>=0)[:,None])|(flag==-1)[:,None]
 selected=selected&(js<N)[:,None]&(keys<T)
 lower = (tl.full((64,), 1, tl.uint64) << ts.to(tl.uint64)) - 1
 rank = tl.inline_asm_elementwise("popc.b64 $0, $1;", "=r,l", [word[:,None].to(tl.uint64) & lower[None,:]], dtype=tl.int32, is_pure=True, pack=1)
 rank = tl.where((flag == -1)[:,None], ts[None,:], rank)
 offset=tl.load(OFFSETS+row*N+js,js<N,0);ptr=tl.load(PTR+row)
 pages=(bh+HEAD_BASE)*T+keys if PACKED else keys*H+bh+HEAD_BASE
 tl.store(INDEX+ptr+offset[:,None]+rank,pages,selected)
