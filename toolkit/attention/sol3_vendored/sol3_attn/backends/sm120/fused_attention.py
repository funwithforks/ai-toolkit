"""Exact-token compaction and joint representative/exact online softmax.

No prompt, layer, seed or output-dependent policy. Selection is supplied by the
unchanged complete-current-call mu router. Representatives use fixed block means.
"""
import math
import torch
import triton
import triton.language as tl
CACHE = {}
# Use popcount by default when building the compact token index.
CSR_IMPL = "popc"
CSR_CALLS = 0

@triton.jit
def offsets_kernel(F, OFF, COUNT, T:tl.constexpr, N:tl.constexpr, NN:tl.constexpr):
    qi,bh=tl.program_id(0),tl.program_id(1)
    j=tl.arange(0,NN)
    f=tl.load(F+(bh*N+qi)*N+j,j<N,-2)
    length=tl.minimum(64,tl.maximum(T-j*64,0))
    c=tl.where(f==-1,length,tl.maximum(f,0))
    end=tl.cumsum(c,0)
    tl.store(OFF+(bh*N+qi)*N+j,end-c,j<N)
    tl.store(COUNT+bh*N+qi,tl.sum(c,0))

@triton.jit
def compact_kernel(F, BITS, OFF, IDS, T:tl.constexpr,N:tl.constexpr,J:tl.constexpr=8):
    qi,bh,jtile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    js=jtile*J+tl.arange(0,J);ts=tl.arange(0,64)
    ix=(bh*N+qi)*N+js
    f=tl.load(F+ix,js<N,-2)
    if tl.sum((f==-1)|(f>0),0)>0:
        bits=tl.load(BITS+ix,js<N,0).to(tl.uint64)
        selected=((f==-1)[:,None]|((f>0)[:,None]&(((bits[:,None]>>ts[None,:].to(tl.uint64))&1)!=0)))&((js[:,None]*64+ts[None,:])<T)&(js<N)[:,None]
        rank=tl.cumsum(selected.to(tl.int32),1)-1
        start=tl.load(OFF+ix,js<N,0)
        tl.store(IDS+(bh*N+qi).to(tl.int64)*T+start[:,None]+rank,js[:,None]*64+ts[None,:],selected)

@triton.jit
def compact_popc(F,BITS,OFF,IDS,T:tl.constexpr,N:tl.constexpr,J:tl.constexpr=32):
    qi,bh,jtile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    js=jtile*J+tl.arange(0,J);ts=tl.arange(0,64)
    ix=(bh*N+qi)*N+js
    f=tl.load(F+ix,js<N,-2)
    if tl.sum((f==-1)|(f>0),0)>0:
        bits=tl.load(BITS+ix,js<N,0).to(tl.uint64)
        selected=((f==-1)[:,None]|((f>0)[:,None]&(((bits[:,None]>>ts[None,:].to(tl.uint64))&1)!=0)))&((js[:,None]*64+ts[None,:])<T)&(js<N)[:,None]
        lower=(tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64))-1
        rank=tl.inline_asm_elementwise("popc.b64 $0, $1;","=r,l",[bits[:,None]&lower[None,:]],dtype=tl.int32,is_pure=True,pack=1)
        rank=tl.where((f==-1)[:,None],ts[None,:],rank)
        start=tl.load(OFF+ix,js<N,0)
        tl.store(IDS+(bh*N+qi).to(tl.int64)*T+start[:,None]+rank,js[:,None]*64+ts[None,:],selected)


@triton.jit
def packed_attention(Q,K,V,KC,VM,F,IDS,COUNT,O,KS,VS,COV,
                     T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
                     PREFIX:tl.constexpr,SCALE:tl.constexpr,BK:tl.constexpr=64,HEAD_MAJOR:tl.constexpr=False,FP8:tl.constexpr=False,BQ:tl.constexpr=64,GAMMA:tl.constexpr=0.,QST:tl.constexpr=0,KST:tl.constexpr=0,VST:tl.constexpr=0,OST:tl.constexpr=0):
    qtile,bh=tl.program_id(0),tl.program_id(1)
    if (qtile+1)*BQ <= PREFIX:
        return
    qi=qtile//(64//BQ)
    b,h=bh//H,bh%H
    rows=qtile*BQ+tl.arange(0,BQ);ds=tl.arange(0,D)
    q=tl.load(Q+(b*T+rows[:,None]).to(tl.int64)*QST+h*D+ds[None,:],(rows<T)[:,None],0)
    if FP8:
        ks=tl.load(KS+bh*D+ds);vs=tl.load(VS+bh*D+ds)
        scaled_q=q.to(tl.float32)*ks[None,:]
        qs=tl.maximum(tl.max(tl.abs(scaled_q),1)/448.,1.e-12)
        q8=(scaled_q/qs[:,None]).to(tl.float8e4nv)
    m=tl.full((BQ,),-float('inf'),tl.float32)
    l=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,D),tl.float32)
    # Fixed original-block value means; residual mass is remaining token count.
    for start in range(0,N,64):
        js=start+tl.arange(0,64)
        f=tl.load(F+(bh*N+qi)*N+js,js<N,-1)
        length=tl.minimum(64,tl.maximum(T-js*64,0))
        remaining=tl.where(f==-1,0,length-tl.maximum(f,0))
        kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0)
        vm=tl.load(VM+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0)
        score=tl.dot(q,tl.trans(kc))*SCALE+tl.log2(tl.maximum(remaining,1).to(tl.float32))[None,:]
        score=tl.where(((js<N)&(remaining>0))[None,:],score,-float('inf'))
        mm=tl.maximum(m,tl.max(score,1))
        safe=tl.where(mm==-float('inf'),0.,mm)
        alpha=tl.exp2(m-safe)
        p=tl.exp2(score-safe[:,None])
        acc=acc*alpha[:,None]+tl.dot(p.to(tl.bfloat16),vm)
        l=l*alpha+tl.sum(p,1);m=mm
    if GAMMA != 0.:
        cd=tl.arange(0,D)
        cov=tl.load(COV+bh*D*D+ds[:,None]*D+cd[None,:])
        correction=tl.dot(q,cov)*(D**-.5)*GAMMA
        acc+=l[:,None]*correction
    total=tl.load(COUNT+bh*N+qi)
    for start in range(0,total,BK):
        pos=start+tl.arange(0,BK)
        ids=tl.load(IDS+(bh*N+qi).to(tl.int64)*T+pos,pos<total,0)
        kbase=(bh*T+ids[:,None]).to(tl.int64)*D if HEAD_MAJOR else (b*T+ids[:,None]).to(tl.int64)*KST+h*D
        k=tl.load(K+kbase+ds[None,:],(pos<total)[:,None],0.)
        vbase=(bh*T+ids[:,None]).to(tl.int64)*D if HEAD_MAJOR else (b*T+ids[:,None]).to(tl.int64)*VST+h*D
        v=tl.load(V+vbase+ds[None,:],(pos<total)[:,None],0.)
        if FP8:
            score=tl.dot(q8,tl.trans(k))*qs[:,None]*SCALE
        else:
            score=tl.dot(q,tl.trans(k))*SCALE
        score=tl.where((pos<total)[None,:],score,-float('inf'))
        mm=tl.maximum(m,tl.max(score,1))
        alpha=tl.exp2(m-mm);p=tl.exp2(score-mm[:,None])
        if FP8:
            delta=tl.dot((p*448.).to(tl.float8e4nv),v)*(vs[None,:]/448.)
        else:
            delta=tl.dot(p.to(tl.bfloat16),v)
        acc=acc*alpha[:,None]+delta
        l=l*alpha+tl.sum(p,1);m=mm
    result=acc/l[:,None]
    tl.store(O+(b*T+rows[:,None]).to(tl.int64)*OST+h*D+ds[None,:],result,((rows<T)&(rows>=PREFIX))[:,None])

def prepare(q,route):
    b,t,h,d=q.shape;n=triton.cdiv(t,64);key=(q.device,tuple(q.shape))
    if key not in CACHE:
        if len(CACHE)>=2:CACHE.clear()
        CACHE[key]=(torch.empty((b*h,n,n),device=q.device,dtype=torch.int32),
                    torch.empty((b*h,n),device=q.device,dtype=torch.int32),
                    torch.empty((b*h,n,t),device=q.device,dtype=torch.int32))
    off,count,ids=CACHE[key]
    offsets_kernel[(n,b*h)](route['flags'],off,count,t,n,triton.next_power_of_2(n),num_warps=4)
    global CSR_CALLS
    if CSR_IMPL=="popc":kernel,j=compact_popc,16
    elif CSR_IMPL=="scan":kernel,j=compact_kernel,32
    else:raise ValueError("Unknown CSR implementation")
    kernel[(n,b*h,triton.cdiv(n,j))](route['flags'],route['bits'],off,ids,t,n,j,num_warps=4)
    CSR_CALLS+=1
    return ids,count

def attention(q,k,v,route,*,prefix_tokens=0,dense=None,bk=64,head_major=False,warps=4,fp8=False,stages=1,covariance=None,cov_gamma=0.):
    b,t,h,d=q.shape;n=triton.cdiv(t,64)
    ids,count=prepare(q,route)
    out=torch.empty_like(q)
    original_k,original_v=k,v
    ks=vs=q
    if fp8:
        k,v,ks,vs=quantized_kv(k,v);head_major=True
    elif head_major:
        k=k.permute(0,2,1,3).contiguous();v=v.permute(0,2,1,3).contiguous()
    packed_attention[(n,b*h)](q,k,v,route['kc'],route['vm'],route['flags'],ids,count,out,ks,vs,covariance if covariance is not None else q,
                              t,n,h,d,prefix_tokens,d**-.5*math.log2(math.e),bk,head_major,fp8,GAMMA=float(cov_gamma),
                              num_warps=warps,num_stages=stages)
    if prefix_tokens:
        if dense is None:raise RuntimeError('Selected common dense backend required')
        out[:,:prefix_tokens]=dense(q[:,:prefix_tokens],original_k,original_v)
    return out

@triton.jit
def channel_partial(K,V,PK,PV,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NT:tl.constexpr,BT:tl.constexpr=256):
    tile,bh=tl.program_id(0),tl.program_id(1)
    ts=tile*BT+tl.arange(0,BT);ds=tl.arange(0,D)
    ix=((bh//H*T+ts[:,None]).to(tl.int64)*H+bh%H)*D+ds[None,:]
    k=tl.load(K+ix,(ts<T)[:,None],0).to(tl.float32)
    v=tl.load(V+ix,(ts<T)[:,None],0).to(tl.float32)
    tl.store(PK+(bh*NT+tile)*D+ds,tl.max(tl.abs(k),0))
    tl.store(PV+(bh*NT+tile)*D+ds,tl.max(tl.abs(v),0))
@triton.jit
def channel_scale(PK,PV,KS,VS,NT:tl.constexpr,ND:tl.constexpr,D:tl.constexpr,BD:tl.constexpr=32):
    bh,dt=tl.program_id(0),tl.program_id(1)
    ts=tl.arange(0,ND);ds=dt*BD+tl.arange(0,BD)
    ix=(bh*NT+ts[:,None])*D+ds[None,:]
    k=tl.load(PK+ix,(ts<NT)[:,None]&(ds<D)[None,:],0.)
    v=tl.load(PV+ix,(ts<NT)[:,None]&(ds<D)[None,:],0.)
    tl.store(KS+bh*D+ds,tl.maximum(tl.max(k,0)/448.,1.e-12),ds<D)
    tl.store(VS+bh*D+ds,tl.maximum(tl.max(v,0)/448.,1.e-12),ds<D)
@triton.jit
def quantize_channels(K,V,KS,VS,K8,V8,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BT:tl.constexpr=128):
    tile,bh=tl.program_id(0),tl.program_id(1)
    ts=tile*BT+tl.arange(0,BT);ds=tl.arange(0,D)
    ix=((bh//H*T+ts[:,None]).to(tl.int64)*H+bh%H)*D+ds[None,:]
    k=tl.load(K+ix,(ts<T)[:,None],0).to(tl.float32);v=tl.load(V+ix,(ts<T)[:,None],0).to(tl.float32)
    ks=tl.load(KS+bh*D+ds);vs=tl.load(VS+bh*D+ds)
    dst=(bh*T+ts[:,None]).to(tl.int64)*D+ds[None,:]
    tl.store(K8+dst,(k/ks[None,:]).to(tl.float8e4nv),(ts<T)[:,None])
    tl.store(V8+dst,(v/vs[None,:]).to(tl.float8e4nv),(ts<T)[:,None])

def quantized_kv(k,v):
    b,t,h,d=k.shape;nt=triton.cdiv(t,256)
    pk=torch.empty((b*h,nt,d),device=k.device,dtype=torch.float32);pv=torch.empty_like(pk)
    ks=torch.empty((b*h,d),device=k.device,dtype=torch.float32);vs=torch.empty_like(ks)
    k8=torch.empty((b*h,t,d),device=k.device,dtype=torch.float8_e4m3fn);v8=torch.empty_like(k8)
    channel_partial[(nt,b*h)](k,v,pk,pv,t,h,d,nt,num_warps=8)
    channel_scale[(b*h,triton.cdiv(d,32))](pk,pv,ks,vs,nt,triton.next_power_of_2(nt),d,num_warps=4)
    quantize_channels[(triton.cdiv(t,128),b*h)](k,v,ks,vs,k8,v8,t,h,d,num_warps=4)
    return k8,v8,ks,vs
