"""Block pooling, sparse routing and attention kernels for Sol³."""
import torch
import triton
import triton.language as tl
import os
BLOCK=64
BETA1_DELTA=float(os.environ.get("SOL3_BETA1_DELTA","1.0"))
BETA2=float(os.environ.get("SOL3_BETA2","0.0"))
BETA3=float(os.environ.get("SOL3_BETA3","1.0"))
BV=int(os.environ.get("SOL3_BV","128"))
WARPS=int(os.environ.get("SOL3_WARPS","4"))

@triton.jit
def pool(K,V,KC,VS,C,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
    j,bh=tl.program_id(0),tl.program_id(1)
    b,h=bh//H,bh%H
    t=j*64+tl.arange(0,64);d=tl.arange(0,D)
    ix=((b*T+t[:,None]).to(tl.int64)*H+h)*D+d[None,:]
    valid=t<T; n=tl.minimum(64,T-j*64).to(tl.float32)
    k=tl.load(K+ix,valid[:,None],0).to(tl.float32)
    v=tl.load(V+ix,valid[:,None],0).to(tl.float32)
    km=tl.sum(k,0)/n; vs=tl.sum(v,0); vm=vs/n
    var=tl.sum(tl.sum(tl.where(valid[:,None],(v-vm[None,:])*(v-vm[None,:]),0),0),0)/(n*D)
    rms=tl.sqrt(tl.maximum(var,0))
    si=((b*N+j)*H+h)*D+d
    tl.store(KC+si,km);tl.store(VS+si,vs)
    tl.store(C+(b*N+j)*H+h,tl.log2(n*tl.maximum(rms,1.e-6)))

@triton.jit
def moments(KC,C,M,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
    bh=tl.program_id(0); b,h=bh//H,bh%H
    ds=tl.arange(0,D);js=tl.arange(0,32)
    sm=tl.zeros((D,),tl.float32);ss=tl.zeros((D,),tl.float32)
    sc=tl.full((),0.,tl.float32);sc2=tl.full((),0.,tl.float32)
    for start in range(0,N,32):
        j=start+js;valid=j<N
        k=tl.load(KC+((b*N+j[:,None])*H+h)*D+ds[None,:],valid[:,None],0).to(tl.float32)
        c=tl.load(C+(b*N+j)*H+h,valid,0)
        sm+=tl.sum(k,0);ss+=tl.sum(k*k,0)
        sc+=tl.sum(c,0);sc2+=tl.sum(c*c,0)
    mean=sm/N;var=tl.maximum(ss/N-mean*mean,0);mc=sc/N;vc=tl.maximum(sc2/N-mc*mc,0)
    tl.store(M+bh*(2*D+2)+ds,mean)
    tl.store(M+bh*(2*D+2)+D+ds,var)
    tl.store(M+bh*(2*D+2)+2*D,mc)
    tl.store(M+bh*(2*D+2)+2*D+1,vc)


@triton.jit
def query_pool(Q,QB,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr):
 qi,bh=tl.program_id(0),tl.program_id(1);b,h=bh//H,bh%H
 ts=qi*64+tl.arange(0,64);ds=tl.arange(0,D)
 q=tl.load(Q+((b*T+ts[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(ts<T)[:,None],0).to(tl.float32)
 n=tl.minimum(64,T-qi*64).to(tl.float32)
 tl.store(QB+(bh*N+qi)*D+ds,tl.sum(q,0)/n)

@triton.jit
def route(QB,K,KC,C,M,FLAGS,MASK,
          SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
          B1:tl.constexpr,B2:tl.constexpr,B3:tl.constexpr,SINK0:tl.constexpr,SINK1:tl.constexpr):
 tile,j,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);b,h=bh//H,bh%H
 qs=tile*16+tl.arange(0,16);ds=tl.arange(0,D);ts=tl.arange(0,64)
 qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0)
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds)
 mc=tl.load(M+bh*(2*D+2)+2*D);vc=tl.load(M+bh*(2*D+2)+2*D+1)
 s=SCALE*1.4426950408889634
 mu=tl.sum(qb*mk[None,:],1)*s;var=tl.sum(qb*qb*vk[None,:],1)*s*s
 th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6);th2=mu+mc+B2*tl.sqrt(tl.maximum(var+vc,0)+1.e-6)
 kc=tl.load(KC+((b*N+j)*H+h)*D+ds).to(tl.float32);c=tl.load(C+(b*N+j)*H+h)
 proxy=tl.sum(qb*kc[None,:],1)*s
 ex=(qs<N)&((proxy>th1)|(tl.abs(qs-j)<=1)|((j>=SINK0)&(j<SINK1)))
 pure=(qs<N)&~ex&(proxy+c<th2);mix=(qs<N)&~ex&~pure
 mask=tl.full((16,),0,tl.uint64);count=tl.zeros((16,),tl.int32)
 if tl.sum(mix.to(tl.int32),0)>0:
  kt=j*64+ts;valid=kt<T;n=tl.minimum(64,T-j*64).to(tl.float32)
  k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],valid[:,None],0).to(tl.float32)
  a=tl.dot(qb,tl.trans(k),input_precision="tf32x3")*s
  am=tl.sum(tl.where(valid[None,:],a,0.),1)/n
  av=tl.sum(tl.where(valid[None,:],(a-am[:,None])*(a-am[:,None]),0.),1)/n
  sel=mix[:,None]&valid[None,:]&(a>am[:,None]+B3*tl.sqrt(tl.maximum(av,0)+1.e-6)[:,None])
  count=tl.sum(sel.to(tl.int32),1)
  bits=tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64)
  mask=tl.sum(tl.where(sel,bits[None,:],0),1)
  compact=tl.sort(tl.where(sel,ts[None,:],64),descending=False)
  ix=(bh*N+qs)*N+j
  tl.store(MASK+ix[:,None]*64+ts[None,:],compact,(qs<N)[:,None]&(ts[None,:]<count[:,None]))
 flags=tl.where(ex,-1,tl.where(pure,-2,count))
 ix=(bh*N+qs)*N+j
 tl.store(FLAGS+ix,flags,qs<N)

@triton.jit
def forward(Q,K,V,KC,VS,C,M,O,FLAGS,MASK,
            SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
            B1:tl.constexpr,B2:tl.constexpr,B3:tl.constexpr,
            SINK0:tl.constexpr,SINK1:tl.constexpr,BV:tl.constexpr):
    vi,qi,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=bh//H,bh%H
    ts=tl.arange(0,64); ds=tl.arange(0,D); vd=vi*BV+tl.arange(0,BV); gs=tl.arange(0,32)
    qt=qi*64+ts;validq=qt<T
    qo=((b*T+qt[:,None]).to(tl.int64)*H+h)*D+ds[None,:]
    q=tl.load(Q+qo,validq[:,None],0)
    qn=tl.minimum(64,T-qi*64).to(tl.float32)
    qb=tl.sum(q.to(tl.float32),0)/qn
    mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds)
    mc=tl.load(M+bh*(2*D+2)+2*D);vc=tl.load(M+bh*(2*D+2)+2*D+1)
    s=SCALE*1.4426950408889634
    mu=tl.sum(qb*mk,0)*s;var=tl.sum(qb*qb*vk,0)*s*s
    th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6)
    th2=mu+mc+B2*tl.sqrt(tl.maximum(var+vc,0)+1.e-6)
    acc=tl.zeros((64,BV),tl.float32);l=tl.zeros((64,),tl.float32);mx=tl.full((64,),-float('inf'),tl.float32)
    for start in range(0,N,32):
        js=start+gs;vj=js<N
        kc=tl.load(KC+((b*N+js[:,None])*H+h)*D+ds[None,:],vj[:,None],0)
        scores=tl.dot(q,kc.T)*s
        proxy=tl.sum(scores,0)/qn
        c=tl.load(C+(b*N+js)*H+h,vj,0)
        flags=tl.load(FLAGS+(bh*N+qi)*N+js,vj,-2)
        exact=vj & (flags==-1)
        pure=vj & (flags!=-1)
        mixed=vj & (flags>=0)
        has=tl.sum(pure.to(tl.int32),0)>0
        safe=tl.where(has,tl.where(pure[None,:],scores,-float('inf')),0.)
        nm=tl.where(has,tl.maximum(mx,tl.max(safe,1)),mx)
        al=tl.exp2(tl.where(has,mx-nm,0.))
        p=tl.where(has & pure[None,:],tl.exp2(safe-tl.where(has,nm,0.)[:,None]),0.)
        vs=tl.load(VS+((b*N+js[:,None])*H+h)*D+vd[None,:],vj[:,None],0)
        acc=acc*al[:,None]+tl.dot(p,vs,input_precision="tf32x3")
        ns=tl.minimum(64,tl.maximum(0,T-js*64)).to(tl.float32)
        l=l*al+tl.sum(p*ns[None,:],1);mx=nm
        eo=tl.where(exact,gs,32)
        for z in range(tl.sum(exact.to(tl.int32),0)):
            off=tl.min(eo,0);j=start+off;eo=tl.where(gs==off,32,eo)
            kt=j*64+ts;validk=kt<T
            ko=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:]
            k=tl.load(K+ko,validk[:,None],0)
            exs=tl.dot(q,k.T)*s+tl.where(validk[None,:],0.,-float('inf'))
            nm=tl.maximum(mx,tl.max(exs,1));al=tl.exp2(mx-nm);ep=tl.exp2(exs-nm[:,None])
            vo=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+vd[None,:]
            v=tl.load(V+vo,validk[:,None],0)
            acc=acc*al[:,None]+tl.dot(ep.to(q.dtype),v)
            l=l*al+tl.sum(ep,1);mx=nm
        counts=tl.maximum(flags,0)
        ends=tl.cumsum(counts,0)
        beginnings=ends-counts
        total=tl.sum(counts,0)
        for chunk in range(tl.cdiv(total,64)):
            cr=chunk*64+ts;ok=cr<total
            bo=tl.minimum(tl.sum((cr[:,None]>=ends[None,:]).to(tl.int32),1),31)
            begin=tl.gather(beginnings,bo,0)
            local=tl.minimum(tl.maximum(cr-begin,0),63)
            token=tl.load(MASK+((bh*N+qi)*N+start+bo)*64+local,ok,0).to(tl.int32)
            kt=(start+bo)*64+token
            ko=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:]
            sk=tl.load(K+ko,ok[:,None],0)
            ms=tl.dot(q,sk.T)*s+tl.where(ok[None,:],0.,-float('inf'))
            means=tl.gather(scores,tl.broadcast_to(bo[None,:],(64,64)),1)
            nm=tl.maximum(mx,tl.max(ms,1));al=tl.exp2(mx-nm)
            diff=tl.where(ok[None,:],tl.exp2(ms-nm[:,None])-tl.exp2(means-nm[:,None]),0.)
            vo=((b*T+kt[:,None]).to(tl.int64)*H+h)*D+vd[None,:]
            sv=tl.load(V+vo,ok[:,None],0)
            acc=acc*al[:,None]+tl.dot(diff.to(q.dtype),sv)
            l=l*al+tl.sum(diff,1);mx=nm
    oo=((b*T+qt[:,None]).to(tl.int64)*H+h)*D+vd[None,:]
    tl.store(O+oo,acc/l[:,None],validq[:,None])

BUFFERS={}
ROUTE_BUFFERS={}
def prepare(k,v):
    b,t,h,d=k.shape;n=triton.cdiv(t,64)
    key=(tuple(k.shape),k.device)
    buf=BUFFERS.get(key)
    if buf is None:
        kc=torch.empty((b,n,h,d),device=k.device,dtype=torch.bfloat16)
        vs=torch.empty((b,n,h,d),device=k.device,dtype=torch.float32)
        c=torch.empty((b,n,h),device=k.device,dtype=torch.float32)
        m=torch.empty((b*h,2*d+2),device=k.device,dtype=torch.float32)
        buf=BUFFERS[key]=(kc,vs,c,m)
    kc,vs,c,m=buf
    pool[(n,b*h)](k,v,kc,vs,c,t,n,h,d,num_warps=4,num_stages=1)
    moments[(b*h,)](kc,c,m,t,n,h,d,num_warps=4,num_stages=1)
    return buf

def _sol3(q,k,v,*,scale=None,tau=1.,thresh_type="diag",kv_splits=1,sink_tokens=0,sink_start=None,query_threshold_bias=None,beta1=None,beta2=None,beta3=None,bv=None,warps=None):
    assert q.is_cuda and q.dtype==torch.bfloat16 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q.shape==k.shape==v.shape and q.shape[-1]==128 and kv_splits==1 and query_threshold_bias is None
    # Use native attention for the exact-only path.
    if tau <= -100.:
        from sol3_attn.backends.workspace import native_attention as sol_attn
        return sol_attn(q,k,v,scale=scale,tau=tau,thresh_type=thresh_type,kv_splits=1,sink_tokens=sink_tokens,sink_start=sink_start)
    assert thresh_type=="diag"
    b,t,h,d=q.shape;n=triton.cdiv(t,64);scale=d**-.5 if scale is None else scale
    start=(t-sink_tokens if sink_start is None else sink_start) if sink_tokens else 0
    s0=start//64;s1=triton.cdiv(start+sink_tokens,64) if sink_tokens else 0
    b1=tau+BETA1_DELTA if beta1 is None else beta1;b2=BETA2 if beta2 is None else beta2;b3=BETA3 if beta3 is None else beta3
    kc,vs,c,m=prepare(k,v);o=torch.empty_like(v)
    vv=BV if bv is None else bv;ww=WARPS if warps is None else warps
    rkey=(tuple(q.shape),q.device)
    rb=ROUTE_BUFFERS.get(rkey)
    if rb is None:
        qb=torch.empty((b*h,n,d),device=q.device,dtype=torch.float32)
        flags=torch.empty((b*h,n,n),device=q.device,dtype=torch.int32)
        mask=torch.empty((b*h,n,n,64),device=q.device,dtype=torch.uint8)
        rb=ROUTE_BUFFERS[rkey]=(qb,flags,mask)
    qb,flags,mask=rb
    query_pool[(n,b*h)](q,qb,t,n,h,d,num_warps=4,num_stages=1)
    route[(triton.cdiv(n,16),n,b*h)](qb,k,kc,c,m,flags,mask,scale,t,n,h,d,b1,b2,b3,s0,s1,num_warps=4,num_stages=1)
    forward[(d//vv,n,b*h)](q,k,v,kc,vs,c,m,o,flags,mask,scale,t,n,h,d,b1,b2,b3,s0,s1,vv,num_warps=ww,num_stages=1)
    return o

def sol3(q,k,v,**kwargs):
    with torch.cuda.device(q.device):
        return _sol3(q,k,v,**kwargs)
