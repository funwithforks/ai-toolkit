"""Full raw-Key mu: fuse eligible moments into each score GEMM tile.

No scores/tokens are sampled. Current-call FP32 logits are consumed by the
existing selector, and scratch is reused only for storage between calls.
"""
import torch, triton
import triton.language as tl
from .sol3_global_mu_cached import cached_route
CACHE={}; LAST={}

@triton.jit
def partial_scores(QB,K,FLAGS,P,SCORE,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,HEAD0:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,J:tl.constexpr):
    tile,jtile,local=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    bh=local+HEAD0; b,h=bh//H,bh%H; NB:tl.constexpr=BK//64
    qs=tile*BQ+tl.arange(0,BQ); js=jtile*NB+tl.arange(0,NB)
    ds=tl.arange(0,D); kt=jtile*BK+tl.arange(0,BK)
    flag=tl.load(FLAGS+(bh*N+qs[:,None])*N+js[None,:],(qs<N)[:,None]&(js<N)[None,:],-2)
    mix=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(js<N)[None,:]&(flag>=0)
    count=tl.zeros((BQ,),tl.float32); mean=tl.zeros((BQ,),tl.float32); m2=tl.zeros((BQ,),tl.float32)
    if tl.sum(tl.sum(mix.to(tl.int32),1),0)>0:
        q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
        key=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
        scores=tl.dot(q,tl.trans(key))*SCALE*1.4426950408889634
        active=tl.reshape(tl.broadcast_to(mix[:,:,None],(BQ,NB,64)),(BQ,BK))&(kt<T)[None,:]
        count=tl.sum(active.to(tl.float32),1)
        mean=tl.sum(tl.where(active,scores,0.),1)/tl.maximum(count,1.)
        delta=scores-mean[:,None]
        m2=tl.sum(tl.where(active,delta*delta,0.),1)
        tl.store(SCORE+(local*N+qs[:,None])*T+kt[None,:],scores,active)
    ix=((local*N+qs)*J+jtile)*3
    tl.store(P+ix,count,qs<N); tl.store(P+ix+1,mean,qs<N); tl.store(P+ix+2,m2,qs<N)

@triton.jit
def combine(P,CUT,MU,SIGMA,N:tl.constexpr,J:tl.constexpr,JJ:tl.constexpr,BETA:tl.constexpr):
    qi,bh=tl.program_id(0),tl.program_id(1); js=tl.arange(0,JJ)
    ix=((bh*N+qi)*J+js)*3
    count=tl.load(P+ix,js<J,0.); mean=tl.load(P+ix+1,js<J,0.); m2=tl.load(P+ix+2,js<J,0.)
    amount=tl.sum(count,0); mu=tl.sum(count*mean,0)/tl.maximum(amount,1.)
    variance=tl.sum(m2+count*(mean-mu)*(mean-mu),0)/tl.maximum(amount,1.)
    sigma=tl.sqrt(tl.maximum(variance,0.)+1.e-6)
    cut=tl.where(amount>0,mu+BETA*sigma,float('inf'))
    tl.store(CUT+bh*N+qi,cut); tl.store(MU+bh*N+qi,mu); tl.store(SIGMA+bh*N+qi,sigma)

def threshold_route(qb,k,flags,bits,cut,*,t,n,h,d,scale,qskip,beta,chunk=12,bq=32,bk=256,stages=2,route_qtile=32):
    heads=k.shape[0]*h; chunk=min(int(chunk),heads); j=triton.cdiv(t,bk)
    assert chunk>0 and bq in (16,32,64) and bk in (128,256,512) and route_qtile in (16,32)
    key=(tuple(k.shape),k.device,chunk,bk)
    if key not in CACHE:
        CACHE[key]=(torch.empty((chunk,n,j,3),device=k.device,dtype=torch.float32),torch.empty((chunk,n,t),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32))
    p,scores,mu,sigma=CACHE[key]
    for start in range(0,heads,chunk):
        amount=min(chunk,heads-start)
        partial_scores[(triton.cdiv(n,bq),j,amount)](qb,k,flags,p,scores,t,n,h,d,scale,qskip,start,bq,bk,j,num_warps=4,num_stages=stages)
        combine[(n,amount)](p,cut[start:start+amount],mu[start:start+amount],sigma[start:start+amount],n,j,triton.next_power_of_2(j),beta,num_warps=4)
        cached_route[(triton.cdiv(n,route_qtile),n,amount)](scores,flags,bits,cut,t,n,start,qskip,route_qtile,num_warps=4,num_stages=1)
    LAST.clear(); LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,current_score_cache=scores,chunk=chunk,bq=bq,bk=bk,stages=stages,score_cache_bytes=scores.numel()*4,partial_bytes=p.numel()*4,old_selection_reused=False,statistics='all eligible original Key logits, stable centered tile M2 then weighted Welford merge')
