"""Complete eligible-key mu/sigma in three launches without a raw-score cache.

Q groups share each K tile through L2. The second dot reproduces the exact
same scores after the global reduction; no sampling or approximate thresholds.
"""
import torch,triton
import triton.language as tl
CACHE={};LAST={}
@triton.jit
def moments(QB,K,F,P,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,NT:tl.constexpr):
    qtile,ktile,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    qs=qtile*BQ+tl.arange(0,BQ);ds=tl.arange(0,D);kt=ktile*BK+tl.arange(0,BK);b,h=bh//H,bh%H
    f=tl.load(F+(bh*N+qs[:,None])*N+kt[None,:]//64,(qs<N)[:,None]&(kt<T)[None,:],-2)
    eligible=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(kt<T)[None,:]&(f>=0)
    count=tl.sum(eligible.to(tl.float32),1);mean=tl.zeros((BQ,),tl.float32);m2=tl.zeros((BQ,),tl.float32)
    if tl.sum(count,0)>0:
        q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
        k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
        score=tl.dot(q,tl.trans(k))*SCALE*1.4426950408889634
        mean=tl.sum(tl.where(eligible,score,0.),1)/tl.maximum(count,1.)
        m2=tl.sum(tl.where(eligible,(score-mean[:,None])*(score-mean[:,None]),0.),1)
    ix=((bh*N+qs)*NT+ktile)*3
    tl.store(P+ix,count,qs<N);tl.store(P+ix+1,mean,qs<N);tl.store(P+ix+2,m2,qs<N)
@triton.jit
def reduce(P,CUT,MU,SIGMA,N:tl.constexpr,NT:tl.constexpr,NN:tl.constexpr,BETA:tl.constexpr):
    qi,bh=tl.program_id(0),tl.program_id(1);js=tl.arange(0,NN);ix=((bh*N+qi)*NT+js)*3
    count=tl.load(P+ix,js<NT,0.);mean=tl.load(P+ix+1,js<NT,0.);m2=tl.load(P+ix+2,js<NT,0.)
    amount=tl.sum(count,0);mu=tl.sum(count*mean,0)/tl.maximum(amount,1.)
    var=tl.sum(m2+count*(mean-mu)*(mean-mu),0)/tl.maximum(amount,1.)
    sigma=tl.sqrt(tl.maximum(var,0.)+1.e-6);cut=tl.where(amount>0,mu+BETA*sigma,float('inf'))
    target=bh*N+qi;tl.store(CUT+target,cut);tl.store(MU+target,mu);tl.store(SIGMA+target,sigma)
@triton.jit
def select(QB,K,F,BITS,CUT,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
    qtile,ktile,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    qs=qtile*BQ+tl.arange(0,BQ);ds=tl.arange(0,D);kt=ktile*BK+tl.arange(0,BK);b,h=bh//H,bh%H
    js=ktile*(BK//64)+tl.arange(0,BK//64)
    ix=(bh*N+qs[:,None])*N+js[None,:]
    f=tl.load(F+ix,(qs<N)[:,None]&(js<N)[None,:],-2)
    active=(qs<N)&((qs+1)*64>QSKIP)
    bits=tl.full((BQ,BK//64),0,tl.uint64);count=tl.zeros((BQ,BK//64),tl.int32)
    if tl.sum(tl.sum((f>=0)&active[:,None],1),0)>0:
        q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
        k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
        score=tl.dot(q,tl.trans(k))*SCALE*1.4426950408889634
        cut=tl.load(CUT+bh*N+qs,qs<N,float('inf'))
        chosen=tl.reshape((score>cut[:,None])&(kt<T)[None,:],(BQ,BK//64,64))
        chosen=chosen&(f>=0)[:,:,None]&active[:,None,None]
        count=tl.sum(chosen.to(tl.int32),2)
        tokenbit=tl.full((64,),1,tl.uint64)<<tl.arange(0,64).to(tl.uint64)
        bits=tl.sum(tl.where(chosen,tokenbit[None,None,:],0),2)
    valid=(qs<N)[:,None]&(js<N)[None,:]
    tl.store(F+ix,tl.where(f>=0,count,f),valid);tl.store(BITS+ix,bits,valid)
def route(qb,k,flags,bits,cut,*,t,n,h,d,scale,qskip,beta,bq=32,bk=256):
    nt=triton.cdiv(t,bk);key=(tuple(k.shape),k.device,bq,bk)
    if key not in CACHE:
        CACHE.clear()
        CACHE[key]=(torch.empty((k.shape[0]*h,n,nt,3),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device),torch.empty((k.shape[0]*h,n),device=k.device))
    p,mu,sigma=CACHE[key];grid=(triton.cdiv(n,bq),nt,k.shape[0]*h)
    moments[grid](qb,k,flags,p,t,n,h,d,scale,qskip,bq,bk,nt,num_warps=4,num_stages=2)
    reduce[(n,k.shape[0]*h)](p,cut,mu,sigma,n,nt,triton.next_power_of_2(nt),beta,num_warps=4)
    select[grid](qb,k,flags,bits,cut,t,n,h,d,scale,qskip,bq,bk,num_warps=4,num_stages=2)
    LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,all_eligible_keys_each_query=True,score_recomputed=True)
