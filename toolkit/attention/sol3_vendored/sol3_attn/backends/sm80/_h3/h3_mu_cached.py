"""Complete per-query eligible-key mu with bounded FP32 current-query-group staging."""
import torch,triton
import triton.language as tl
CACHE={};LAST={}
@triton.jit
def score_moments(QB,K,FLAGS,RAW,P,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,NT:tl.constexpr,PAD:tl.constexpr,QBASE):
 ktile,bh=tl.program_id(0),tl.program_id(1);b,h=bh//H,bh%H
 qs=tl.arange(0,BQ);gqs=QBASE+qs;ds=tl.arange(0,D);kt=ktile*BK+tl.arange(0,BK)
 flag=tl.load(FLAGS+(bh*N+gqs[:,None])*N+kt[None,:]//64,(gqs<N)[:,None]&(kt<T)[None,:],-2)
 eligible=(gqs<N)[:,None]&((gqs+1)*64>QSKIP)[:,None]&(kt<T)[None,:]&(flag>=0)
 count=tl.sum(eligible.to(tl.float32),1);mean=tl.zeros((BQ,),tl.float32);m2=tl.zeros((BQ,),tl.float32)
 if tl.sum(count,0)>0:
  qb=tl.load(QB+(bh*N+gqs[:,None])*D+ds[None,:],(gqs<N)[:,None],0).to(tl.bfloat16)
  k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  scores=tl.dot(qb,tl.trans(k))*SCALE*1.4426950408889634
  mean=tl.sum(tl.where(eligible,scores,0.),1)/tl.maximum(count,1.)
  m2=tl.sum(tl.where(eligible,(scores-mean[:,None])*(scores-mean[:,None]),0.),1)
  tl.store(RAW+(bh*BQ+qs[:,None])*PAD+kt[None,:],scores,(gqs<N)[:,None]&(kt<T)[None,:])
 ix=((bh*BQ+qs)*NT+ktile)*3
 tl.store(P+ix,count,gqs<N);tl.store(P+ix+1,mean,gqs<N);tl.store(P+ix+2,m2,gqs<N)
@triton.jit
def reduce_moments(P,CUT,MU,SIGMA,N:tl.constexpr,NT:tl.constexpr,NN:tl.constexpr,BETA:tl.constexpr,BQ:tl.constexpr,QBASE):
 qi,bh=tl.program_id(0),tl.program_id(1);js=tl.arange(0,NN);ix=((bh*BQ+qi)*NT+js)*3
 count=tl.load(P+ix,js<NT,0.);mean=tl.load(P+ix+1,js<NT,0.);m2=tl.load(P+ix+2,js<NT,0.)
 amount=tl.sum(count,0);mu=tl.sum(count*mean,0)/tl.maximum(amount,1.)
 var=tl.sum(m2+count*(mean-mu)*(mean-mu),0)/tl.maximum(amount,1.)
 sigma=tl.sqrt(tl.maximum(var,0.)+1.e-6);cut=tl.where(amount>0,mu+BETA*sigma,float('inf'))
 target=bh*N+QBASE+qi;tl.store(CUT+target,cut);tl.store(MU+target,mu);tl.store(SIGMA+target,sigma)
@triton.jit
def select_cached(FLAGS,BITS,RAW,CUT,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,PAD:tl.constexpr,QBASE):
 j,bh=tl.program_id(0),tl.program_id(1);qs=tl.arange(0,BQ);gqs=QBASE+qs;ts=tl.arange(0,64);kt=j*64+ts
 ix=(bh*N+gqs)*N+j;flag=tl.load(FLAGS+ix,gqs<N,-2)
 mixed=(gqs<N)&((gqs+1)*64>QSKIP)&(flag>=0)
 scores=tl.load(RAW+(bh*BQ+qs[:,None])*PAD+kt[None,:],mixed[:,None]&(kt<T)[None,:],0.)
 cut=tl.load(CUT+bh*N+gqs,gqs<N,float('inf'))
 chosen=mixed[:,None]&(kt<T)[None,:]&(scores>cut[:,None])
 count=tl.sum(chosen.to(tl.int32),1);tokenbit=tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64)
 bits=tl.sum(tl.where(chosen,tokenbit[None,:],0),1)
 tl.store(FLAGS+ix,tl.where(flag>=0,count,flag),gqs<N);tl.store(BITS+ix,bits,gqs<N)
def route(qb,k,flags,bits,cut,*,t,n,h,d,scale,qskip,beta,bq=32,bk=256):
 pad=n*64;nt=triton.cdiv(t,bk);key=(tuple(k.shape),k.device,bq,bk)
 if key not in CACHE:
  CACHE.clear()
  CACHE[key]=(torch.empty((k.shape[0]*h,bq,pad),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,bq,nt,3),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device,dtype=torch.float32))
 raw,p,mu,sigma=CACHE[key]
 for qbase in range(0,n,bq):
  rows=min(bq,n-qbase)
  score_moments[(nt,k.shape[0]*h)](qb,k,flags,raw,p,t,n,h,d,scale,qskip,bq,bk,nt,pad,qbase,num_warps=4,num_stages=2)
  reduce_moments[(rows,k.shape[0]*h)](p,cut,mu,sigma,n,nt,triton.next_power_of_2(nt),beta,bq,qbase,num_warps=4)
  select_cached[(n,k.shape[0]*h)](flags,bits,raw,cut,t,n,h,qskip,bq,pad,qbase,num_warps=4,num_stages=1)
 LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,raw=raw,partial_tiles=nt,raw_fresh_each_query_group=True,all_eligible_keys_each_query=True,query_groups=triton.cdiv(n,bq),bounded_raw_bytes=raw.numel()*raw.element_size())
