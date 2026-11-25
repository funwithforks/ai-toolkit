"""Cache FP32 key scores per head chunk and reuse them for routing in the same call.
"""
import torch,triton
import triton.language as tl
from .sol3_global_mu import combine_moments
CACHE={};LAST={}
@triton.jit
def partial_and_score(QB,K,FLAGS,P,SCORES,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,HEAD0:tl.constexpr):
 tile,j,local=tl.program_id(0),tl.program_id(1),tl.program_id(2);bh=local+HEAD0;b,h=bh//H,bh%H
 qs=tile*16+tl.arange(0,16);ds=tl.arange(0,D);ts=tl.arange(0,64);kt=j*64+ts
 flag=tl.load(FLAGS+(bh*N+qs)*N+j,qs<N,-2);mix=(qs<N)&((qs+1)*64>QSKIP)&(flag>=0)
 count=tl.zeros((16,),tl.float32);mean=tl.zeros((16,),tl.float32);m2=tl.zeros((16,),tl.float32)
 if tl.sum(mix.to(tl.int32),0)>0:
  qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
  key=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  scores=tl.dot(qb,tl.trans(key))*SCALE*1.4426950408889634
  ns=tl.minimum(64,T-j*64).to(tl.float32);av=tl.sum(tl.where((kt<T)[None,:],scores,0.),1)/ns
  vv=tl.sum(tl.where((kt<T)[None,:],(scores-av[:,None])*(scores-av[:,None]),0.),1)
  count=tl.where(mix,ns,0.);mean=tl.where(mix,av,0.);m2=tl.where(mix,vv,0.)
  tl.store(SCORES+(local*N+qs[:,None])*T+kt[None,:],scores,mix[:,None]&(kt<T)[None,:])
 ix=((local*N+qs)*N+j)*3;tl.store(P+ix,count,qs<N);tl.store(P+ix+1,mean,qs<N);tl.store(P+ix+2,m2,qs<N)
@triton.jit
def cached_route(SCORES,FLAGS,BITS,CUT,T:tl.constexpr,N:tl.constexpr,HEAD0:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr):
 tile,j,local=tl.program_id(0),tl.program_id(1),tl.program_id(2);bh=local+HEAD0;qs=tile*BQ+tl.arange(0,BQ);ts=tl.arange(0,64);kt=j*64+ts
 flag=tl.load(FLAGS+(bh*N+qs)*N+j,qs<N,-2);cut=tl.load(CUT+bh*N+qs,qs<N,float('inf'));mix=(qs<N)&((qs+1)*64>QSKIP)&(flag>=0)&(cut<float('inf'))
 scores=tl.load(SCORES+(local*N+qs[:,None])*T+kt[None,:],mix[:,None]&(kt<T)[None,:],float('-inf'))
 chosen=mix[:,None]&(kt<T)[None,:]&(scores>cut[:,None]);count=tl.sum(chosen.to(tl.int32),1);one=tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64);bits=tl.sum(tl.where(chosen,one[None,:],0),1)
 tl.store(FLAGS+(bh*N+qs)*N+j,tl.where(flag>=0,count,flag),qs<N);tl.store(BITS+(bh*N+qs)*N+j,bits,qs<N)
def threshold_route(qb,k,flags,bits,cut,*,t,n,h,d,scale,qskip,beta,chunk=14,route_qtile=32):
 heads=k.shape[0]*h;chunk=min(chunk,heads);assert chunk>0 and route_qtile in (16,32)
 key=(tuple(k.shape),k.device,chunk)
 if key not in CACHE:
  CACHE[key]=(torch.empty((chunk,n,n,3),device=k.device,dtype=torch.float32),torch.empty((chunk,n,t),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32))
 p,scores,mu,sigma=CACHE[key]
 for begin in range(0,heads,chunk):
  amount=min(chunk,heads-begin)
  partial_and_score[(triton.cdiv(n,16),n,amount)](qb,k,flags,p,scores,t,n,h,d,scale,qskip,begin,num_warps=4,num_stages=1)
  combine_moments[(n,amount)](p,cut[begin:begin+amount],mu[begin:begin+amount],sigma[begin:begin+amount],t,n,triton.next_power_of_2(n),beta,num_warps=4)
  cached_route[(triton.cdiv(n,route_qtile),n,amount)](scores,flags,bits,cut,t,n,begin,qskip,route_qtile,num_warps=4,num_stages=1)
 LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,current_score_cache=scores,chunk=chunk,route_qtile=route_qtile,score_cache_bytes=scores.numel()*scores.element_size(),old_selection_reused=False)
