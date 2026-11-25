from sol3_attn.backends.sm80.routing import launch_selection
"""Current-score tiled raw-mu cache eliminates second K-score GEMM.

Every eligible original Key is still scanned; no sampled/covariance moments.
The FP32 score cache holds one head chunk at a time.
"""
import torch,triton
import triton.language as tl
from .sol3_global_mu import combine_moments
@triton.jit
def cached_route(SCORES,FLAGS,BITS,CUT,T:tl.constexpr,N:tl.constexpr,HEAD0:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr):
 tile,j,local=tl.program_id(0),tl.program_id(1),tl.program_id(2);bh=local+HEAD0;qs=tile*BQ+tl.arange(0,BQ);ts=tl.arange(0,64);kt=j*64+ts
 flag=tl.load(FLAGS+(bh*N+qs)*N+j,qs<N,-2);cut=tl.load(CUT+bh*N+qs,qs<N,float('inf'));mix=(qs<N)&((qs+1)*64>QSKIP)&(flag>=0)&(cut<float('inf'))
 scores=tl.load(SCORES+(local*N+qs[:,None]).to(tl.int64)*T+kt[None,:],mix[:,None]&(kt<T)[None,:],float('-inf'))
 chosen=mix[:,None]&(kt<T)[None,:]&(scores>cut[:,None]);count=tl.sum(chosen.to(tl.int32),1);one=tl.full((64,),1,tl.uint64)<<ts.to(tl.uint64);bits=tl.sum(tl.where(chosen,one[None,:],0),1)
 tl.store(FLAGS+(bh*N+qs)*N+j,tl.where(flag>=0,count,flag),qs<N);tl.store(BITS+(bh*N+qs)*N+j,bits,qs<N)
CACHE={};LAST={}
@triton.jit
def partial_scores(QB,K,FLAGS,P,SCORE,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,HEAD0:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
 tile,jtile,local=tl.program_id(0),tl.program_id(1),tl.program_id(2);bh=local+HEAD0;b,h=bh//H,bh%H;NB:tl.constexpr=BK//64
 qs=tile*BQ+tl.arange(0,BQ);js=jtile*NB+tl.arange(0,NB);ds=tl.arange(0,D);kt=jtile*BK+tl.arange(0,BK)
 flag=tl.load(FLAGS+(bh*N+qs[:,None])*N+js[None,:],(qs<N)[:,None]&(js<N)[None,:],-2)
 mix=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(js<N)[None,:]&(flag>=0)
 count=tl.zeros((BQ,NB),tl.float32);mean=tl.zeros((BQ,NB),tl.float32);m2=tl.zeros((BQ,NB),tl.float32)
 if tl.sum(tl.sum(mix.to(tl.int32),1),0)>0:
  q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
  key=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  scores=tl.dot(q,tl.trans(key))*SCALE*1.4426950408889634
  chunks=tl.reshape(scores,(BQ,NB,64));valid=tl.reshape(kt<T,(NB,64));ns=tl.maximum(tl.minimum(64,T-js*64),1).to(tl.float32)
  av=tl.sum(tl.where(valid[None,:,:],chunks,0.),2)/ns[None,:];vv=tl.sum(tl.where(valid[None,:,:],(chunks-av[:,:,None])*(chunks-av[:,:,None]),0.),2)
  count=tl.where(mix,ns[None,:],0.);mean=tl.where(mix,av,0.);m2=tl.where(mix,vv,0.)
  active=tl.reshape(tl.broadcast_to(mix[:,:,None],(BQ,NB,64)),(BQ,BK))&(kt<T)[None,:]
  tl.store(SCORE+(local*N+qs[:,None]).to(tl.int64)*T+kt[None,:],scores,active)
 ix=((local*N+qs[:,None])*N+js[None,:])*3;mask=(qs<N)[:,None]&(js<N)[None,:]
 tl.store(P+ix,count,mask);tl.store(P+ix+1,mean,mask);tl.store(P+ix+2,m2,mask)
def threshold_route(qb,k,flags,bits,cut,*,t,n,h,d,scale,qskip,beta,chunk=10,bq=32,bk=256,stages=2,route_qtile=32):
 heads=k.shape[0]*h;chunk=min(int(chunk),heads);assert chunk>0 and route_qtile in (16,32)
 key=(tuple(k.shape),k.device,chunk)
 if key not in CACHE:
  CACHE[key]=(torch.empty((chunk,n,n,3),device=k.device,dtype=torch.float32),torch.empty((chunk,n,t),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32),torch.empty((heads,n),device=k.device,dtype=torch.float32))
 p,scores,mu,sigma=CACHE[key]
 for start in range(0,heads,chunk):
  amount=min(chunk,heads-start)
  partial_scores[(triton.cdiv(n,bq),triton.cdiv(n,bk//64),amount)](qb,k,flags,p,scores,t,n,h,d,scale,qskip,start,bq,bk,num_warps=4,num_stages=stages)
  combine_moments[(n,amount)](p,cut[start:start+amount],mu[start:start+amount],sigma[start:start+amount],t,n,triton.next_power_of_2(n),beta,num_warps=4)
  launch_selection(cached_route, ((triton.cdiv(n,route_qtile),n,amount)), scores,flags,bits,cut,t,n,start,qskip,route_qtile,num_warps=4,num_stages=1)
 LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,bq=bq,bk=bk,stages=stages,current_score_cache=scores,chunk=chunk,score_cache_bytes=scores.numel()*scores.element_size(),old_selection_reused=False)
