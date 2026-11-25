"""Compute moments across eligible keys with multiple 64-key blocks per CTA.
"""
import torch,triton
import triton.language as tl
from .sol3_global_mu import combine_moments
CACHE={};LAST={}
@triton.jit
def tiled_partial(QB,K,FLAGS,P,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
 tile,jtile,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);b,h=bh//H,bh%H
 NB:tl.constexpr=BK//64
 qs=tile*BQ+tl.arange(0,BQ);js=jtile*NB+tl.arange(0,NB);ds=tl.arange(0,D);kt=jtile*BK+tl.arange(0,BK)
 flag=tl.load(FLAGS+(bh*N+qs[:,None])*N+js[None,:],(qs<N)[:,None]&(js<N)[None,:],-2)
 mix=(qs<N)[:,None]&((qs+1)*64>QSKIP)[:,None]&(js<N)[None,:]&(flag>=0)
 count=tl.zeros((BQ,NB),tl.float32);mean=tl.zeros((BQ,NB),tl.float32);m2=tl.zeros((BQ,NB),tl.float32)
 if tl.sum(tl.sum(mix.to(tl.int32),1),0)>0:
  q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
  k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  score=tl.dot(q,tl.trans(k))*SCALE*1.4426950408889634
  chunks=tl.reshape(score,(BQ,NB,64));valid=tl.reshape(kt<T,(NB,64));ns=tl.maximum(tl.minimum(64,T-js*64),1).to(tl.float32)
  av=tl.sum(tl.where(valid[None,:,:],chunks,0.),2)/ns[None,:]
  vv=tl.sum(tl.where(valid[None,:,:],(chunks-av[:,:,None])*(chunks-av[:,:,None]),0.),2)
  count=tl.where(mix,ns[None,:],0.);mean=tl.where(mix,av,0.);m2=tl.where(mix,vv,0.)
 ix=((bh*N+qs[:,None])*N+js[None,:])*3;mask=(qs<N)[:,None]&(js<N)[None,:]
 tl.store(P+ix,count,mask);tl.store(P+ix+1,mean,mask);tl.store(P+ix+2,m2,mask)
def threshold(qb,k,flags,cut,*,t,n,h,d,scale,qskip,beta,bq=16,bk=128,warps=4,stages=1):
 assert bq in (16,32) and bk in (128,256) and warps==4 and stages in (1,2)
 key=(tuple(k.shape),k.device)
 if key not in CACHE:
  CACHE[key]=(torch.empty((k.shape[0]*h,n,n,3),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device,dtype=torch.float32))
 p,mu,sigma=CACHE[key]
 tiled_partial[(triton.cdiv(n,bq),triton.cdiv(n,bk//64),k.shape[0]*h)](qb,k,flags,p,t,n,h,d,scale,qskip,bq,bk,num_warps=warps,num_stages=stages)
 combine_moments[(n,k.shape[0]*h)](p,cut,mu,sigma,t,n,triton.next_power_of_2(n),beta,num_warps=4)
 LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p,bq=bq,bk=bk,stages=stages)
