"""Compute third moments over all eligible keys in two streaming passes."""
import torch,triton
import triton.language as tl
CACHE={};LAST={}
@triton.jit
def partial_moments(QB,K,FLAGS,P,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr):
 tile,j,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);b,h=bh//H,bh%H
 qs=tile*16+tl.arange(0,16);ds=tl.arange(0,D);ts=tl.arange(0,64);kt=j*64+ts
 flag=tl.load(FLAGS+(bh*N+qs)*N+j,qs<N,-2);mix=(qs<N)&((qs+1)*64>QSKIP)&(flag>=0)
 count=tl.zeros((16,),tl.float32);mean=tl.zeros((16,),tl.float32);m2=tl.zeros((16,),tl.float32)
 if tl.sum(mix.to(tl.int32),0)>0:
  qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
  k=tl.load(K+((b*T+kt[:,None]).to(tl.int64)*H+h)*D+ds[None,:],(kt<T)[:,None],0)
  scores=tl.dot(qb,tl.trans(k))*SCALE*1.4426950408889634
  ns=tl.minimum(64,T-j*64).to(tl.float32)
  av=tl.sum(tl.where((kt<T)[None,:],scores,0.),1)/ns
  vv=tl.sum(tl.where((kt<T)[None,:],(scores-av[:,None])*(scores-av[:,None]),0.),1)
  count=tl.where(mix,ns,0.);mean=tl.where(mix,av,0.);m2=tl.where(mix,vv,0.)
 ix=((bh*N+qs)*N+j)*3;tl.store(P+ix,count,qs<N);tl.store(P+ix+1,mean,qs<N);tl.store(P+ix+2,m2,qs<N)
@triton.jit
def combine_moments(P,CUT,MU,SIGMA,T:tl.constexpr,N:tl.constexpr,NN:tl.constexpr,BETA:tl.constexpr):
 qi,bh=tl.program_id(0),tl.program_id(1);js=tl.arange(0,NN);ix=((bh*N+qi)*N+js)*3
 count=tl.load(P+ix,js<N,0.);mean=tl.load(P+ix+1,js<N,0.);m2=tl.load(P+ix+2,js<N,0.)
 amount=tl.sum(count,0);mu=tl.sum(count*mean,0)/tl.maximum(amount,1.)
 var=tl.sum(m2+count*(mean-mu)*(mean-mu),0)/tl.maximum(amount,1.)
 sigma=tl.sqrt(tl.maximum(var,0.)+1.e-6);cut=tl.where(amount>0,mu+BETA*sigma,float('inf'))
 tl.store(CUT+bh*N+qi,cut);tl.store(MU+bh*N+qi,mu);tl.store(SIGMA+bh*N+qi,sigma)
def threshold(qb,k,flags,cut,*,t,n,h,d,scale,qskip,beta):
 key=(tuple(k.shape),k.device)
 if key not in CACHE:
  CACHE[key]=(torch.empty((k.shape[0]*h,n,n,3),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device,dtype=torch.float32),torch.empty((k.shape[0]*h,n),device=k.device,dtype=torch.float32))
 p,mu,sigma=CACHE[key]
 partial_moments[(triton.cdiv(n,16),n,k.shape[0]*h)](qb,k,flags,p,t,n,h,d,scale,qskip,num_warps=4,num_stages=1)
 combine_moments[(n,k.shape[0]*h)](p,cut,mu,sigma,t,n,triton.next_power_of_2(n),beta,num_warps=4)
 LAST.clear();LAST.update(mu=mu,sigma=sigma,cut=cut,partial=p)
