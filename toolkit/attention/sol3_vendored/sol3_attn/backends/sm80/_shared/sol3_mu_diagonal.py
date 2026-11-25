"""Stream block moments for global mu+beta3 sigma: an explicit diagonal covariance approximation."""
import torch,triton
import triton.language as tl
CACHE={};LAST={}
@triton.jit
def block_moments(QB,KC,KV,FLAGS,CUT,MU,SD,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,QSKIP:tl.constexpr,BETA:tl.constexpr,RQ:tl.constexpr,BK:tl.constexpr):
 tile,bh=tl.program_id(0),tl.program_id(1);qs=tile*RQ+tl.arange(0,RQ);ds=tl.arange(0,D);keys=tl.arange(0,BK)
 active=(qs<N)&((qs+1)*64>QSKIP)
 q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0).to(tl.bfloat16)
 qf=q.to(tl.float32);q2=qf*qf;amount=tl.zeros((RQ,),tl.float32);mean=tl.zeros((RQ,),tl.float32);m2=tl.zeros((RQ,),tl.float32)
 for base in range(tl.cdiv(N,BK)):
  js=base*BK+keys;f=tl.load(FLAGS+(bh*N+qs[:,None])*N+js[None,:],(qs<N)[:,None]&(js<N)[None,:],-2)
  weights=tl.where(active[:,None]&(js<N)[None,:]&(f>=0),tl.maximum(0,tl.minimum(64,T-js*64)).to(tl.float32)[None,:],0.)
  if tl.sum(tl.sum(weights,1),0)>0:
   kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0)
   kv=tl.load(KV+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0)
   scores=tl.dot(q,tl.trans(kc))*SCALE*1.4426950408889634
   within=tl.maximum(tl.dot(q2,tl.trans(kv),input_precision='tf32x3'),0.)*SCALE*SCALE*1.4426950408889634*1.4426950408889634
   count=tl.sum(weights,1);mu=tl.sum(weights*scores,1)/tl.maximum(count,1.)
   varsum=tl.sum(weights*(within+(scores-mu[:,None])*(scores-mu[:,None])),1)
   total=amount+count;delta=mu-mean
   m2=m2+varsum+delta*delta*amount*count/tl.maximum(total,1.)
   mean=mean+delta*count/tl.maximum(total,1.);amount=total
 sd=tl.sqrt(tl.maximum(m2/tl.maximum(amount,1.),0.)+1.e-6)
 cut=tl.where(active&(amount>0),mean+BETA*sd,float('inf'))
 tl.store(CUT+bh*N+qs,cut,qs<N);tl.store(MU+bh*N+qs,mean,qs<N);tl.store(SD+bh*N+qs,sd,qs<N)
def threshold(qb,kc,kv,flags,cut,*,t,n,h,d,scale,qskip,beta):
 key=(tuple(qb.shape),qb.device)
 if key not in CACHE:CACHE[key]=(torch.empty((qb.shape[0],n),device=qb.device,dtype=torch.float32),torch.empty((qb.shape[0],n),device=qb.device,dtype=torch.float32))
 mu,sd=CACHE[key]
 block_moments[(triton.cdiv(n,16),qb.shape[0])](qb,kc,kv,flags,cut,mu,sd,t,n,h,d,scale,qskip,beta,16,64,num_warps=4,num_stages=1)
 LAST.clear();LAST.update(mu=mu,sigma=sd,cut=cut,statistical_scope='All eligible mixed blocks with valid-token weighting; BF16 block mean proxy plus within-block diagonal Key variance; off-diagonal covariance omitted.')
