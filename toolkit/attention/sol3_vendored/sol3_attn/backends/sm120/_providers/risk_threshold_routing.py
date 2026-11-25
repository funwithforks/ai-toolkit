"""Compute risk thresholds and route blocks for sparse attention."""

import torch,triton,triton.language as tl
from . import sol3 as base
@triton.jit
def risk_threshold(QB,KC,C,M,OUT,B2,
 SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,B1:tl.constexpr,
 SINK0:tl.constexpr,SINK1:tl.constexpr,RAW:tl.constexpr,REMAIN:tl.constexpr):
 tile,bh=tl.program_id(0),tl.program_id(1);b,h=bh//H,bh%H
 qs=tile*16+tl.arange(0,16);ds=tl.arange(0,D);js0=tl.arange(0,64)
 qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0)
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds)
 s=SCALE*1.4426950408889634
 mu=tl.sum(qb*mk[None,:],1)*s;var=tl.sum(qb*qb*vk[None,:],1)*s*s
 th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6)
 sm=tl.zeros((16,),tl.float32);ss=tl.zeros((16,),tl.float32);count=tl.zeros((16,),tl.float32)
 rmax=tl.full((16,),-float('inf'),tl.float32)
 for start in range(0,N,64):
  js=start+js0;vj=js<N
  kc=tl.load(KC+((b*N+js[:,None])*H+h)*D+ds[None,:],vj[:,None],0).to(tl.float32)
  c=tl.load(C+(b*N+js)*H+h,vj,0)
  proxy=tl.dot(qb,tl.trans(kc),input_precision="tf32x3")*s
  value=proxy+c[None,:]
  keep=(qs<N)[:,None]&vj[None,:]
  if REMAIN:
   ex=(proxy>th1[:,None])|(tl.abs(qs[:,None]-js[None,:])<=1)|((js>=SINK0)&(js<SINK1))[None,:]
   keep=keep&~ex
  cnt=tl.sum(keep.to(tl.float32),1);count+=cnt
  if RAW:
   active=cnt>0
   nm=tl.maximum(rmax,tl.max(tl.where(keep,value,-float('inf')),1))
   alpha=tl.exp2(tl.where(active,rmax-nm,0.))
   e=tl.where(keep,tl.exp2(value-tl.where(active,nm,0.)[:,None]),0.)
   sm=sm*alpha+tl.sum(e,1);ss=ss*alpha*alpha+tl.sum(e*e,1)
   rmax=tl.where(active,nm,rmax)
  else:
   sm+=tl.sum(tl.where(keep,value,0.),1)
   ss+=tl.sum(tl.where(keep,value*value,0.),1)
 mean=sm/tl.maximum(count,1.);variance=tl.maximum(ss/tl.maximum(count,1.)-mean*mean,0.)
 if RAW:
  cut=mean+B2*tl.sqrt(variance)
  result=tl.where((cut>0.)&(count>0.),tl.log2(tl.maximum(cut,1.e-30))+rmax,-float('inf'))
 else:
  result=tl.where(count>0.,mean+B2*tl.sqrt(variance+1.e-6),-float('inf'))
 tl.store(OUT+bh*N+qs,result,qs<N)


@triton.jit
def route_variant(QB,K,KC,C,M,TH2,FLAGS,MASK,
          SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
          B1:tl.constexpr,B2:tl.constexpr,B3:tl.constexpr,SINK0:tl.constexpr,SINK1:tl.constexpr):
 tile,j,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);b,h=bh//H,bh%H
 qs=tile*16+tl.arange(0,16);ds=tl.arange(0,D);ts=tl.arange(0,64)
 qb=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0)
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds)
 mc=tl.load(M+bh*(2*D+2)+2*D);vc=tl.load(M+bh*(2*D+2)+2*D+1)
 s=SCALE*1.4426950408889634
 mu=tl.sum(qb*mk[None,:],1)*s;var=tl.sum(qb*qb*vk[None,:],1)*s*s
 th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6);th2=tl.load(TH2+bh*N+qs,qs<N,0)
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


EXTRA={}
def variant(q,k,v,*,name,beta2,scale=None,tau=1.,sink_start=None,sink_tokens=0,**kw):
 if name=='log_all_diag':return base.sol3(q,k,v,scale=scale,tau=tau,sink_start=sink_start,sink_tokens=sink_tokens,beta2=beta2,**kw)
 with torch.cuda.device(q.device):
  b,t,h,d=q.shape;n=triton.cdiv(t,64);s=d**-.5 if scale is None else scale
  kc,vs,c,m=base.prepare(k,v);key=(tuple(q.shape),q.device)
  if key not in base.ROUTE_BUFFERS:
   base.sol3(q,k,v,scale=s,tau=tau,sink_start=sink_start,sink_tokens=sink_tokens)
  qb,flags,compact=base.ROUTE_BUFFERS[key]
  out=torch.empty_like(q);th2=EXTRA.get(key)
  if th2 is None:th2=EXTRA[key]=torch.empty((b*h,n),device=q.device,dtype=torch.float32)
  st=(t-sink_tokens if sink_start is None else sink_start) if sink_tokens else 0
  s0=st//64;s1=triton.cdiv(st+sink_tokens,64) if sink_tokens else 0
  b1=tau+base.BETA1_DELTA;b3=base.BETA3
  base.query_pool[(n,b*h)](q,qb,t,n,h,d,num_warps=4,num_stages=1)
  risk_threshold[(triton.cdiv(n,16),b*h)](qb,kc,c,m,th2,float(beta2),s,t,n,h,d,b1,s0,s1,name.startswith('raw'),name.endswith('remaining'),num_warps=4,num_stages=1)
  route_variant[(triton.cdiv(n,16),n,b*h)](qb,k,kc,c,m,th2,flags,compact,s,t,n,h,d,b1,0.,b3,s0,s1,num_warps=4,num_stages=1)
  base.forward[(d//base.BV,n,b*h)](q,k,v,kc,vs,c,m,out,flags,compact,s,t,n,h,d,b1,0.,b3,s0,s1,base.BV,num_warps=base.WARPS,num_stages=1)
  return out
