# H3 quantile routing with fused block classification.
import math,torch,triton
import triton.language as tl
from .block_selector import *
from .tiled_selection import global_route_pair
CACHE={}
@triton.jit
def risk_pool_class(QB,KC,C,M,KV,RISK,FLAGS,TH2,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,CAP:tl.constexpr,ALPHA:tl.constexpr,SINK0:tl.constexpr,SINK1:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr=16,BK:tl.constexpr=64):
 tile,bh=tl.program_id(0),tl.program_id(1);qs=tile*BQ+tl.arange(0,BQ);ds=tl.arange(0,D);js0=tl.arange(0,BK)
 active=(qs<N)&((qs+1)*64>QSKIP)
 q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0);q2=q*q
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds)
 s=SCALE*1.4426950408889634;mu=tl.sum(q*mk[None,:],1)*s;var=tl.sum(q2*vk[None,:],1)*s*s
 th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6);ss=tl.zeros((BQ,),tl.float32);ss2=tl.zeros((BQ,),tl.float32)
 for start in range(0,N,BK):
  js=start+js0;valid=js<N
  kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],valid[:,None],0).to(tl.float32)
  c=tl.load(C+bh*N+js,valid,0)
  proxy=tl.dot(q,tl.trans(kc),input_precision='tf32x3')*s
  score=proxy+c[None,:]
  if CAP>0:
   kv=tl.load(KV+(bh*N+js[:,None])*D+ds[None,:],valid[:,None],0)
   delta=.5*SCALE*SCALE*tl.dot(q2,tl.trans(kv),input_precision='tf32x3')
   score+=ALPHA*tl.minimum(tl.maximum(delta,0.),CAP)*1.4426950408889634
  full=active[:,None]&((proxy>th1[:,None])|(tl.abs(qs[:,None]-js[None,:])<=1)|((js>=SINK0)&(js<SINK1))[None,:])
  ix=(bh*N+qs[:,None])*N+js[None,:];mask=(qs<N)[:,None]&valid[None,:]
  tl.store(RISK+ix,score,mask);tl.store(FLAGS+ix,tl.where(full,-1,0),mask)
  ss+=tl.sum(tl.where(valid[None,:],score,0.),1)
  if B2!=0:ss2+=tl.sum(tl.where(valid[None,:],score*score,0.),1)
 mean=ss/N
 if B2!=0:
  variance=tl.maximum(ss2/N-mean*mean,0.);mean+=B2*tl.sqrt(variance+1.e-6)
 tl.store(TH2+bh*N+qs,mean,qs<N)
@triton.jit
def finish_class(RISK,FLAGS,TH2,N:tl.constexpr,TOTAL:tl.constexpr,QSKIP:tl.constexpr,BLOCK:tl.constexpr=1024):
 ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=ix<TOTAL;query=(ix//N)%N;head=ix//(N*N)
 risk=tl.load(RISK+ix,valid,0.);flag=tl.load(FLAGS+ix,valid,0);cut=tl.load(TH2+head*N+query,valid,0.)
 pure=(risk<cut)&((query+1)*64>QSKIP)
 tl.store(FLAGS+ix,tl.where(flag==-1,-1,tl.where(pure,-2,0)),valid)

def route_only(q,k,v,*,cfg,scale=None,tau=1.,thresh_type='diag',kv_splits=1,sink_tokens=0,sink_start=None,**kw):
 assert cfg.get('global_budget') is not None and not cfg.get('risk_value_contrast',False) and cfg.get('value_mode','fixed')=='fixed'
 assert q.shape==k.shape==v.shape and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
 b,t,h,d=q.shape;n=triton.cdiv(t,64);s=d**-.5 if scale is None else scale
 st=(t-sink_tokens if sink_start is None else sink_start) if sink_tokens else 0;s0=st//64;s1=triton.cdiv(st+sink_tokens,64) if sink_tokens else 0;qskip=sink_tokens if cfg.get('skip_prefix',True) and st==0 else 0
 cap=float('inf') if cfg.get('disable_risk_cap',True) else float(cfg.get('risk_keyvar_cap',0.));alpha=float(cfg.get('risk_keyvar_alpha',1.))
 kc,vm,c,mom,kv,qb,cut,flags,bits,off,full,counts,index,us,ls,ms,th2=buffers(q,1,cfg.get('summary_dtype','bf16')=='bf16',True)
 pool_qkv[(n,b*h)](q,k,v,qb,kc,vm,c,kv,t,n,h,d,True,num_warps=4,num_stages=1)
 moments_fast[(b*h,)](kc,c,mom,t,n,h,d,num_warps=4,num_stages=1)
 key=(tuple(q.shape),q.device)
 if key not in CACHE:CACHE[key]=torch.empty((b*h,n,n),device=q.device,dtype=torch.float32)
 risk=CACHE[key]
 risk_pool_class[(triton.cdiv(n,16),b*h)](qb,kc,c,mom,kv,risk,flags,th2,t,n,h,d,s,float(cfg['beta1']),float(cfg['beta2']),cap,alpha,s0,s1,qskip,num_warps=4,num_stages=1)
 finish_class[(triton.cdiv(b*h*n*n,1024),)](risk,flags,th2,n,b*h*n*n,qskip,num_warps=4)
 samples=cfg.get('threshold_samples',1024);skey=(tuple(q.shape),q.device,samples)
 if skey not in SAMPLE_CACHE:SAMPLE_CACHE[skey]=torch.empty((b*h,n,samples),device=q.device,dtype=torch.float32)
 scores=SAMPLE_CACHE[skey]
 if skey not in GOAL_CACHE:GOAL_CACHE[skey]=torch.empty((b*h,n),device=q.device,dtype=torch.int32)
 goals=GOAL_CACHE[skey]
 budget_goals[(n,b*h)](flags,goals,t,n,triton.next_power_of_2(n),float(cfg['global_budget']),num_warps=4)
 sample_scores[(triton.cdiv(n,16),triton.cdiv(samples,128),b*h)](qb,k,flags,goals,scores,t,n,h,d,samples,s,True,num_warps=4,num_stages=1)
 radix_cut[(n,b*h)](scores,goals,cut,t,n,samples,num_warps=cfg.get('cut_warps',4))
 rt=cfg.get('route_qtile',32)
 global_route_pair[(triton.cdiv(n,rt),triton.cdiv(n,2),b*h)](qb,k,flags,bits,cut,s,t,n,h,d,True,qskip,rt,num_warps=4,num_stages=1)
 return dict(b=b,t=t,h=h,d=d,n=n,s=s,qskip=qskip,kc=kc,vm=vm,kv=kv,qb=qb,flags=flags,bits=bits,off=off,full=full,counts=counts)
