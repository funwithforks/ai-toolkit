"""H3 three-layer route with complete mixed raw-key global mu selection."""
import math,torch,triton
from .block_selector import *
from .tiled_selection import global_route_pair
from .fused_risk_routing import risk_pool_class, finish_class
CACHE={};LAST={}
def route_only(q,k,v,*,cfg,scale=None,tau=1.,thresh_type='diag',kv_splits=1,sink_tokens=0,sink_start=None,**kw):
 assert cfg.get('global_budget') is None and cfg.get('third_selection')=='global_mu_sigma'
 assert not cfg.get('risk_value_contrast',False) and cfg.get('value_mode','fixed')=='fixed'
 assert q.shape==k.shape==v.shape and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
 b,t,h,d=q.shape;n=triton.cdiv(t,64);s=d**-.5 if scale is None else scale
 st=(t-sink_tokens if sink_start is None else sink_start) if sink_tokens else 0;s0=st//64;s1=triton.cdiv(st+sink_tokens,64) if sink_tokens else 0;qskip=sink_tokens if cfg.get('skip_prefix',True) and st==0 else 0
 riskcap=float('inf') if cfg.get('disable_risk_cap',True) else float(cfg.get('risk_keyvar_cap',0.));alpha=float(cfg.get('risk_keyvar_alpha',1.))
 kc,vm,c,mom,kv,qb,cut,flags,bits,off,full,counts,index,us,ls,ms,th2=buffers(q,1,cfg.get('summary_dtype','bf16')=='bf16',True)
 pool_qkv[(n,b*h)](q,k,v,qb,kc,vm,c,kv,t,n,h,d,True,num_warps=4,num_stages=1)
 moments_fast[(b*h,)](kc,c,mom,t,n,h,d,num_warps=4,num_stages=1)
 if cfg.get('fused_risk_classify',False):
  key=(tuple(q.shape),q.device)
  if key not in CACHE:CACHE[key]=torch.empty((b*h,n,n),device=q.device,dtype=torch.float32)
  risk=CACHE[key]
  risk_pool_class[(triton.cdiv(n,16),b*h)](qb,kc,c,mom,kv,risk,flags,th2,t,n,h,d,s,float(cfg['beta1']),float(cfg['beta2']),riskcap,alpha,s0,s1,qskip,num_warps=4,num_stages=1)
  finish_class[(triton.cdiv(b*h*n*n,1024),)](risk,flags,th2,n,b*h*n*n,qskip,num_warps=4)
 else:
  risk_keyvar_threshold[(triton.cdiv(n,16),b*h)](qb,kc,c,kv,c,th2,t,n,h,d,s,float(cfg['beta2']),riskcap,alpha,qskip,16,64,False,num_warps=4,num_stages=1)
  rt=cfg.get('route_qtile',32)
  classify_blocks[(triton.cdiv(n,rt),n,b*h)](qb,k,kc,c,mom,kv,c,th2,flags,bits,s,t,n,h,d,float(cfg['beta1']),float(cfg['beta2']),float(cfg['beta3']),s0,s1,True,True,qskip,rt,riskcap,alpha,False,num_warps=4,num_stages=1)
 if cfg.get('mu_moment_impl')=='cached_tiled':
  from ._cached_dispatch import route
  route(qb,k,flags,bits,cut,t=t,n=n,h=h,d=d,scale=s,qskip=qskip,beta=float(cfg['beta3']),bq=int(cfg.get('mu_bq',32)),bk=int(cfg.get('mu_bk',256)))
 else:
  from .sol3_global_mu import threshold
  threshold(qb,k,flags,cut,t=t,n=n,h=h,d=d,scale=s,qskip=qskip,beta=float(cfg['beta3']))
  rt=cfg.get('route_qtile',32)
  global_route_pair[(triton.cdiv(n,rt),triton.cdiv(n,2),b*h)](qb,k,flags,bits,cut,s,t,n,h,d,True,qskip,rt,num_warps=4,num_stages=1)
 LAST.clear();LAST.update(cut=cut)
 return dict(b=b,t=t,h=h,d=d,n=n,s=s,qskip=qskip,kc=kc,vm=vm,kv=kv,qb=qb,flags=flags,bits=bits,off=off,full=full,counts=counts)
