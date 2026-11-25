"""Merge representatives weighted by q·mean(K) + log(valid remaining count)."""
import triton
import triton.language as tl
@triton.jit
def rep_merge_direct(Q,KC,VM,FLAGS,OS,LS,O,
 SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
 FIXED:tl.constexpr,QSKIP:tl.constexpr,BK:tl.constexpr,VM_BF16:tl.constexpr,USE_ACC:tl.constexpr,SKIPTOK:tl.constexpr,LSE_BASE2:tl.constexpr,HEAD_BASE:tl.constexpr):
 qi,bh=tl.program_id(0),tl.program_id(1)+HEAD_BASE;b,h=bh//H,bh%H
 if (qi+1)*64<=QSKIP:return
 qs=qi*64+tl.arange(0,64);ds=tl.arange(0,D);js0=tl.arange(0,BK);vq=qs<T
 q=tl.load(Q+((b*T+qs[:,None]).to(tl.int64)*H+h)*D+ds[None,:],vq[:,None],0)
 acc=tl.zeros((64,D),tl.float32);den=tl.zeros((64,),tl.float32);mx=tl.full((64,),-float('inf'),tl.float32)
 s=SCALE*1.4426950408889634
 for start in range(0,N,BK):
  js=start+js0;vj=js<N;flag=tl.load(FLAGS+(bh*N+qi)*N+js,vj,-1)
  ns=tl.minimum(64,tl.maximum(T-js*64,0)).to(tl.float32)
  count=ns-tl.maximum(flag,0).to(tl.float32) if FIXED else ns
  keep=vj&(flag!=-1)&(count>0)
  kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],vj[:,None],0)
  scores=tl.dot(q,tl.trans(kc))*s
  scores+=tl.log2(tl.maximum(count,1.))[None,:]
  scores=tl.where(keep[None,:],scores,-float('inf'));has=tl.sum(keep.to(tl.int32),0)>0
  nm=tl.where(has,tl.maximum(mx,tl.max(scores,1)),mx)
  alpha=tl.exp2(tl.where(has,mx-nm,0.))
  p=tl.where(keep[None,:],tl.exp2(scores-tl.where(has,nm,0.)[:,None]),0.)
  vm=tl.load(VM+(bh*N+js[:,None])*D+ds[None,:],vj[:,None],0)
  if VM_BF16:
   if USE_ACC:acc=tl.dot(p.to(tl.bfloat16),vm,acc*alpha[:,None])
   else:acc=acc*alpha[:,None]+tl.dot(p.to(tl.bfloat16),vm)
  else:
   if USE_ACC:acc=tl.dot(p,vm,acc*alpha[:,None],input_precision='tf32x3')
   else:acc=acc*alpha[:,None]+tl.dot(p,vm,input_precision='tf32x3')
  den=den*alpha+tl.sum(p,1);mx=nm
 ix=((b*T+qs[:,None]).to(tl.int64)*H+h)*D+ds[None,:];ro=(b*T+qs)*H+h
 live=vq&(qs>=QSKIP)
 packed=(bh-HEAD_BASE)*(T-SKIPTOK)+qs-SKIPTOK
 selected=tl.load(OS+packed[:,None].to(tl.int64)*D+ds[None,:],live[:,None],0.).to(tl.float32)
 lse=tl.load(LS+packed,live,-float('inf'))
 if not LSE_BASE2:lse*=1.4426950408889634
 # Algebraically combine the unnormalized representative accumulator and
 # the normalized exact-token output. Avoid log2(den) and acc/den entirely.
 common=tl.maximum(mx,lse)
 wr=tl.exp2(mx-common);ws=tl.exp2(lse-common)
 out=(wr[:,None]*acc+ws[:,None]*selected)/(wr*den+ws)[:,None]
 tl.store(O+ix,out,live[:,None])
