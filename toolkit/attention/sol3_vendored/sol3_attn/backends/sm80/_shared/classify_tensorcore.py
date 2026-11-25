"""Block classification using TF32x3 dot products and FP32 accumulation."""
import triton
import triton.language as tl
@triton.jit
def classify_tiled(QB,KC,C,M,KV,TH2,FLAGS,T:tl.constexpr,N:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,B1:tl.constexpr,SINK0:tl.constexpr,SINK1:tl.constexpr,QSKIP:tl.constexpr,RISK_ALPHA:tl.constexpr,BQ:tl.constexpr=32,BK:tl.constexpr=16):
 qt,kt,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2);qs=qt*BQ+tl.arange(0,BQ);js=kt*BK+tl.arange(0,BK);ds=tl.arange(0,D)
 active=(qs<N)&((qs+1)*64>QSKIP);q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0);q2=q*q
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds);s=SCALE*1.4426950408889634
 mu=tl.sum(q*mk[None,:],1)*s;var=tl.sum(q2*vk[None,:],1)*s*s;th1=mu+B1*tl.sqrt(tl.maximum(var,0.)+1.e-6);th2=tl.load(TH2+bh*N+qs,qs<N,0.)
 kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0).to(tl.float32);c=tl.load(C+bh*N+js,js<N,0.)
 kv=tl.load(KV+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0)
 proxy=tl.dot(q,tl.trans(kc),input_precision='tf32x3')*s;delta=.5*SCALE*SCALE*tl.dot(q2,tl.trans(kv),input_precision='tf32x3');risk=proxy+c[None,:]+RISK_ALPHA*delta*1.4426950408889634
 full=active[:,None]&((proxy>th1[:,None])|(tl.abs(qs[:,None]-js[None,:])<=1)|((js>=SINK0)&(js<SINK1))[None,:])
 pure=active[:,None]&~full&(risk<th2[:,None]);flag=tl.where(full,-1,tl.where(pure,-2,0));ix=(bh*N+qs[:,None])*N+js[None,:]
 tl.store(FLAGS+ix,flag,(qs<N)[:,None]&(js<N)[None,:])
