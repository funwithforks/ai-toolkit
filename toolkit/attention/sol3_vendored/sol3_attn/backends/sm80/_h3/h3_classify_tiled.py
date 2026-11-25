import triton
import triton.language as tl
@triton.jit
def classify_blocks_tiled(QB,KC,C,M,KV,TH2,FLAGS,
 SCALE:tl.constexpr,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
 B1:tl.constexpr,B2:tl.constexpr,SINK0:tl.constexpr,SINK1:tl.constexpr,USE_TH2:tl.constexpr,QSKIP:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr,RISK_CAP:tl.constexpr,RISK_ALPHA:tl.constexpr):
 qi,kj,bh=tl.program_id(0),tl.program_id(1),tl.program_id(2)
 qs=qi*BQ+tl.arange(0,BQ);js=kj*BK+tl.arange(0,BK);ds=tl.arange(0,D)
 q=tl.load(QB+(bh*N+qs[:,None])*D+ds[None,:],(qs<N)[:,None],0)
 kc=tl.load(KC+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0).to(tl.float32)
 mk=tl.load(M+bh*(2*D+2)+ds);vk=tl.load(M+bh*(2*D+2)+D+ds)
 mc=tl.load(M+bh*(2*D+2)+2*D);vc=tl.load(M+bh*(2*D+2)+2*D+1)
 s=SCALE*1.4426950408889634
 mu=tl.sum(q*mk[None,:],1)*s;var=tl.sum(q*q*vk[None,:],1)*s*s
 th1=mu+B1*tl.sqrt(tl.maximum(var,0)+1.e-6)
 th2=mu+mc+B2*tl.sqrt(tl.maximum(var+vc,0)+1.e-6)
 if USE_TH2:th2=tl.load(TH2+bh*N+qs,qs<N,0)
 proxy=tl.dot(q,tl.trans(kc),input_precision='tf32x3')*s
 c=tl.load(C+bh*N+js,js<N,0);risk=proxy+c[None,:]
 if RISK_CAP>0:
  kv=tl.load(KV+(bh*N+js[:,None])*D+ds[None,:],(js<N)[:,None],0)
  delta=.5*SCALE*SCALE*tl.dot(q*q,tl.trans(kv),input_precision='tf32x3')
  risk+=RISK_ALPHA*tl.minimum(delta,RISK_CAP)*1.4426950408889634
 active=(qs<N)&((qs+1)*64>QSKIP)
 ex=active[:,None]&((proxy>th1[:,None])|(tl.abs(qs[:,None]-js[None,:])<=1)|((js>=SINK0)&(js<SINK1))[None,:])
 pure=active[:,None]&~ex&(risk<th2[:,None])
 flags=tl.where(ex,-1,tl.where(pure,-2,0))
 ix=(bh*N+qs[:,None])*N+js[None,:]
 tl.store(FLAGS+ix,flags,(qs<N)[:,None]&(js<N)[None,:])
