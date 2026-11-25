import ast,inspect,types,textwrap
import torch
import flashinfer.prefill as prefill
CACHE={}
def vectorize_plan(wrapper):
 original=inspect.unwrap(prefill.BatchPrefillWithPagedKVCacheWrapper.plan);key=(original.__module__,inspect.getsource(original))
 if key not in CACHE:
  source=textwrap.dedent(key[1]);tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef));fn.decorator_list=[]
  class Transformer(ast.NodeTransformer):
   def __init__(self):self.changes=[]
   def visit_Call(self,node):
    node=self.generic_visit(node)
    if isinstance(node.func,ast.Name) and node.func.id=='max' and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id in ['qo_indptr_host','kv_lens_arr_host']:
     self.changes.append(node.args[0].id);node.func=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='amax',ctx=ast.Load())
    return node
  transform=Transformer();tree=transform.visit(tree);ast.fix_missing_locations(tree);assert sorted(transform.changes)==['kv_lens_arr_host','qo_indptr_host'],transform.changes
  namespace=dict(original.__globals__);exec(compile(tree,'<sol3_vectorized_flashinfer_plan>','exec'),namespace);revised=namespace[fn.name]
  # Valid nonempty integer host vectors: exact max value, no approximation.
  for vector in [torch.tensor([0,17,31,33],dtype=torch.int32),torch.tensor([3,1,7,7,2],dtype=torch.int32)]:assert max(vector).item()==torch.amax(vector).item()
  CACHE[key] = (revised, transform.changes)
 plan, changes = CACHE[key]
 wrapper.plan = types.MethodType(plan, wrapper)
 wrapper._sol3_vectorized_plan_metadata = {"changes": changes}
 return wrapper
