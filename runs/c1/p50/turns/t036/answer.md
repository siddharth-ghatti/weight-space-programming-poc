## reasoning summary

I'm checking whether position is recoverable from the hidden state, and it is, with R² around 0.999. The CLS token sits at a position near 22, so the early pools already give it little weight, and SEP at about 24 behaves similarly. The late pool is different: at those positions CLS and SEP would get picked up for short sentences, so I need the c factor to suppress them, which works out to about 13.

I'm setting up the late pool so it favors the last tokens before SEP, and I'm defining the head directions for the class heads, the two early heads, the uniform head, and the late head. Next I'm building the value directions from normalized ridge columns and token PCA, then setting the output projection to identity with a bias that cancels the residual, and zeroing the feed-forward output and keeping layer norms neutral.

I'm overwriting layer 11's attention query, key, and value weights and biases, along with the attention output projection, the attention layer norm, and the output projection and output layer norm. I'll leave the intermediate dense layer alone since it doesn't matter here. Then I'll run the structure with layers set to [1] and read the final CLS hidden state for all 1496 sentences.



Position is linear in the embeddings (R² = 0.999), and CLS and SEP sit near positions 22–24, so I'm subtracting a CLS-direction term for the end-weighted pool while writing E1's edits into layer 11.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\npd=np.load('pool_design.npz'); W=pd['W']; r=pd['r']; c=pd['c']; xcls=pd['xcls']; pca=pd['pca']\nWn=W/np.linalg.norm(W,axis=0)\nVdirs=np.concatenate([Wn.T, pca[:56]],0)   # 64x768\nQ,_=np.linalg.qr(Vdirs.T); Vdirs=Q.T            # orthonormal rows\ndirs=[2.5*W[:,k] for k in range(8)]+[-r/1.5, -r/4.0, np.zeros(768), r/3.0-3*c]\nL='bert.encoder.layer.11.'\nWq=np.zeros((768,768),np.float32); bq=np.zeros(768,np.float32)\nWk=np.zeros((768,768),np.float32); bk=np.zeros(768,np.float32)\nWv=np.zeros((768,768),np.float32); bv=np.zeros(768,np.float32)\nfor h in range(12):\n    bq[h*64]=8.0; Wk[h*64]=dirs[h]; Wv[h*64:(h+1)*64]=Vdirs\nsave_edit(L+'attention.self.query.weight',Wq); save_edit(L+'attention.self.query.bias',bq)\nsave_edit(L+'attention.self.key.weight',Wk); save_edit(L+'attention.self.key.bias',bk)\nsave_edit(L+'attention.self.value.weight',Wv); save_edit(L+'attention.self.value.bias',bv)\nsave_edit(L+'attention.output.dense.weight',np.eye(768,dtype=np.float32)); save_edit(L+'attention.output.dense.bias',(-xcls).astype(np.float32))\nsave_edit(L+'attention.output.LayerNorm.weight',np.ones(768,np.float32)); save_edit(L+'attention.output.LayerNorm.bias',np.zeros(768,np.float32))\nsave_edit(L+'output.dense.weight',np.zeros((768,3072),np.float32)); save_edit(L+'output.dense.bias',np.zeros(768,np.float32))\nsave_edit(L+'output.LayerNorm.weight',np.ones(768,np.float32)); save_edit(L+'output.LayerNorm.bias',np.zeros(768,np.float32))\nprint(set_structure(layers=[11], ffn_keep=1))\nopen('notes.md','a').write(\"\"\"\n## Findings (run 1)\n- 5-fold CV underestimates; 20-fold CV of orig features linear ~98.2% ~= test 98.31%. Use 10/20-fold.\n- L0 mean (bag of embeddings) linear 96.3% (10f). Kernel K=S+GELU(S)+GELU(-S) on cosnorm feats: M0 97.7%(20f), M0M1 98.0%, orig 98.4%.\n- Naive FFN pruning (m=1024, top std units + bias comp) destroys deep features (exp A, pass_5).\n- Plan: 1-layer model (structure [11]) = attention pools over embeddings + FFN kernel machine (1 unit per anchor, 1496 anchors + 16 linear pair units) writing 8 class scores into reserved dims.\n- E1: layer 11 rewritten: heads0-7 class-evidence pools (2.5*w_c), h8/h9 early-position pools, h10 uniform, h11 late pool; V=8 class dirs+56 tokPCA; FFN off, LN identity -> final CLS = u.\n\"\"\")"
}
```

## tool_use: run_python

```json
{
 "code": "print(json.dumps([f'p{i:04d}' for i in range(1024,1496)])) if False else None\nimport json; print(len(json.load(open('pass_2/meta.json'))['ids']))"
}
```

