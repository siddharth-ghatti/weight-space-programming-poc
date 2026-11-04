## reasoning summary

I'm working out which dimensions to look at: the token states at layers T0, T6, and T11 across the 3970 tokens from 128 texts, along with per-dimension variance and mean absolute values. I also want to read the weight column norms for layers 0 through 8, since those tell me how much each dimension feeds into the attention and MLP projections.

I'm estimating each dimension's ablation cost from its variance times the squared norms of its columns in the attention and MLP weights, using T0 variance for early layers and T6 for later ones. Then I'll check the value path's effect and the LayerNorm gains. I'll pick eight dimensions with the best score and skip the known outliers.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nT=np.load('tok128.npz'); T0=T['T0'].astype(np.float64); T6=T['T6'].astype(np.float64); T11=T['T11'].astype(np.float64)\nv0=T0.var(0); v6=T6.var(0); v11=T11.var(0)\nscore=np.zeros(768); parts=[]\nfor l in range(9):\n    L=f'bert.encoder.layer.{l}.'\n    var=v0 if l<3 else v6\n    q=np.linalg.norm(base(L+'attention.self.query.weight'),axis=0)**2\n    k=np.linalg.norm(base(L+'attention.self.key.weight'),axis=0)**2\n    vv=np.linalg.norm(base(L+'attention.output.dense.weight')@base(L+'attention.self.value.weight'),axis=0)**2\n    w1=np.linalg.norm(base(L+'intermediate.dense.weight'),axis=0)**2\n    score+=var*(q+k+vv+w1)\nF=np.load('feats.npz'); C9=F['C9']; \norder=np.argsort(score)\nprint('best dims', order[:20])\nprint('score', np.round(score[order[:20]],3), 'median', np.median(score))\nprint('v0', np.round(v0[order[:12]],4), 'median v0', np.median(v0))\nprint('v6', np.round(v6[order[:12]],4), 'median v6', np.median(v6))\nprint('C9 var', np.round(C9.var(0)[order[:12]],4), 'mean |T6|', np.round(np.abs(T6.mean(0))[order[:12]],3))\nch=np.argsort(C9.var(0))[:8]; print('readout ch dims (low var C9)', ch)\nnp.save('stash_score.npy',score)"
}
```

