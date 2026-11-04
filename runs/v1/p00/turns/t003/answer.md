## reasoning summary

I'm moving on to the embedding statistics now, looking at the γ_e and β_e parameters, the SVD of the position embeddings, and their norms.



Next I'll examine the embedding statistics the design relies on, covering the LayerNorm ranges, the word and position embedding norms, and the position embedding spectrum to decide how much of that subspace to project out.



## tool_use: run_python

```json
{
 "code": "g = base('bert.embeddings.LayerNorm.weight'); b = base('bert.embeddings.LayerNorm.bias')\nprint('gamma', g.min(), g.max(), np.percentile(g,[1,50,99]), 'beta', b.min(), b.max())\nprint('small gamma count', (np.abs(g)<0.1).sum())\nWd = base('bert.embeddings.word_embeddings.weight'); Pe = base('bert.embeddings.position_embeddings.weight'); T = base('bert.embeddings.token_type_embeddings.weight')\ndef cn(v): return np.linalg.norm(v - v.mean(-1, keepdims=True), axis=-1)\nprint('word centered norm pct', np.percentile(cn(Wd),[1,10,50,90,99]))\nprint('pos centered norm 0..64', cn(Pe[:64]).round(2))\nprint('type0 norm', cn(T[0]))\n# SVD of pos 0..127 (with ones removed)\ndef rm1(M): return M - M.mean(-1, keepdims=True)\nfor K in [64,128]:\n    U,S,Vt = np.linalg.svd(rm1(Pe[:K]+T[0]), full_matrices=False)\n    e = np.cumsum(S**2)/np.sum(S**2)\n    print(K, [ (k, round(float(e[k-1]),4)) for k in [5,10,20,30,40,60]])\n"
}
```

