## reasoning summary

Checking the layer-0 CLS accumulator against a CLS that skips layers 1 through 10, since the first option pulls in the bag-of-words mean and the second has to avoid the distortion from transiting many layers.

Still looking at freezing the CLS in layers 1 through 10, which isn't possible per token in post-LN BERT. I'm considering instead reserving a subspace across all tokens that layers 1 to 10 ignore, by projecting their weights away from it.

Now I'm working out how to reserve a set of coordinate dimensions that layers 1 through 10 neither read nor write, with the layer norm gains and biases adjusted so those dimensions pass through unchanged.

Now I'm looking at how the scalar shifts and scalings from each layer norm act on the reserved dimensions. The per-token mean subtraction only moves all reserved coordinates by the same amount, so encoding the signal as differences across those coordinates would make it invisible. The per-token rescaling by the standard deviation compounds across layers, which is the harder part to control.

I'm checking whether the per-token rescaling by each layer norm's standard deviation is the real obstacle, since the division compounds across layers and could shrink the reserved signal to almost nothing. If the pre-LN std is typically well above 1, that product could be tiny, so I'd set the reserved gains to a typical constant to keep magnitudes roughly stable. The per-token variation would remain, but that's tolerable.

I'm checking which dimensions to reserve as low-importance, since BERT's outlier dimensions like 308 and 381 carry large values that feed the layer norms, and removing eight of 768 dimensions may or may not hurt much. The layer-0 write would then put the per-token bag-of-words signal into those reserved dimensions, with that layer's other writes to them zeroed out.

I'm considering whether the embedding values already sitting in the reserved dimensions could carry the signal, since they'd persist through every layer. The catch is that the mean over tokens of those coordinates only matches the discriminative projection I want if I can pick the right coordinates, and an arbitrary choice won't do that.

Checking that GELU(u) minus GELU(-u) equals u, which would let a pair of FFN neurons give an exact linear write into the reserved dimensions at layer 0.

Trying to see whether the layer-0 attention can write each token's projection into itself, using position embeddings so that a query at position i matches only the key at position i.

Now I'm weighing whether to pick the reserved coordinates greedily, scoring each dimension by how well its value in the mean embedding separates the classes. Layers 0 and 1 through 10 would leave those coordinates untouched, so the pooled layer-11 state over those dimensions would approximate a weighted average of the layer-0 embeddings.

Now I'm checking how much the per-token scaling from each layer norm shrinks the reserved dimensions, since with about 22 normalizations the signal could fall to nearly nothing unless the typical standard deviation is known.

The per-token scaling from each layer norm is what I'm worried about now: about 22 normalizations could shrink the reserved signal toward nothing, or leave it as a noisy, randomly weighted average. Setting the reserved gains to a typical standard deviation would keep the magnitude roughly stable, though the per-token variation would remain.

I'm weighing the CLS-accumulator route against the residual-stream route. The accumulator would sacrifice one layer-0 head and pass the bag-of-words projection through the CLS token, so the open question is how much that head's removal disturbs the other tokens.

Checking whether layer 0's attention heads can keep their original behavior for the non-CLS tokens while the CLS query is made uniform, so the sacrificed head's effect is limited to the CLS position.

Now I'm setting the query for the CLS position to zero, so its attention in that head comes out uniform, while the other tokens' queries barely change.

Now I'm working out how to give CLS the full 768-dimensional projection of mean0 by zeroing its query in all twelve layer-0 heads, so the uniform attention carries the value projection straight into the CLS position.

I'm checking whether the CLS token can be made to carry the full mean-of-embeddings projection after layer 0. By zeroing CLS's query in every head, its attention becomes uniform, so the value maps push the linear mean0 content straight into the CLS position. Other tokens' queries barely change, so the rest of layer 0 stays close to the original, though later layers will still shift how CLS's content gets used.

I'm weighing whether the rank-1 correction to the layer-0 query weights can leave CLS's query at zero in every head while barely disturbing the other tokens' queries. If the correction direction is nearly orthogonal to the other tokens' embeddings, their attention would stay close to the original, and only the CLS position would shift toward uniform attention.

I'm checking whether the layer-0 attention for the CLS token can be made uniform by zeroing its query, so that its value projections carry the full mean-of-embeddings content forward. The other tokens' queries barely shift, since the correction is nearly orthogonal to their embeddings. What remains unclear is whether that information stays decodable once the later layers process it.

Now I'm asking whether the same trick can work at deeper layers, where CLS's input depends on the sentence and an exact zero query is out of reach. A direction that is roughly 1 on the CLS hidden state and roughly 0 on the other tokens might still push CLS's query close to zero, giving near-uniform attention.

Next I'm weighing whether the CLS token could carry the mean-of-embeddings content in every layer, since uniform attention would let it accumulate averages across depth. That would need a separate zeroing direction for each layer, so I'll start with layer 0 alone and see how much it gains.

I'm working out how to pick the reserved coordinates so that the layer-0 embedding gives an exact projection, starting from the CLS hidden state and the per-token layer-norm inputs. The aim is a direction r that gives 1 on the CLS embedding and roughly 0 on the other tokens, so only the CLS position is affected.

Now I'm working out whether the reserved coordinates can be chosen so that the position and time embeddings drop out, leaving only the word contributions to compare across the vocabulary.

I'm now looking for a direction g that is close to zero on most vocabulary words, so that the perturbation to other tokens' queries stays small. I'll check whether the ratio of its effect on ordinary tokens to its effect on CLS comes out near 0.05.

Next I'm picking the reserved direction so it stays nearly orthogonal to the position, time, and typical word embeddings, with its effect on the CLS embedding normalized to 1. Then I'll check how much it perturbs the other tokens relative to CLS.



Lower layers help noticeably, but cls12 doesn't. Next I'll compute a rank-1 edit to W_q from the frozen embeddings so that [CLS] gets layer-0 information.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nWemb=base('bert.embeddings.word_embeddings.weight').astype(np.float64)\nPemb=base('bert.embeddings.position_embeddings.weight').astype(np.float64)\nTemb=base('bert.embeddings.token_type_embeddings.weight').astype(np.float64)\ngam=base('bert.embeddings.LayerNorm.weight').astype(np.float64); bet=base('bert.embeddings.LayerNorm.bias').astype(np.float64)\nV=vocab(); cls_id=V['[CLS]']; sep_id=V['[SEP]']; print(cls_id,sep_id)\nD=np.load('base_feats.npz'); eC=D['cls0'][0].astype(np.float64); print('cls0 const?', np.abs(D['cls0']-D['cls0'][0]).max())\n# token frequency in practice\ntoks=tokenize([p['text'] for p in practice()])\nfreq=np.zeros(len(Wemb))\nfor t in toks:\n    for i in t['ids']: freq[i]+=1\nprint('distinct tokens used', (freq>0).sum())\n# constraints on g: g.1=0, g.(bet/gam)=0, g.t0=0, g.p_k=0 k=1..K\nK=160\nC=np.vstack([np.ones(768), bet/gam, Temb[0], Pemb[1:K+1]])\n# null space of C\nU,S,Vt=np.linalg.svd(C); rank=(S>1e-8*S[0]).sum(); N=Vt[rank:].T  # 768 x (768-rank)\nprint('constraints rank',rank,'free dims',N.shape[1])\nwts=freq.copy(); wts[cls_id]=0; wts=wts/wts.sum()+ 0.2/len(Wemb)  # usage weight + uniform floor\nwts[cls_id]=0\nSw=(Wemb*wts[:,None]).T@Wemb\nu=Wemb[cls_id]+Pemb[0]   # g.(v_CLS) direction (t0 term zero)\nfor reg in [1e-6,1e-4,1e-3]:\n    Q=N.T@Sw@N + reg*np.eye(N.shape[1])\n    z=np.linalg.solve(Q, N.T@u); g=N@z\n    r=g/gam; r=r/(r@eC)   # normalize r.e_CLS=1\n    g=r*gam\n    # evaluate r.x for all words at a few positions\n    def rx(ids,pos):\n        v=Wemb[ids]+Pemb[pos]+Temb[0]\n        m=v.mean(1,keepdims=True); s=np.sqrt(((v-m)**2).mean(1,keepdims=True)+1e-12)\n        x=gam*(v-m)/s+bet\n        return x@r\n    used=np.where(freq>0)[0]; used=used[used!=cls_id]\n    vals=np.concatenate([rx(used,p) for p in [1,5,20,60]])\n    allv=rx(np.arange(len(Wemb)),3)\n    print(reg,'|r|',np.linalg.norm(r).round(3),'used tokens r.x: mean abs',np.abs(vals).mean().round(4),'max',np.abs(vals).max().round(4),'99pct',np.percentile(np.abs(vals),99).round(4),'| all vocab max',np.abs(allv).max().round(3), 'check eC',r@eC)\n"
}
```

