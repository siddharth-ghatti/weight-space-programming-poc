## reasoning summary

I'm noting that position 1 is less similar to the other positions (cosine around 0.87 to 0.92), while positions 5 through 60 look more alike. Subtracting a per-position mean would help, but the detector is a fixed linear function and can't depend on position, so I'm considering choosing the direction to be orthogonal to position-variation directions by folding that variation into the within-class scatter.

I'm laying out the detector plan: sampling a set of positions, computing per-token hidden states at each, and building a covariance from both token-level and position-level variation. I'd then form each direction from that covariance, normalize it, and check self-responses against cross-responses for similar tokens like "cover"/"covers".

I'm settling on a least-squares fit for the output weights, since the FFN acts like a random-feature model, so the readout has a closed-form solution. Next I'll compute the covariance and detectors and check their selectivity against the full vocabulary, processed in chunks to keep memory manageable, and I'll put the helper functions in a shared `lib.py`.



Cosine similarity to position 1 is about 0.87–0.92, while distinct tokens stay nearly orthogonal except close pairs like cover/covers (~0.6). I'm building whitened detectors that include position-induced variance next, starting with shared helpers in lib.py.



## tool_use: run_python

```json
{
 "code": "lib = r'''\nimport numpy as np, scipy.sparse as sp, json\nfrom pathlib import Path\ndef ln(x, gg, bb, eps=1e-12):\n    m = x.mean(-1,keepdims=True); v = ((x-m)**2).mean(-1,keepdims=True)\n    return (x-m)/np.sqrt(v+eps)*gg+bb\nDF = np.array([145,175,529,121,216,654,159,541])\nclass Emb:\n    def __init__(self, base):\n        self.E = base('bert.embeddings.word_embeddings.weight'); self.P = base('bert.embeddings.position_embeddings.weight')\n        self.T = base('bert.embeddings.token_type_embeddings.weight')\n        self.g = base('bert.embeddings.LayerNorm.weight'); self.b = base('bert.embeddings.LayerNorm.bias')\n        self.gam = np.ones(768,np.float32); self.gam[DF]=0\n    def h0(self, tokids, pos):\n        \"\"\"layer-0 FFN input given attention no-op and LN_a(gamma=1 except DF=0, beta=0)\"\"\"\n        x0 = ln(self.E[tokids] + self.P[pos] + self.T[0], self.g, self.b)\n        return ln(x0, self.gam, np.zeros(768,np.float32)).astype(np.float32)\ndef gelu(x):\n    from scipy.special import erf\n    return 0.5*x*(1+erf(x/np.sqrt(2)))\ndef load_data(practice, tokenize, LABELS):\n    P = practice(); lab2i = {l:i for i,l in enumerate(LABELS)}\n    y = np.array([lab2i[p['label']] for p in P])\n    ids = [t['ids'] for t in tokenize([p['text'] for p in P])]\n    rows=[];cols=[]\n    for i,t in enumerate(ids):\n        for j in t: rows.append(i);cols.append(j)\n    X = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(P),30522))\n    return P,y,ids,X\n'''\nopen('lib.py','w').write(lib)\nimport sys; sys.path.insert(0,'.')\nfrom lib import *\nem = Emb(base)\nprint(em.h0(np.array([101,102,0]), 0).shape)"
}
```

