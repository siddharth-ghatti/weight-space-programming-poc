## reasoning summary

The pipeline checks out on the training items at 100%, so I estimate real accuracy around 97.5%. With 20 passes left, I'm looking at improvements, starting with a layer-0 bag-of-embeddings head feeding into CLS. Then I'm reconsidering a simpler route to the same mean-of-layer-0 information: a uniform-attention head that adds one shared vector to every token, kept small in output.

I'm working out the constraints for a layer-0 CLS head: I want a direction that gives CLS a value of 1 and every other token roughly 0, which means zeroing out its response to the position embeddings and the bias term.

I'm working through the constraints on the direction g: it needs to be orthogonal to the position embeddings, the type embedding, the constant vector, and the LayerNorm bias, and to the word embeddings of the other tokens. That leaves g·(w_CLS + p_0) as the only free quantity to maximize against the residual.

I'm now weighing a least-squares version of the constraints: exact orthogonality to the position embeddings, the type embedding, the constant vector and beta/gamma, with the word embeddings handled by minimizing their projections. Since there are more vocabulary words than dimensions, perfect orthogonality to all of them can't hold, so the leftover residual will show whether the CLS target is feasible. BERT's word embeddings are fairly anisotropic, and the [CLS] embedding may stand apart from the rest, which could make that possible.

Still checking the scoring math: the CLS query stays exactly zero as long as the constraint holds, and the small float errors should be negligible.

I'm settling the scoring side now: for a non-CLS query, the score against the CLS key scales with αβ/8 times (1 − r·x_j), so a large enough αβ/8 pulls attention onto CLS. That leaves the remaining question of how small r·x_j must be for the non-CLS tokens.

The loose tolerance is what makes this workable: a large enough αβ/8 means non-CLS tokens only need their projections kept modestly small, not exactly zero. Now I'm working out what the value vector should look like, with the bias chosen so that CLS contributes nothing.

I'm settling how the 64-dim value head carries information: uniform attention on the CLS query gives the mean of the layer-0 embeddings, so I'm weighing which 64 directions of that mean to keep, with LDA giving only about seven useful ones.

I'm weighing whether the information injected at layer 0 would survive through layers 1 to 10 of the CLS stream, given that residual connections and normalization each pass it along. If the injected vector is large, the CLS state becomes unusual for BERT, which could change how other tokens attend to it. The open question is how much of the gain would actually reach layer 11.

I'm weighing whether to inject a label-free representation of the layer-0 mean, like its top 64 PCA components, so the final cross-validation stays honest. The open check is whether those components still carry discriminative signal, both alone with LDA and combined with the layer-11 features.

I could use several layer-0 heads to get more dimensions, but each one I sacrifice costs more, so one head at 64 dims is the starting point. I could also inject at a later layer, though the CLS token is only cleanly identifiable at layer 0 since its embedding is constant there. For which head to give up, I'm considering the smallest output-projection norm or the one that attends mostly to itself and position, though I'd need attention data to be sure, which would cost another pass.

I'm checking whether the layer-0 head can be swapped out without hurting the other tokens: the replaced head's output is lost everywhere, and the CLS change could shift how layers 1 to 10 process the other tokens. The net effect is unknown, so I'd need about two passes to read cls11, mean11, and cls1 to see whether the injection actually takes.

Checking whether 511 position constraints still leave room for the word orthogonality, and whether capping the sequence length at around 150 tokens is enough.

I'm now checking the constraint on the other tokens: I'll set the position range to around 128 since tests stay under 64 tokens, and then normalize the direction so that its projection onto the CLS embedding comes out to exactly 1, using the value already read from the model rather than recomputing the layer norm. After that I still need to verify how large the projections are for the non-CLS tokens.

I'm checking the value projections for the non-CLS tokens, computing g·v_j directly from the weights, and treating the layer-norm scale as roughly the same as for CLS. I'll confirm that assumption against actual token reads on a few texts afterward.

I'm setting up the constraints for g now: it has to be orthogonal to the constant vector, the LayerNorm bias, the type embedding, and the position embeddings, while keeping the word-embedding projections small. I'll solve it by minimizing the weighted word projections inside the nullspace of those constraints, then fix the scale so the CLS projection comes out to 1.

Next I'll pick the vector that minimizes the word projections within the null space of the constraints, then scale it so the CLS projection is 1. Before getting into the full solve, I want to check how much discriminative signal the layer-0 PCA features carry.



The full pipeline runs end-to-end, scoring 100% on training items and about 97.5% on held-out data. Next I'll test whether a 64-dim label-free PCA summary of mean0 adds anything on top of cls11 + mean11, before building the bag-of-embeddings head.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nm0=D['mean0'].astype(np.float64); c11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nmu=m0.mean(0); U,S,Vt=np.linalg.svd(m0-mu,full_matrices=False)\nprint('var explained 64:', (S[:64]**2).sum()/(S**2).sum())\nfor k in [32,64]:\n    Z=(m0-mu)@Vt[:k].T\n    print(k,'pca alone lda', cv(Z,y,folds,lda_fit,0.1)[0])\n    X=np.hstack([Z,c11,m11])\n    print(k,'combo ridge', [round(cv(X,y,folds,ridge_fit,l)[0],4) for l in [100,300,1000]], 'lda',[round(cv(X,y,folds,lda_fit,a)[0],4) for a in [0.1,0.3]],flush=True)\nX=np.hstack([m0,c11,m11]); print('full m0 combo ridge', [round(cv(X,y,folds,ridge_fit,l)[0],4) for l in [100,300,1000]])"
}
```

