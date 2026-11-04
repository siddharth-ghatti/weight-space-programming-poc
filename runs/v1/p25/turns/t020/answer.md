## reasoning summary

I'm looking at how much attention other tokens pay to CLS: it's low in layers 4 through 11, around 2 to 5 percent, and higher in layers 1 through 3. So changing CLS's state from layer 4 on should matter fairly little to other tokens, though a few heads pay it more attention. I also need to consider that changing CLS's state changes its key, which could shift how much attention it receives in either direction.

I'm working out a design where CLS pools the layer-4 hidden states uniformly, so I can check which layer combinations keep the reconstruction accurate. The shallow features look like the most promising source, but the layers 1 through 3 attention to CLS is heavy, so I want to test whether carrying the state through those layers changes things.

I'm weighing whether CLS needs uniform attention at all, since what matters is collecting information from layer 4. Another option is to change CLS's keys instead of its queries, but I'm not sure that works. A simpler idea is to have CLS pool only the final layer's states, since other tokens barely attend to it from layer 4 on.

I'm considering whether CLS needs uniform attention at all, since what matters is that it gathers the layer-4 information. I'm also wondering if the state carried from layer 4 through layers 5 to 10 gets corrupted by CLS's own attention and FFN updates, and whether keeping those contributions small and roughly linear would let it carry the shallow features cleanly until layer 11.

I'm weighing an alternative where CLS pools the layer-11 states uniformly, and then wondering how to keep the layer-4 information in its residual stream without the later layers' updates overwriting it.

I'm working out whether CLS needs uniform attention at all: if the layer-11 output is a sum of its residual state and a pooled term, the pooled part could sit in directions the residual barely uses. I'll test that offline by projecting the pooled M11 features into the low-variance subspace of base CLS features and checking how the linear readout does.

I'm checking whether the superposition of CLS and pooled M11 features stays separable, and I'll test that offline against the concatenated baseline. The open question is whether CLS still carries the layer-4 information after layers 5 through 10, which I can only settle by running the modified model and evaluating the concatenated features.

I'm considering whether CLS needs uniform attention at all, and whether I could instead make its query uniform across all layers 4 through 10 so it keeps accumulating a running mean of the hidden states. I'd then compare that against the real CLS output, since the FFN transforms and the layer norm will change the proxy.

I'm wondering whether CLS can be forced into a constant state at layer 3, which would make its query constant at layer 4 and let a single correction work exactly. The catch is that getting CLS constant at layer 3 needs the same kind of CLS-specific processing I'm trying to avoid, so this might recurse. I'm also checking whether CLS's state at early layers is already close to constant.

I'm checking how much CLS's state varies across sentences at the early layers compared with its mean norm, since a near-constant CLS would make its query uniform for free. I also want to keep CLS on its base computation through layer 3, because other tokens attend to it heavily in layers 1 through 3. One idea is to exploit FFN gating so CLS responds to a direction that separates it from ordinary tokens.

I'm exploring whether the FFN could gate units so they fire only for CLS, since CLS separates from ordinary tokens along some direction. The idea is that a unit computed as a GELU of a shifted weighted sum would act roughly linearly on CLS and stay near zero on other tokens. Taking the difference of two such units with opposite signs would give CLS an approximately linear map.

I'm considering whether FFN units can be gated so they only act on CLS, using a direction that separates CLS from other tokens, so that a GELU pair yields an approximately linear CLS-only map. Units would have to be swapped out for these, which slightly changes how ordinary tokens get processed, so I'd rather keep the rest of the network intact if I can.

Checking how CLS distributes its attention in layers 4 through 10, since if it mostly lands on SEP that is effectively a no-op I could exploit. I'm also considering a rank-1 change to CLS's query weights, which could make its attention more uniform without touching the other tokens.

I'm considering a rank-1 change to CLS's query weights, which would make its query large along one direction. Its keys would still vary, so the uniform attention I want doesn't follow directly.

I'm considering a key-side change instead, but it would push attention toward CLS from every token, so that's a poor trade. Another route is to scale CLS's query down so its attention flattens toward uniform, though multiplicative gating isn't something a linear weight change can do. Last, I'm weighing a head-level change in layer 4 that makes every token's query uniform, which would sacrifice that head for all tokens.

I'm checking how much CLS attends to SEP and itself in layers 4 through 10, since a pattern dominated by SEP would make changing its query close to a no-op. I'm also considering whether a linear functional can keep CLS's query roughly constant across sentences, so the attention becomes broad enough to pool sentence information without exact uniformity.

I'm comparing the CLS deviation from its mean across sentences, since if that query stays small, the attention it produces will be close to uniform. Next I want the key vectors from layer 4 to measure how much those scores would actually spread across tokens.

I'm now looking at layer 6 as a stand-in for the analysis, since CLS attends little to other tokens there, and I'll measure the score spread per head from the CLS deviation and the layer-6 keys. I also want to check whether any head already has CLS attending nearly uniformly in layers 4 through 10, though any change to a head's value or output weights would still affect every token.

I'm weighing CLS-only modifications, since the FFN can gate units to CLS and pass it through a near-linear map. Then I can't pool across tokens directly, so I'm wondering whether the broad heads' outputs already in CLS's post-attention state could be transformed usefully without touching the shared attention weights.

I'm considering using FFN gating to build CLS-specific linear maps, since that's a strong tool. The cost would be giving up some FFN units. A linear map of rank r would need about 2r units, so rank 64 would take roughly 4% of the 3072 units, and pruning the least important ones should be a small perturbation.

I'm considering using the FFN to erase CLS's residual state each layer, keeping only a protected subspace, by having a CLS-gated unit subtract the components outside it.

I'm weighing whether the bottom singular directions of W_o could serve as a protected subspace S, since attention writes very little there. Then a CLS-gated FFN could strip everything outside S, though that would be a rank-(768 − |S|) map and seems too costly.

I'm weighing a way to pin CLS's state in layers 4 through 10 by having CLS-gated FFN units push it toward a fixed marker vector, so its own processing stops mattering. That would take a lot of units, and base FFN units would still act on CLS, so the idea is only partly workable.

I'm weighing whether the broadcast-head idea, sacrificing one head at some layer so every token carries eight scores in its residual, would avoid the perturbation problem I was worried about.

I'm weighing whether CLS-gated FFN units could carry a summary across layers 4 through 10, since tokens barely attend to CLS there and that avoids the all-token broadcast cost. The open question is what CLS's post-attention state can supply, since the true sentence mean isn't directly available in it.

I'm checking whether base CLS states from the early layers carry the bag-of-words signal as well as the means do, and they seem to fall short. So I'm looking at a uniform CLS attention pattern at layer 0, where the CLS input is exactly constant, so zeroing its query weights would be exact.

I'm weighing whether the layer-3 CLS-gated FFN could reset CLS's state to a constant vector, which would make the layer-4 attention pattern from CLS predictable. That would work only if the reset cancels the input state exactly and writes a fixed marker in its place.

I'm weighing a swamping approach instead: CLS-gated units would write a large constant vector so that CLS's state gets dominated by a fixed direction, keeping only a small sentence-dependent remainder. That would make the layer-4 attention from CLS nearly predictable, though the remainder would still carry some signal.

I'm working out how a constant CLS state could make the layer-4 d-trick nearly exact, by choosing a direction where token states are close to zero.

I'm working out a constant CLS state in layer 4 whose fixed direction sits where token states are small, so a query vector separating CLS from tokens can be chosen cleanly. The key remaining question is how the tokens' attention behaves once CLS holds that constant state.

I'm now checking whether the CLS state could be pinned to a constant vector through a large constant write, which would make its layer-4 attention predictable. The remaining cost is that tokens attending to CLS would see that fixed vector instead of its base state, and I'm still weighing whether the downstream effect stays small.

I'm weighing whether the shallow sentence summary can survive six more layers of CLS-only processing, since each layer adds deep content to the same state. Resetting at every layer would wipe out what was carried, so repeated resets don't help. I'm considering re-pooling the shallow information at the end instead, but the final tokens don't carry it.

I'm considering whether CLS can carry the shallow mean signal forward through layers 5 to 10 by itself, or whether its own processing wipes it out. The rest of the carried state would need to survive six layers of CLS-only updates, so I'm checking how much of the layer-5 CLS state can be linearly recovered from the layer-11 state across the base sentences.

I'm checking whether CLS's layer-11 state still carries enough of its earlier layer-4 and layer-5 content to be linearly recovered, using held-out ridge regression across sentences. If that recovery is strong, the shallow info could ride the natural residual stream without any extra reset machinery.



[CLS] attention from other tokens is high in layers 1–3 but drops to about 2–5% from layer 4 on, so it may work as a fairly private carrier there. I'll check offline whether layer-4 information survives to layer 11 and whether a shallow layer-4/5 pool still adds to M11.



## tool_use: run_python

```json
{
 "code": "F=np.load('feats.npz'); y=F['y']; folds=np.load('folds.npy')\ndef stdz(X,tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd\ndef cv(featfn, lam):\n    accs=[]\n    for f in range(5):\n        tr=folds!=f; te=~tr\n        Z=featfn(tr); mu=Z[tr].mean(0); Zc=Z-mu\n        Y=np.eye(8)[y[tr]]-1/8\n        K=Zc@Zc[tr].T; a=np.linalg.solve(K[tr]+lam*np.eye(tr.sum()),Y); pr=K[te]@a\n        accs.append((np.argmax(pr,1)==y[te]).mean())\n    return round(np.mean(accs),4)\nfor k in [4,5,6,8]:\n    tr=folds!=0; te=~tr\n    A=stdz(F['C11'],tr); B=F[f'C{k}']; B=B-B[tr].mean(0)\n    for lam in [10,100]:\n        R=np.linalg.solve(A[tr].T@A[tr]+lam*np.eye(768),A[tr].T@B[tr])\n        print('C11->C%d'%k,lam, round(1-((A[te]@R-B[te])**2).sum()/((B[te])**2).sum(),3))\nfor Ls in [['M4','M11'],['M5','M11'],['M4','M10'],['M2','M11'],['M4','M6','M11']]:\n    fn=lambda tr: np.concatenate([stdz(F[k],tr) for k in Ls],1)/np.sqrt(len(Ls))\n    print(Ls,[cv(fn,l) for l in [30,100,300]])\n# variability of CLS states across sentences\nfor k in [3,4,5]:\n    C=F[f'C{k}']; print('C%d'%k,'mean norm',np.linalg.norm(C.mean(0)).round(2),'std norm',np.sqrt(((C-C.mean(0))**2).sum(1).mean()).round(2))"
}
```

