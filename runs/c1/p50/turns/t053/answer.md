## reasoning summary

The one-layer pooled features are no help, and they may even hurt, so attention pooling over the embeddings isn't adding much. The single-layer bag-of-embeddings ceiling sits around 97.7%, which isn't enough. I need contextual depth; M0 plus M12 reaches 98.4%, so the final layer's features need deeper processing. I'm checking which shallow combinations reach the 98.3% threshold, since the scan showed M0 with M1 through M6 stuck near 97.8%.

I'm checking whether M0 with M10, M8, or M11 gets close to the 98.4% that M12 reached, along with CLS features from various layers. I also want to test whether M12's layer norm leaks information, though the leak looks small since only two scalars per sentence are involved.

I'm weighing whether to go deep: the 12-layer route is costly, so I'm considering dropping some middle layers or pruning the FFNs of the deep ones, though naive FFN pruning has already damaged the deep features. I'll test M0 with M7 through M12 and some three-layer combinations to find the shallowest depth that still clears 98.3%.

I'm weighing whether dropping some layers can keep accuracy high at a smaller size. Each experiment needs a fresh forward pass over the features, so I'd like to be selective about which configurations I test. A smarter route may be to fit a new pooler and classifier on the features from the reduced model, rather than keeping the original heads.

I'm looking at whether the deep layers can be kept while dropping middle ones, with layer 11 acting as the pooling layer on top of the content layers 0 through 10. I'm weighing which subsets to test, since each configuration needs a fresh forward pass, and whether the pooling layer can also do useful work beyond pooling.

I'm considering pooling at the final layer over the kept content layers, with the CLS output as the input to a refit kernel classifier. I also wonder whether layer 11 could pool heads from different layer outputs, not just the last one.

I'm working out how the reduced model's features should be read out, since dropping layers shifts what the original carried scores mean. The plan is to read features from the modified structure and refit the classifier rather than reuse the old heads. I'm also weighing whether a redesigned carry head in a late layer could write sentence-level scores into the reserved dims, which would need a two-step fit.

I'm planning a depth experiment: keep a subset of the content layers and test accuracy on mean-pooled features. The pooling layer costs parameters on its FFN without doing much, so I'm weighing whether the last content layer could do the pooling itself by repurposing a head as a uniform pooler.

I'm noticing that only the CLS output at the last kept layer matters, so all of its heads could be repurposed for pooling. That would leave the last layer's token-level attention with no real job, which changes what the reduced depth can do.

I'm realizing the last layer only matters at the CLS position, so its token-level processing is irrelevant and it effectively acts as a pooling layer no matter which original index it came from. That means the real content is in the layers before it, with the final layer serving purely as the pooler. Its 12 heads could carry uniform means of the last content layer's tokens, feeding a kernel classifier.

I'm weighing whether the M0 bag-of-embeddings signal can survive into the reduced model, since M0 combined with the deep layer reaches 98.4% while the deep layer alone gets 97.5%. One idea is a carry head in an early layer that writes the M0-derived mean into reserved dimensions, though those dimensions get zeroed after layer 1 and skipped by later reads, so it would need more reserved room.

I'm going to fill in the accuracy curve for the middle layers, M7 through M11, and check the CLS variants. I also want to see whether the three-way combination of M0, a middle layer, and C12 does better than the pairs.



Even with leakage, the keyed pools don't beat the bag-of-embeddings baseline (97.7%), so I need deeper contextual features. I'll fill in the M0 + M_l models for layers 7 through 11 to find the shallowest depth that reaches 98.2%.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base()\ndef feat(l, kind='mean'):\n    X=B[f'{kind}_hidden_L{l}'][:,KEEP].astype(np.float64); X=X-X.mean(0); return X/np.sqrt((X**2).sum(1).mean())\ndef kcv(Z, gs=(0.5,1), lams=(0.003,0.01,0.03), k=10):\n    Z=Z/np.linalg.norm(Z,axis=1,keepdims=True); S=Z@Z.T\n    return {g:max(krr_cv(np.exp(g*(S-1)),y,lams=lams,k=k,seeds=(0,)).values()) for g in gs}\nfor l in [7,8,9,10,11]:\n    print('M0+M%d'%l, kcv(np.concatenate([feat(0),feat(l)],1)), ' M%d'%l, kcv(feat(l)))\n"
}
```

