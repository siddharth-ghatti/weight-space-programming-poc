## reasoning summary

Pruning to 1024 neurons by std wrecks the deep features, with about 94% lost by layer 10, so the FFNs clearly carry a lot of the signal. The deviation is already 0.49 at layer 1, which makes me suspect the selection criterion itself, since I'm not sure what the std was even computed over. Layer dropping with the full FFN might be a cleaner alternative.

I'm weighing whether layer dropping can get the size down enough, since keeping only a few layers barely helps. Skipping middle layers would perturb the residual stream, and dropping the top layers only yields about 6.5% savings, so the FFN width itself has to shrink.

I'm thinking about which units to keep in the FFN, since it's a slicing problem rather than a low-rank one. The current std-based score probably ignores how much each unit actually contributes through its output column, so I'm weighing a score based on the expected magnitude of each unit's contribution, or a closed-form least-squares refit of the output weights on the kept units, which would need the real FFN input activations.

I'm checking whether the FFN input can be read out without reimplementing the forward pass, since the regression needs real activations. Each read pass gives roughly 1900 tokens, but exposing one layer's FFN input would mean modifying that layer, which shifts everything downstream, and I can't get all layers at once.

I'm weighing a trick to get each layer's FFN input: zero that layer's FFN output and set the LayerNorm parameters to identity, so the layer's output exposes the normalized attention residual. Running one pass per layer would give about 1900 tokens each, and I could compute the unit activations from there, though it edges toward reimplementing part of the forward pass.

I'm wary of computing FFN activations from the returned inputs, since that starts to look like reimplementing part of the forward pass. Without them, weight-only importance scores like ||W2[:,i]|| * ||W1[i]|| usually perform poorly, and I don't know how the existing ffn_scores were derived, so I'm not inclined to rely on them.

I'm comparing fewer layers with full-width FFNs against all layers with narrower FFNs. Keeping 8 layers at full width would cut about 26%, and narrowing every layer to 1536 units would land at a similar size.

I'm weighing whether to test skipping individual layers, but that's eleven separate runs. Known BERT behavior suggests the early and late layers matter more than the middle ones, so I'm wondering if pruning the FFN could still work if the mean-compensation step is what's failing.

I'm wondering whether the pruning failure at layer 1 (deviation 0.49) comes from layer 0's dense FFN, since keeping only 1024 units there may cut too much. Layer dropping hasn't helped much either, so I'm now focusing on whether the deep features only need to preserve class-discriminative information, since the classifier gets refit anyway.

I'm weighing whether to stay with the forward-pass tool alone or compute FFN activations from the returned inputs. Doing the GELU step myself looks like reimplementing part of the forward pass, which the rules forbid, so I'm inclined to avoid it even though it would make a bigger-width or better-scored pruning feasible.

I'm circling back to whether fitting classifier weights on tanh features from read activations counts as the same kind of closed-form step as computing FFN unit activations for pruning. If it's just ridge on returned features, the FFN case looks equally acceptable, and the real line is simulating the full model rather than fitting weights on activations.

Now I'm checking whether the FFN inputs can be pulled from a modified-model read, and how many passes that would take. Each read covers only 64 sentences, about 1900 tokens, which is enough for selection statistics but tight for a least-squares refit on a 768×m matrix. One pass per layer seems necessary unless the structure can expose several layers at once.

I'm considering whether the FFN inputs at each layer could come from a modified read: keep only the layers up to the one I want, switch off that layer's FFN output and set its LayerNorms to identity, so the output exposes the normalized attention residual. That would take one pass per layer, eleven in all, which is expensive but still feasible with 43 passes left. I'm also wondering whether mean-based statistics could avoid that, since they only need the pooled hidden states.

I'm weighing whether the FFN input h_attn can stand in for the layer input h_l when choosing units, though the previous run may have done exactly that and failed. I'm also stepping back to ask whether the realistic floor is around 60–80M parameters, and whether non-standard ways of getting deep-like features might cost less.

I'm considering whether a shallow model with designed attention could produce bigram-like features, since mean pooling is linear and only the FFN gives a token-level nonlinearity. Units like GELU(a·x_t + b·x_{t-1} + c) could detect specific pairs such as "not covered", but choosing those pairs would take supervised feature engineering in weight space.

I'm weighing whether shallow models with random bigram-like features could approach deep BERT accuracy at roughly 32M parameters, but accuracy is uncertain. Since a shallow BERT with two layers only reaches about 97%, I suspect bigram features would land in a similar range, around 97.8%, and deep contextual semantics may matter more.

Next I'm weighing whether the largest savings come from trimming the FFNs inside the existing depth. The pooling layer's FFN could be repurposed as kernel units at the CLS position, while pruning and narrowing the other layers would need to be checked experimentally.

I'm weighing whether I can capture every layer's FFN inputs in a single read by duplicating structure, but the layers must appear in order, so that won't work. Reading attention tensors per head is another route I'm considering.

I'm settling on the LN-trick read as the cleanest route, since the forward pass hands back the FFN input directly. Computing the unit activations from those inputs is one matmul plus GELU, and the actual evaluation still runs through forward passes, so I'm comfortable with it.

I'm considering whether the FFN inputs could be captured more cheaply than eleven separate passes, but the LN-based read is still the cleanest option. Before committing those passes, I want a quick feasibility check on one layer, comparing activation-based unit selection with a least-squares refit against a naive baseline.

I'm recalling the structured-pruning work on BERT, which prunes FFN neurons with a mask search and least-squares reconstruction and keeps accuracy within about 1% at roughly half the FFN width. If that holds here, around 1536 units per layer looks plausible, which would put the model near 76M parameters, and dropping a couple of layers could bring it closer to 67M.

I'm weighing a middle path between the 27M single-layer model and a 70M pruned one: keeping three to five layers with their full FFNs. Early-layer features alone score lower, but pairing layer 0 with layers 8 or 9 already reaches roughly 98.3–98.5%, with the standard error around 0.35%.

I'm now testing whether layer 9's features hold up, and whether pooling several layers' means together, such as layers 0, 4 and 8, could beat the original editor's combination of layers 2, 6, 10 and CLS.

I'm weighing whether the FFN outputs need to be computed in numpy as the reconstruction target, since that would amount to reimplementing the sublayer rather than reading activations from forward passes.

I'm checking whether the FFN outputs can be recovered from forward passes too, by switching off the LayerNorm scaling and bias so the normalized output reflects the FFN applied to the residual. The idea is two reads per layer, one with the FFN off and one with it on, and then recovering the FFN output from the difference up to a per-token scale and shift.

I'm checking whether the FFN output can be recovered from two forward reads, one with the sublayer off and one on, but the normalization mixes in unknown per-token shift and scale, so it isn't cleanly identifiable. Scaling the FFN down would only linearize things and get messy. I'll accept reading the FFN inputs from the forward pass and computing the unit activations from them, since that's ordinary activation analysis and not a full end-to-end reimplementation.

I'm checking whether the unit activations can come straight out of a forward pass, by reshaping one layer's output to expose a chosen subset of the GELU units. The FFN is 3072 wide and the hidden state only 768, so a direct read can't cover all of them.

I'm considering whether the residual stream could be made to show the GELU units directly: if the output projection maps a chosen subset of units into the hidden dimensions with a large scale, the layer norm would roughly return those activations, normalized, and the residual would matter little.

I'm going over whether the task designers would expect unit selection from activations computed from the hidden states. Pruning based on activation statistics looks like the intended route, and I'm treating the forward pass as off-limits for simulation but fine for reading activations.

I'm weighing whether to estimate the cost of pruning with a concrete plan before committing, and I'll keep the accuracy checks on forward passes even if the FFN activations are computed from the inputs they return.

Now I'm costing out the pruned options: with 1536 units per layer the model lands near 76M parameters, at 1024 units about 68M, and at 768 about 64M, while keeping all layers instead would give roughly 81M. I'm also tallying the budget, about 16 of the 43 passes, with the main worry being that pruning one layer's error carries forward into the layers after it.

I'm thinking about the ordering: prune one layer at a time, with each layer's FFN input read from the model that already has the earlier layers pruned. Each reconstruction then targets the full FFN applied to that same input, which keeps the local fit simple, even though it means errors from earlier layers aren't corrected.

Now I'm thinking about the size of the calibration set: only about 1900 tokens per layer come from 64 sentences, so a least-squares refit of W2 over 1024 kept units would be underdetermined and need ridge regularization. I could pull more sentences per pass, though the activation-variance selection should still be workable, and I want a quick sanity check before committing to the full pipeline.

I'm going to check the pruning idea against a quick experiment before spending the full budget: first, whether dropping a few middle layers, such as 3, 5 and 7, keeps the classifier accurate.

I'm checking how the pool layer's own computation fits in: the mean-pooling and CLS step are currently a separate layer, and I'm wondering whether the last content layer could absorb that role so the budget isn't spent on an extra full FFN.

I'm noting that the last layer's attention and FFN only matter for the CLS output, so I can treat it as the pooling layer and rewrite it to average the layer-9 outputs directly.

I'm working out how to reach the pooling step with fewer full FFN blocks, since the last content layer's output at non-CLS positions is never used. I'm also keeping the earlier-layer carry in mind, since adding the first layer's mean to the final features helped in the measurements.

I'm weighing whether dropping a few middle layers, like 3, 5 and 7, keeps the classifier accurate, since the ridge-scored carry heads leak across folds. The simplest honest route is to ignore the reserved dimensions in cross-validation and let the classifier skip them, and then see how accuracy holds up.

I'm going to run a cheap experiment first: drop a few middle layers, keep the rest full, and read the mean hidden state at every kept layer. That single pass also tells me how a pooling step placed after each layer would fare, so one read covers a whole family of truncated structures before I commit to the expensive FFN pruning.

I'm weighing whether dropping middle layers, such as 5, 7 and 9, keeps the representation usable without re-fine-tuning, since nearby layers tend to be redundant. That makes a single pass worth running first to see how a pooling step after each kept layer would fare.

I'm going to measure how much each layer changes its input, using the token hidden states from the 64 sentences, by computing the relative change between consecutive layers. The layers that barely move the representation would be the first candidates to drop, and I can check this on the mean states too.



Pruning with std-based scoring hurt deep layers (layer 10 fell from 97.6% to 94.2%), so the FFNs matter. I'll next measure how much each base layer changes token states to find the safest ones to drop.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\nT=np.load('tok64.npz')\nH=[T[f'L{l}'][:,KEEP].astype(np.float64) for l in range(13)]\nfor l in range(12):\n    a,b=H[l],H[l+1]\n    cos=np.mean(np.sum(a*b,1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1))\n    rel=np.mean(np.linalg.norm(b-a,axis=1)/np.linalg.norm(a,axis=1))\n    print('layer %d: cos(in,out)=%.3f rel change=%.3f'%(l,cos,rel))\n# skip-two similarity\nfor l in range(11):\n    a,b=H[l],H[l+2]\n    print('skip %d..%d cos=%.3f'%(l,l+1,np.mean(np.sum(a*b,1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1))))\n"
}
```

