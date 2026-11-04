## reasoning summary

I'm checking which tokens the mean-pooled hidden state includes, and it covers CLS and SEP. Attention to CLS is heavy in layers 2 to 4 but light after that, so changing CLS at layer 0 would disturb those middle layers, while changing it at layer 4 or later would be less disruptive.

I'm weighing whether the BoW signal has to come from layer 0 or whether a later injection, say at layer 1 with layer 9, could carry it. The layer 1 and 9 pair scores well, so the shallow layers look necessary, and I'm considering whether a late change to the CLS token's state could hide extra information without disturbing layers 1 to 4.

I'm weighing whether to inject the BoW signal at layer 0 anyway and measure the damage, since CLS is a heavy attention sink in layers 2-4. My concern is that changing CLS's state would shift its key and value there, so I'm considering a small perturbation that keeps h1 close to normal and see whether a ridge readout at a later layer can still pick it up.

I'm considering whether a small perturbation to CLS at layer 0 could carry the BoW signal, since the change would roughly propagate linearly. The catch is that the Jacobian depends on context, so decoding that signal with a ridge readout may be unreliable. I'm also weighing a later injection at layer 4 instead, which scored 0.9645 against 0.9705 for layer 0 and 0.959 for layer 9 alone.

I'm considering whether the shallow-layer information could instead be spread across all tokens at layer 0 in a way that doesn't disturb the attention patterns, perhaps through heads that matter less. Injecting into CLS late, where it's rarely attended, would only recover what mean pooling already gives, so that route seems to add nothing new. I'm also wondering whether a nonlinear per-token feature built inside the attention softmax at layer a could carry more than a plain average.

I'm checking whether the CLS state at layers 1 to 4 can be made to look normal while still carrying the layer 0 bag-of-words signal, since CLS is a heavy attention sink there. I'm considering whether the CLS states at those layers are close to constant across sentences, which would give some room to hide a small perturbation.

I'm planning a fourth experimental pass to test a CLS hack, falling back if the damage is too large. I'm also wondering whether several independent edits could share that pass, and I'm considering a layer-0 change that averages attention over all tokens but leaves the value path alone, though layer normalization may blend everything together anyway. Next I want to measure the CLS state statistics across layers 1 to 5 and see how much other tokens' attention depends on the CLS key.

I'm planning a fourth pass that hacks CLS at layer 0 with uniform attention across all heads, then reading out both the CLS and mean states across layers on 1024 items. If the concatenated readout reaches at least 0.97, I'll go ahead with it, and I'm wondering how far CLS's layer-1 state will drift from normal.

I'm checking whether CLS already attends nearly uniformly at layer 1, since if it does, the layer-0 hack would change very little. I'll look at each head's attention entropy against uniform and how much attention CLS puts on itself.



Mean pooling over hidden states includes [CLS] and [SEP], and [CLS] draws heavy attention in layers 2–4 but little after layer 5, so altering [CLS] at layer 0 could disrupt those layers. Next I'll check layer 0's attention to [CLS] and how much the [CLS] states vary.



## tool_use: run_python

```json
{
 "code": "\nmask = np.load('pass_3/attention_mask.npy').astype(bool)\nA = np.load('pass_3/attention_L1.npy')\nfor h in range(12):\n    ents=[]; selfw=[]; sepw=[]; tv=[]\n    for i in range(64):\n        n = mask[i].sum(); a = A[i,h,0,:n]\n        ents.append(-(a*np.log(a+1e-12)).sum()/np.log(n)); selfw.append(a[0]); sepw.append(a[n-1])\n        tv.append(0.5*np.abs(a-1/n).sum())\n    print(h, 'normEnt %.2f self %.2f sep %.2f TVfromUniform %.2f'%(np.mean(ents),np.mean(selfw),np.mean(sepw),np.mean(tv)))\nfor l in range(0,13):\n    Tl = np.load(f'pass_3/token_hidden_L{l}.npy')\n    cls = Tl[:,0]; toks = np.concatenate([Tl[i][mask[i]][1:] for i in range(64)])\n    print(l, 'cls var %.3f  token var %.3f  cls-mean-dist %.2f norm %.2f'%(cls.var(0).sum(), toks.var(0).sum(), np.linalg.norm(cls.mean(0)-toks.mean(0)), np.linalg.norm(cls.mean(0))))\n"
}
```

