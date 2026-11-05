## reasoning summary

The layer 1 drift is much larger than expected, with the CLS R² coming out near -50. I suspect the layer 0 compensation went wrong, since its output norm (4.27) differs from the compensation norm (2.15) by a wide margin. The negative R² could also be partly explained by the CLS token having almost no variance across inputs at early layers, because it sits at position 0.

Relative error per layer might be a better measure than R² here, since the CLS and mean-hidden variances are so small across inputs. Pruning the FFN to 1536 with this proxy importance does seem to hurt agreement (91.7%, versus 95.5% for truncating to 5 layers and refitting). So the bias compensation may be the problem if the proxy is poor, and I want to check where the error actually enters layer by layer before changing the importance estimate.

I'm working out a way to recover each token's pre-FFN activation from the edited model's output, by solving the LayerNorm constraints for the scale and shift. The catch is that it needs one extra forward pass per layer, so I'm weighing whether that cost is worth it.

I'm checking whether multiple layers could be probed in one forward pass, but the edited model is a single chain, so a probed layer's successors would see altered input and that doesn't work. Reading the attention probabilities and rebuilding the attention sublayer by hand would mean reimplementing it in numpy, which I'd rather avoid, so I'm weighing six separate passes for an exact FFN input.

I'm weighing whether the FFN pruning is the real problem or whether the spread from the 18% unexplained variance compounds over six layers no matter how the neurons are chosen. Layer dropping looks like the more promising direction to test next.

I'm weighing whether the two pruning paths are actually comparable, since both land around 14M saved parameters with roughly 92–93% agreement. The layer-drop route gives 95.5% at 7.09M, which makes the trade-off look less clear-cut than it first seemed. Given that, I doubt either path reaches the test accuracy I need, since the edited model sits about 5–7 points below the base on practice data.

Comparing the two pruning paths, both land around 92–93% agreement on practice data, which falls well short of the roughly 97–98% needed to avoid losing ground on test. That points to modest compression only, unless the approach gets smarter. One option is to improve FFN pruning with exact input statistics and least-squares compensation through the kept units.

I'm considering a linear approximation for each pruned unit's GELU, so the summed linear parts form a single matrix M acting on the FFN input. Representing M a with the kept units is not straightforward, though. Folding it into the residual as (I + M) before LayerNorm might be a cleaner way to absorb it.

I'm considering whether a few kept FFN units could be set to operate in their linear regime by giving them a large positive bias, so they reproduce a low-rank linear map of the input. Subtracting the offset from the output bias would keep the net effect the same, which might let the pruned units' linear parts be carried over.

I'm realizing the pruned units' linear parts could be carried over exactly: a few kept units pushed into their linear regime with a large positive bias can reproduce a low-rank linear map of the input, with the offset subtracted from the output bias. The cost is one unit per singular direction, so it only helps if that linear map is close to low rank. I'm also wondering whether the same trick could extend to a linear map in the residual path when a layer is dropped.

I'm now considering whether the layer-drop route could carry a linear correction too. The FFN of the previous kept layer feeds into the layer that follows the dropped one, but LayerNorm sits between them, so a general linear map on the hidden state can't be applied there directly. Only an approximate fold, absorbing the map into that FFN's output, looks feasible.

I'm checking whether a linear correction can be folded into the LayerNorm of the previous layer's FFN output, treating the normalization as roughly affine when its scale is nearly constant across tokens so the dropped layer's map can be absorbed.

I'm turning to a different question here: is the FFN itself the bottleneck, or does the 18% unexplained variance compound across six layers no matter which units survive? I'd rather not compare configurations on agreement alone, since the 14M-saving paths both land near 92–93% and that falls short of what the test accuracy needs.

I'm working through how the linear parts of the pruned units might be compensated. Each unit's output splits into a linear term in its input, a constant, and a nonlinear residual. Bias compensation covers the constant, and a linear correction could cover the linear term. The leftover error comes from the residual, which is small for units that stay in one regime but large for units that straddle zero.

I'm now quantifying how much each pruned unit's nonlinear part matters, using the Gaussian approximation where the linear slope equals the expected derivative of GELU. For each layer I want to sort units by that residual importance and compare retained fidelity between keeping the top units with linear compensation and plain pruning.

I'm moving on to the drift check and planning to switch to a relative error metric, since R² is misleading for the CLS token at early layers where its variance across inputs is tiny. Using the norm of the difference relative to the reference norm should give a fairer read on how far the layers have drifted.



Experiment A matches base on 91.7% of practice items, and a refit head only reaches 93%, too lossy for a 1-point margin. I'll check drift with relative error instead, then test whether the linear parts of pruned units can be replaced by a few exact linear units.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK)); from ffnlib import *\nfor e in range(1,7):\n    out=[]\n    for kind in ['cls','mean']:\n        A=np.load(f'pass_3/{kind}_hidden_L{e}.npy'); B=np.load(f'pass_1/{kind}_hidden_L{e}.npy')\n        out.append(np.linalg.norm(A-B,axis=1).mean()/np.linalg.norm(B,axis=1).mean())\n        out.append(np.linalg.norm(B-B.mean(0),axis=1).mean()/np.linalg.norm(B,axis=1).mean())\n    print(f'L{e}: cls rel err {out[0]:.3f} (cross-sample spread {out[1]:.3f}); mean rel err {out[2]:.3f} (spread {out[3]:.3f})')\n# nonlinear importance\nmus=np.load('tok_mu.npy'); covs=np.load('tok_cov.npy')\nfrom scipy.stats import norm as N\ndef lin_coef(m,s):\n    z=m[:,None]+s[:,None]*_gh_x[None,:]\n    dg=N.cdf(z)+z*N.pdf(z)\n    return (dg*_gh_w).sum(1)\nfor l in range(6):\n    W1=base(f'roberta.encoder.layer.{l}.intermediate.dense.weight').astype(np.float64)\n    b1=base(f'roberta.encoder.layer.{l}.intermediate.dense.bias').astype(np.float64)\n    U=base(f'roberta.encoder.layer.{l}.output.dense.weight').astype(np.float64); un=(U**2).sum(0)\n    tot=[];nl=[]\n    for src in [l,l+1]:\n        m,s=unit_stats(W1,b1,mus[src],covs[src]); Eg,Vg=gauss_gelu_moments(m,s); al=lin_coef(m,s)\n        tot.append(Vg*un); nl.append(np.maximum(Vg-al**2*s**2,0)*un)\n    T=np.sqrt(tot[0]*tot[1]); NL=np.sqrt(nl[0]*nl[1])\n    print(f'L{l}: nonlinear share of total var {NL.sum()/T.sum():.3f}; top-1024 by NL keeps {np.sort(NL)[::-1][:1024].sum()/NL.sum():.3f} of NL, top-1536 {np.sort(NL)[::-1][:1536].sum()/NL.sum():.3f}')"
}
```

