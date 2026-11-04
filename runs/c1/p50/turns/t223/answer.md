## reasoning summary

I'm comparing Vc with wraw, where the largest difference is 1.0, and wondering whether that comes from the RES columns of Vc holding 1.0 values. Embedding output in those dimensions is nonzero, so I'm checking whether value rows 56 to 63 are reading the residual dimensions with a weight of 1.

I'm checking whether the value matrix's KEEP columns match wraw transposed exactly, and whether the RES columns and the bias for dimensions 56 to 63 need to be rebuilt from the scaled mean.

I'm checking whether bv[56:64] equals -wraw^T mu0 exactly, and whether the RES-column term is folded in somewhere. Then I'll build the new wraw from the joint fit: scaling the feature weights Wl by 1/sn and λ, and confirming whether mu0 matches the KEEP mean muK over all 1496 rows.

I'm working out the scale for a replacement carry matrix: setting the new weights to the scaled feature weights times a constant λ, with λ chosen so the total sum of squares of the carried values matches the old ones. I'm also noting that the per-class offset in the features will need handling when I compare the two.

I'm writing the new carry-matrix values from the scaled feature weights, with the scale λ set to match the old carry's total sum of squares. I'm also planning to update the residual dimensions by swapping the carried component while keeping the residual noise, and to check the centering in the uncentered case.

I'm now checking whether the residual-stream norm from the KEEP dimensions dominates the LN statistics, so that swapping the carried values in the RES columns barely changes normalization. I'll also set up the deployed pool fit using the updated features.

I'm working out how to set β so the deployed linear term matches the joint fit's average weight, and I'm weighing whether it should be normalized by the mean of k or left unscaled. Either choice gave the same simulated score, so I'm leaning toward the mean-normalized version.

I'm now working out how the attention bias for the residual dimensions should change, since the new carry shifts mu[RES], though that constant offset shouldn't affect the argmax.

I'm checking how the pool's layer-norm scale depends on c, since I measured a and k with c=20 and the values will shift at c=10.

I'm checking whether the pool's layer-norm scale, which I measured at c=20, shifts per sample when I drop to c=10, since the per-sample normalization depends on how far each embedding sits from the mean. Rather than rely on the c=20 numbers, I'll work at the hidden level: shift the RES values by the per-sample factor times the change in carry.

I'm planning to estimate the hidden-level scale at the last content layer by regressing the mean residual-stream values for the RES dimensions against the old carry, both for the tokens and for CLS. Then I'll rebuild the pool's normalized values from the updated hidden states and check the deployed prediction against them.

I'm planning the final fit: a joint fit on all 1496 samples with the current settings, then rescaling the carry weights by λ to match the old magnitude. I'll then predict the new residual-stream values for both the tokens and CLS from the old values plus the change in carry.

I'm now writing the deployed scoring step: recomputing the pool features from the predicted new hidden values, then combining them with the joint-fit coefficients and a scalar β calibrated by least squares against the joint fit's linear term.

I'm laying out the install plan: the L0 carry from the KEEP columns and bias, the pool attention with a diagonal of c=10 on KEEP and 1 on RES, and the pool FFN with anchor directions in the first 1008 rows. I'm also working out the output weights for the RES dimensions so the final residual value combines z[RES], the scaled anchor term, and the pair output.

Working out the pair output for the residual dimensions: each RES dimension gets its own pair of units, with the weights set so that the net contribution is the scaled linear term plus the bias. For the KEEP dimensions I'm checking whether the existing output bias values are still left over from the earlier edit, and whether they should be zeroed.

I'm settling on a gain of γ = 1, which will push the pooler's tanh inputs into its saturating range but should still keep the argmax intact, since the mapping is monotonic. I'm also checking that the only remaining change to the output bias is for the RES dimensions.

I'm now planning to set γ to 0.5, keeping the pooler's tanh inputs near 1.5 so the argmax stays intact, and I'll confirm the LN variance impact is negligible. Next I'll predict logits for all 1496 samples from the updated features and compare them with the edited model's actual outputs, checking that training accuracy holds near 100%.

I'm checking the carry's relation to the bias term and the residual columns against the mean of the original hidden states. I'll also run the hidden-level scale regressions, then write a shared script so the final fit and install steps can reuse the same computations.



The value rows 56–63 carry the direction on the content columns, with a fixed term cancelling the mean on RES, so changing the direction means rewriting Vc[:,KEEP] and bv[56:64]. I'm confirming those relations and measuring the hidden-level carry scale first, so I can predict the new RES values without a full pass.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,str(WORK))\nimport numpy as np\nfrom poolfit import *\ncp=np.load('carry_prov.npz'); wraw=cp['wraw'].astype(np.float64); mu0=cp['mu0'].astype(np.float64)\nP='bert.encoder.layer.0.'\nWv=W(P+'attention.self.value.weight').astype(np.float64); bv=W(P+'attention.self.value.bias').astype(np.float64)\nVc=Wv[56:64]\nprint('KEEP cols match wraw', np.abs(Vc[:,KEEP]-wraw[KEEP].T).max())\nprint('RES cols', np.round(Vc[:,RES],3))\nprint('bv vs -wraw^T mu0', np.abs(bv[56:64]-(-wraw.T@mu0)).max())\nf=np.load('feats_drop46.npz'); M,C,M0=f['M'],f['C'],f['M0']\nprint('mu0 vs M0 mean', np.abs(mu0-M0.mean(0)).max(), 'M0 RES mean', M0.mean(0)[RES].round(3))\ncold=(M0-mu0)@wraw\nfor nm,X in [('M',M[:,RES]),('C',C[:,RES])]:\n    Xc=X-X.mean(1,keepdims=True); pc=cold-cold.mean(1,keepdims=True)\n    a=(Xc*pc).sum()/(pc**2).sum(); k=(Xc*pc).sum(1)/(pc**2).sum(1)/a\n    r=Xc-(a*k)[:,None]*pc\n    # uncentered: X ~ off + a*k*cold\n    print(nm,'a',round(a,4),'k cv',round(k.std()/k.mean(),3),'R2 cent',round(1-(r**2).sum()/(Xc**2).sum(),4), 'X mean', X.mean(0).round(3), 'X std', X.std(0).round(3))\n    # common-mode check: does mean over 8 dims follow a*k*mean(cold)?\n    cm=X.mean(1); cmo=cold.mean(1); print('  common-mode corr', np.corrcoef(cm, (a*k)*cmo)[0,1], 'cm std', cm.std(), 'pred std', ((a*k)*cmo).std())\nx=10*(M-M.mean(0))+C; print('energy RES/total in pool x (c=10)', (x[:,RES]**2).sum(1).mean()/(x**2).sum(1).mean())"
}
```

