## reasoning summary

Kernel methods on these features reach about 98%, against 96.9% for linear. The RBF result with a large bandwidth and small lambda looks close to linear with weak regularization, so I want to check that. Linear ridge with tiny lambda did worse, at 0.915 for lambda 1, and I'm still working out how the small-lambda kernel fits with that.

I'm now weighing how to approximate the kernel's nonlinearity within the pooler's 768 tanh units, since the 3072 FFN units can't be read out through the 768-dim residual bottleneck. The most workable option looks like choosing pooler weights that resemble kernel centers over the training points and then fitting a ridge classifier on the pooled features.

I'm weighing whether to simulate the pooler's tanh layer in numpy to screen candidate weight designs, but the rules forbid reimplementing the forward pass, so I'll instead use the actual model forward pass to read pooled outputs for each candidate design, with about 45 passes available for a few candidates.

I'm considering whether Nyström-style sampling with 768 centers could approximate the 98% kernel result better than my random-direction pooler weights, since the tanh units aren't RBF features and that gap matters.

I'm picturing the pooler's 768 tanh units as a bank of nonlinear features, and I'm weighing whether tanh(a·⟨x,z⟩+b) could stand in for the kernel. Since that's not PSD and 768 units can't match 1496 centers, I'd rather build the pooler weights from data-derived directions, with a few near-linear units for the top components and the rest nonlinear, then fit a ridge classifier on the pooled outputs.

I'm weighing whether to spend a little more effort on the nonlinear readout, since the PCA-directed tanh units plus random ones could land close to the 98% kernel result, and the test split comes from the same distribution, so it seems worth a try.

I'm considering whether the 3072 FFN units could feed a nonlinear readout by routing ridge weights for the class scores into a few residual dimensions, but fitting that would need GELU features I can't read out directly through the 768-dim bottleneck.

I'm sketching the first pooler design now: standardize the final CLS vector, then whiten its top principal components so the tanh units get well-scaled inputs.

I'm planning to approximate the kernel with a pooler of random tanh features over standardized, whitened inputs, and I'm unsure how to pick the scale without simulating the layer myself. Simulation is off the table, so I'd screen two or three candidate designs on a 1024-point subset using the model's own forward pass, then fit the full set once I've chosen one.

I'm planning a hybrid pooler: some units kept near-linear to carry the top whitened PCA components, the rest as nonlinear random features. A ridge fit on the pooled outputs should then recover the linear solution and add some nonlinear gain. I'm considering whether to test standardized rather than whitened inputs for the RBF-like scale.

I'm going to try random tanh features on the standardized vectors, with the weight scale set so the pre-activations have roughly unit variance. Since the class structure may not be isotropic, I'll screen three pre-activation scales, 0.5, 1 and 2, before choosing one.

I'm planning to check each pooler candidate on a 1024-point subset with 5-fold CV against the linear baseline, then pick the best and read the full set. I'm also considering a candidate whose pooler rows are scaled training samples, which would act like a sigmoid kernel over 768 centers.

I'm setting up the pooler's input transform, using the layer-12 CLS vector as the input, and building the pooled weights from a scaled Gaussian projection with a bias shift, and saving the seeds so the setup can be reproduced.



Nonlinear readouts add about 1% (RBF and polynomial kernel ridge reach ~98.0–98.2% vs 96.9% linear), so I'll test three tanh pooler designs on the first 1024 practice items, starting with design A.



## tool_use: run_python

```json
{
 "code": "E=np.load('feats/edA_cls12.npy'); lab=np.load('feats/y.npy')\nmu=E.mean(0); sd=E.std(0)+1e-6\nnp.save('feats/edA_mu.npy',mu); np.save('feats/edA_sd.npy',sd)\ndef make_pool(kind, s, seed, bsd=0.5):\n    rng=np.random.default_rng(seed)\n    if kind=='gauss':\n        G=rng.standard_normal((768,768))/np.sqrt(768)   # unit-variance preact if Z ~ isotropic unit\n    elif kind=='centers':\n        Z=(E-mu)/sd\n        idx=rng.choice(len(E),768,replace=False)\n        G=Z[idx]/np.linalg.norm(Z[idx],axis=1,keepdims=True)  # unit-norm rows -> preact = proj of Z\n        G=G/np.sqrt(np.mean(((Z@G.T))**2))  # normalize so preact var ~1\n    Wp=s*G/sd[None,:]\n    bp=-Wp@mu + bsd*rng.standard_normal(768)\n    return Wp.astype(np.float32), bp.astype(np.float32)\nWp,bp=make_pool('gauss',1.0,1)\nZ=(E-mu)/sd; pre=Z@(Wp*sd[None,:]).T\nprint('preact std (design check only)', pre.std(), np.abs(pre).max())\nsave_edit('bert.pooler.dense.weight',Wp); save_edit('bert.pooler.dense.bias',bp)\nwith open('notes.md','a') as f: f.write(\"\"\"- Edit A verified: edited cls12 == lnorm(mean11 base) (cos .9999997). feats/edA_cls12.npy (pass_4+5). Linear ridge CV 96.9%, LDA 96.9%.\n- Kernel ridge on edA feats: RBF ~98.2%, poly2 ~98.0% => try nonlinear readout via pooler random tanh features, classifier = ridge on read pooled.\n- Pooler candidates (gauss s=1 seed1; gauss s=2.5; centers s=1) read on p0000-p1023, compare 5-fold CV.\n\"\"\")\n"
}
```

