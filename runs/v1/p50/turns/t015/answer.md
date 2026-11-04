## reasoning summary

I'm settling on c=2 since it scored slightly higher (96.4 vs 95.9), though the gap is within noise. The R signal still shows up in both the mean-token and CLS representations, so keeping it is fine. Next I'll start writing the builder script for the edits.

I'm going to write the builder as a reusable script, build_edits.py, that takes the parameters. The layer-2 aggregation head gets its own treatment, and layer 1's output LayerNorm is wiped.

Layer 11 is the aggregator: I'm zeroing its query weights and bias so attention goes uniform, setting values to identity with no bias, scaling the output projection by c=2, and resetting its LayerNorms and FFN output to identity or zero. The pooler and classifier can wait since I'll read the layer-12 CLS hidden state directly.

I'm fitting the ridge regression for D with lambda 300 on the standardized non-R coordinates across all 1496 samples, then deriving the bias so the value bias carries the intercept through the averaging. Next I want to check the typical standard deviation of the hidden states to see how the magnitudes will behave once the LayerNorms divide by sigma.

I'm checking pre-LN magnitudes next: I'll read the CLS and mean-token hidden states at each layer with g=1 on a sample of examples to see the R-coordinate scale, then adjust the LayerNorm gains from that. I'll do a single pass over 256 examples first, then decide whether a second full pass over all 1496 is worth it.

I'm going to run a quick diagnostic on 256 examples first, reading the CLS and mean-token hidden states at every layer, so I can see the R-coordinate scale before deciding on the LayerNorm gains. I'll then adjust those gains and do the full reads over all 1496 samples.

I'm checking the magnitude problem next: the LayerNorm gains could drift badly over 18 layers if the pre-LN scale is off, so a diagnostic read on 256 examples seems worth doing first. I'd read the CLS and mean-token states at every layer and compare the R-coordinate values against the known sentence logits, which gives a per-layer ratio to normalize the gains against.

I'm making the builder into a file, build.py, with a build(params) function so later runs can import it.



c=2 edges out a pure mean (96.4% vs 95.9%, within noise), so I'll use it. Next I'm writing a reusable `build.py` for the edits and saving the first version, leaving the pooler and classifier alone while I check how the carried signal survives the LayerNorms.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np, json\nR = [601, 222, 88, 83, 444, 506, 254, 495]\nH = 0          # sacrificed head in carry layer\nLC = 2         # carry layer (encoder index); wipe at output LN of LC-1\ndef fit_D(lam=300):\n    M=np.load(WORK/'base_mean.npy'); lab=np.load(WORK/'labels.npy'); Y=np.eye(8)[lab]\n    keep=np.setdiff1d(np.arange(768),R)\n    X=M[LC][:,keep]; mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd\n    Wr=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]), Z.T@(Y-Y.mean(0)))\n    D=np.zeros((8,768),np.float32); D[:,keep]=(Wr/sd[:,None]).T\n    b=(-(mu/sd)@Wr + Y.mean(0)).astype(np.float32)\n    return D,b\ndef build(gR=None, gw=1.0, c=2.0, lam=300, save=True):\n    \"\"\"gR: dict layer->(g_ln1,g_ln2) gain for R coords at LNs of layers LC..10\"\"\"\n    E={}\n    pre=lambda l: f'bert.encoder.layer.{l}.'\n    # wipe R at output LN of layer LC-1\n    g=base(pre(LC-1)+'output.LayerNorm.weight').copy(); bb=base(pre(LC-1)+'output.LayerNorm.bias').copy()\n    g[R]=0; bb[R]=0; E[pre(LC-1)+'output.LayerNorm.weight']=g; E[pre(LC-1)+'output.LayerNorm.bias']=bb\n    D,b0=fit_D(lam)\n    for l in range(LC,11):\n        p=pre(l)\n        for n in ['attention.self.query.weight','attention.self.key.weight','attention.self.value.weight','intermediate.dense.weight']:\n            w=base(p+n).copy(); w[:,R]=0; E[p+n]=w\n        for n in ['attention.output.dense','output.dense']:\n            w=base(p+n+'.weight').copy(); w[R,:]=0; E[p+n+'.weight']=w\n            bb=base(p+n+'.bias').copy(); bb[R]=0; E[p+n+'.bias']=bb\n        g1,g2=(gR or {}).get(l,(1.0,1.0))\n        for n,gg in [('attention.output.LayerNorm',g1),('output.LayerNorm',g2)]:\n            w=base(p+n+'.weight').copy(); w[R]=gg; E[p+n+'.weight']=w\n            bb=base(p+n+'.bias').copy(); bb[R]=0; E[p+n+'.bias']=bb\n    # carry head in layer LC\n    p=pre(LC); sl=slice(64*H,64*H+64)\n    qw=E[p+'attention.self.query.weight']; qw[sl,:]=0\n    qb=base(p+'attention.self.query.bias').copy(); qb[sl]=0; E[p+'attention.self.query.bias']=qb\n    vw=E[p+'attention.self.value.weight']; vw[sl,:]=0; vw[64*H:64*H+8,:]=D\n    vb=base(p+'attention.self.value.bias').copy(); vb[sl]=0; vb[64*H:64*H+8]=b0; E[p+'attention.self.value.bias']=vb\n    ow=E[p+'attention.output.dense.weight']; ow[:,sl]=0\n    for k in range(8): ow[R[k],64*H+k]=gw\n    # final layer 11 aggregator\n    p=pre(11)\n    E[p+'attention.self.query.weight']=np.zeros((768,768),np.float32)\n    E[p+'attention.self.query.bias']=np.zeros(768,np.float32)\n    E[p+'attention.self.value.weight']=np.eye(768,dtype=np.float32)\n    E[p+'attention.self.value.bias']=np.zeros(768,np.float32)\n    E[p+'attention.output.dense.weight']=(c*np.eye(768)).astype(np.float32)\n    E[p+'attention.output.dense.bias']=np.zeros(768,np.float32)\n    E[p+'output.dense.weight']=np.zeros((768,3072),np.float32)\n    E[p+'output.dense.bias']=np.zeros(768,np.float32)\n    for n in ['attention.output.LayerNorm','output.LayerNorm']:\n        E[p+n+'.weight']=np.ones(768,np.float32); E[p+n+'.bias']=np.zeros(768,np.float32)\n    if save:\n        for k,v in E.items(): save_edit(k,v.astype(np.float32))\n    return E,D,b0\n'''\nopen(WORK/'build.py','w').write(code)\nexec(code)\nE,D,b0=build()\nprint(len(E), 'edits saved'); print(sorted(edits())[:5])\nnp.save(WORK/'D.npy',D); np.save(WORK/'b0.npy',b0)\nM=np.load(WORK/'base_mean.npy'); lab=np.load(WORK/'labels.npy')\ns=M[2]@D.T+b0; print('in-sample acc of carried logits', (s.argmax(1)==lab).mean(), 's range', s.min().round(2), s.max().round(2))\n"
}
```

