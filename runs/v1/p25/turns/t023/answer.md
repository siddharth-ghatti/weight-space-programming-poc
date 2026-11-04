## reasoning summary

I've got the baseline at about 96.7%, so next I'm turning that into an implementation. I'm fitting the final ridge on all 1496 samples with the combined features, lam=300 and the 0.7 weight on the standardized C block, then converting the logits back into original-space weights.

Now I'm working out how to convert the centered weights so they match what the implementation actually computes. Since the input is centered over the kept dimensions, I can shift each weight vector to sum to zero without changing its output, which makes the implementation line up with the fitted model. Next I'm setting up the layer 11 edits, starting by zeroing the query weights and biases so attention goes uniform.

Next I'm setting up the pooler weights, starting with the first eight rows scaled by s.

Working out the pooler rows now: each one combines the kept-dimension weights with a term on the channel, scaled by s over sigma. I'm checking how the channel's offset from the mean lands in that row, since that term doesn't cancel the way the others do.

I'm checking whether the constant offset from the channel's mean in the pooler rows can be absorbed as a bias shift, since dim 381 is a known outlier dimension and could carry a large mean.

I'm checking how big the channel's mean is, since dim 381 and 308 are outliers and a large constant there would just shift the bias, so I'll fold it into b_k. I'm also weighing whether a scale of c=100 keeps the logit terms dominant enough in the layer-norm variance.

I'm checking whether the constant c=100 is large enough to keep the logit terms dominant in the layer-norm variance, and I'm estimating the scale of the pooled arguments to confirm the tanh outputs stay out of saturation. The contamination from the low-variance channel dimensions looks negligible next to the logit differences, so I'm moving on to checking the attention mask.

I'm going to put the attention-mask question aside, since eval mode means no dropout to worry about. I'll also check whether the harness wraps the model in a classifier head with dropout, though eval mode should cover that. Next I'm writing the edit logic as a reusable script, build_readout.py, so I can rerun it with different C and M features from the tapped model later.



The concat [C11, M11] gives about 96.7% CV with wc=0.7 and λ≈300. Next I'll fit this ridge on all 1496 practice sentences and put the construction in a reusable script.



## tool_use: run_python

```json
{
 "code": "code = r'''\nimport numpy as np\ndef fit_readout(C, M, y, wc=0.7, lam=300.0, nch=8):\n    \"\"\"Concat ridge on [C (cls at L11 input), M (mean at L11)] -> returns dict with B (on keep dims, sum0), A, b, ch, keep\"\"\"\n    v=C.var(0); ch=np.argsort(v)[:nch]; keep=np.setdiff1d(np.arange(768),ch)\n    Ck=C[:,keep]; Cc=Ck-Ck.mean(1,keepdims=True)\n    muC=Cc.mean(0); sdC=Cc.std(0)+1e-6; muM=M.mean(0); sdM=M.std(0)+1e-6\n    Z=np.concatenate([wc*(Cc-muC)/sdC,(M-muM)/sdM],1)\n    Y=np.eye(8)[y]-1/8\n    zm=Z.mean(0); Zc=Z-zm\n    Wt=np.linalg.solve(Zc.T@Zc+lam*np.eye(Z.shape[1]),Zc.T@Y)   # [1528,8]\n    Wc=Wt[:len(keep)]; Wm=Wt[len(keep):]\n    B=(wc*Wc/sdC[:,None]).T          # [8,760] applied to Cc\n    A=(Wm/sdM[:,None]).T             # [8,768] applied to M\n    b=Y.mean(0) - zm@Wt - (wc*muC/sdC)@Wc*0  # placeholder fixed below\n    # exact intercept: logits = Zc@Wt + Ymean ; Z = [wc(Cc-muC)/sdC, (M-muM)/sdM]\n    b = Y.mean(0) - zm@Wt - (wc*muC/sdC)@Wc - (muM/sdM)@Wm\n    B=B-B.mean(1,keepdims=True)\n    return dict(B=B,A=A,b=b,ch=ch,keep=keep)\n\ndef logits_offline(R, C, M):\n    Ck=C[:,R['keep']]\n    return Ck@R['B'].T + M@R['A'].T + R['b']\n\ndef write_layer11_readout(R, save_edit, c=100.0, s=1.0, hch_mean=None):\n    L='bert.encoder.layer.11.'\n    z768=np.zeros(768,np.float32)\n    save_edit(L+'attention.self.query.weight', np.zeros((768,768),np.float32))\n    save_edit(L+'attention.self.query.bias', z768)\n    Vw=np.zeros((768,768),np.float32); Vb=np.zeros(768,np.float32)\n    b=R['b'].copy()\n    if hch_mean is not None: b=b-hch_mean/c\n    Vw[:8]=R['A']; Vb[:8]=b\n    save_edit(L+'attention.self.value.weight', Vw); save_edit(L+'attention.self.value.bias', Vb)\n    Ow=np.zeros((768,768),np.float32)\n    for k in range(8): Ow[R['ch'][k],k]=c\n    save_edit(L+'attention.output.dense.weight', Ow); save_edit(L+'attention.output.dense.bias', z768)\n    save_edit(L+'attention.output.LayerNorm.weight', np.ones(768,np.float32)); save_edit(L+'attention.output.LayerNorm.bias', z768)\n    save_edit(L+'output.dense.weight', np.zeros((768,3072),np.float32)); save_edit(L+'output.dense.bias', z768)\n    save_edit(L+'output.LayerNorm.weight', np.ones(768,np.float32)); save_edit(L+'output.LayerNorm.bias', z768)\n    Pw=np.zeros((768,768),np.float32)\n    for k in range(8):\n        Pw[k,R['keep']]=s*R['B'][k]; Pw[k,R['ch'][k]]=s/c\n    save_edit('bert.pooler.dense.weight', Pw); save_edit('bert.pooler.dense.bias', z768)\n    Cw=np.zeros((8,768),np.float32); Cw[np.arange(8),np.arange(8)]=1.0\n    save_edit('classifier.weight', Cw); save_edit('classifier.bias', np.zeros(8,np.float32))\n'''\nopen('readout.py','w').write(code)\nimport importlib, sys; sys.path.insert(0,'.')\nimport readout\nF=np.load('feats.npz'); y=F['y']\nR=readout.fit_readout(F['C11'],F['M11'],y)\nlg=readout.logits_offline(R,F['C11'],F['M11'])\nprint('train acc offline',(lg.argmax(1)==y).mean())\nprint('ch',R['ch'],'C11 mean at ch',F['C11'][:,R['ch']].mean(0).round(3),'std',F['C11'][:,R['ch']].std(0).round(3))\nprint('logit range',lg.min().round(2),lg.max().round(2), 'B norm',np.linalg.norm(R['B'],axis=1).round(2))\nhm=F['C11'][:,R['ch']].mean(0)\nreadout.write_layer11_readout(R, save_edit, c=100.0, s=1.0, hch_mean=hm)\nnp.savez('readout_base.npz', **R)\nprint(edits())"
}
```

