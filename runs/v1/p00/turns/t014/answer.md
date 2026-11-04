## reasoning summary

I'm settling on 0.3 for the regularization, since the 0.2 versus 0.3 difference is trivial and 0.3 is already saved. Next I'll update the notes and submit, since there are no forward passes left to verify. Before that, I want a quick sanity check on the magnitude of the entries in Wv (G = A/γ) and on the range of the context z values in the proxy.

I'm checking the scale of the context vectors in the proxy next: their norms look like they sit around 0.5 to 1, which makes the scaled norm far larger than the roughly 25 from the CLS term. Now I want the minimum norm across the practice set to confirm that holds everywhere.



Both λ values from 0.2 to 0.3 give the same accuracy, so I'm keeping the saved λ=0.3 for its extra regularization. Next I'll confirm that s·‖z‖ is far larger than the CLS residual, so the residual can't change the argmax.



## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'feat_lib.py').read())\nPm = build_P(30); Fa = feats(Pm,'approx'); sc = np.linalg.norm(Fa,axis=1).mean(); Fa/=sc\nYt = np.eye(8)[y]; Yt -= Yt.mean(0)\nWm = np.linalg.solve(Fa.T@Fa + 0.3*np.eye(768), Fa.T@Yt)\nz = Fa@Wm; zs=np.sort(z,1); m = zs[:,-1]-zs[:,-2]\nprint('||z|| min/median', np.linalg.norm(z,axis=1).min().round(3), np.median(np.linalg.norm(z,axis=1)).round(3))\nprint('margin pct 1/5/50', np.percentile(m,[1,5,50]).round(4), 'min', m.min())\nprint('max |W_v|', np.abs(W('bert.encoder.layer.0.attention.self.value.weight')).max(), 'max |W_o|', np.abs(W('bert.encoder.layer.0.attention.output.dense.weight')).max())\nemb_g = base('bert.embeddings.LayerNorm.weight'); emb_b = base('bert.embeddings.LayerNorm.bias'); d=np.load(WORK/'readout_dims.npy')\nprint('worst-case residual shift / S_BIG:', ((np.abs(emb_g[d])*np.sqrt(768)+np.abs(emb_b[d])).reshape(8,2).sum(1)/np.sqrt(2)/1e4).max().round(4))\nopen(WORK/'notes.md','a').write(\"\\n- Run 14: lambda sweep 5x5 CV: 0.2-0.3 ~95.8% acc/F1; kept 0.3. ||z|| min ~%.2f, so s*||z|| >> residual. READY TO SUBMIT.\\n\" % np.linalg.norm(z,axis=1).min())\n"
}
```

