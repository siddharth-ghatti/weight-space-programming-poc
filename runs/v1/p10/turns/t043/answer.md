## reasoning summary

The cross-validation score on all 1496 sentences looks inflated, since the projection A was fitted on every label, so the dimensions already carry label information. The more honest figure is the nested CV estimate of roughly 97.6–97.8%. The 97.9% holdout result has the same leak, because A was also fit on the 472 holdout sentences.

I'm setting the lam2=0.2 choice aside for now, since the nested CV already accounts for the leakage in the D dims. Scale doesn't matter here because ridge_fit standardizes. Now I'm moving on to the pooler and classifier weights, with the logit magnitudes still to check.

Next I'm making sure the offline accuracy holds end to end: a predict pass over the 512 practice items with the edited model, which should land near the ~99% training accuracy. The pooler orientation is the thing to confirm, since the weight layout matters for the rows I'm setting.



The CV figure on 1496 is inflated because the projection was fit on all the labels, so I'm keeping the nested-CV estimate of about 97.6–97.8% at lam=0.2. Next I'll build the readout with the pooler rows scaled and a 10·I classifier on dims 0–7, then compare its predictions against the offline ridge.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\ny=labels_all()\nX=np.vstack([np.load('pass_7/cls_hidden_L12.npy'),np.load('pass_8/cls_hidden_L12.npy')]).astype(np.float64)\nWf,b=ridge_fit(X,y,0.2)\nL=X@Wf+b\nprint('train acc', (L.argmax(1)==y).mean(), 'logit range', L.min(), L.max())\nWp=np.zeros((768,768),np.float32); bp=np.zeros(768,np.float32)\nWp[:8,:]=(0.1*Wf.T).astype(np.float32); bp[:8]=(0.1*b).astype(np.float32)\nWc=np.zeros((8,768),np.float32); Wc[np.arange(8),np.arange(8)]=10.0\nsave_edit('bert.pooler.dense.weight',Wp); save_edit('bert.pooler.dense.bias',bp)\nsave_edit('classifier.weight',Wc); save_edit('classifier.bias',np.zeros(8,np.float32))\npooled=np.tanh(X@Wp.T.astype(np.float64)+bp)\nlg=pooled@Wc.T\nprint('sim acc', (lg.argmax(1)==y).mean(), 'agree', (lg.argmax(1)==L.argmax(1)).mean())\nnp.save('readout_v3_Wf.npy',Wf); np.save('readout_v3_b.npy',b); np.save('sim_logits_v3.npy',lg)\nwith open('notes.md','a') as f:\n    f.write(\"\\n## v3 readout (saved): ridge lam .2 on pass_7+pass_8 c12 (1496). pooler rows0-7=0.1*Wf^T, bias 0.1*b; classifier 10*I. train acc printed in run 35. Files readout_v3_*.npy, sim_logits_v3.npy\\n- Next: pass9 predict edited on 512 ids to verify pipeline vs sim_logits.\\n\")"
}
```

