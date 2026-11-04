## reasoning summary

Now I'm saving the v4 edits to the intermediate and output dense weights and biases. Then I'll check them with a predict pass on the 512 practice ids, comparing the model's logits against an expected value I still need to work out from the layer-norm output.

Working out the expected logits for each practice sample means recomputing the second layer-norm from the first-layer output, mean-centering and scaling the residual, then pushing it through the tanh-scaled pooling to get the predicted values.



## message

The fit is numerically stable (1e-3 input noise changes logits by less than 1e-3). Now I'll save the v4 FFN edits and compute the expected outputs so I can verify with a predict pass.

## tool_use: run_python

```json
{
 "code": "k=np.load('kernel_v4.npz'); C,mu,sd,A,b0,F,nbar=k['C'],k['mu'],k['sd'],k['A'],k['b0'],k['F'],k['nbar']\nU=np.load('U_v2.npy'); S=100; N=len(C)\np='bert.encoder.layer.11.'\nW1=np.zeros((3072,768)); W1[:N]=C/sd/768\nb1=np.zeros(3072); b1[:N]=-(W1[:N]@mu)\nW2=np.zeros((768,3072)); W2[:,:N]=S*U@A.T\nb2=U@(S*b0-U.T@nbar)\nsave_edit(p+'intermediate.dense.weight',W1.astype(np.float32))\nsave_edit(p+'intermediate.dense.bias',b1.astype(np.float32))\nsave_edit(p+'output.dense.weight',W2.astype(np.float32))\nsave_edit(p+'output.dense.bias',b2.astype(np.float32))\n# expected logits\nn=C*sd+mu\nx=n+(S*F-(U.T@nbar))@U.T\nxm=x-x.mean(1,keepdims=True); ln=xm/np.sqrt((xm**2).mean(1,keepdims=True)+1e-12)\nlogits=10*np.tanh(0.05*ln@U)\nprint((logits.argmax(1)==k['yc']).mean(), logits[:2])\nnp.save('v4_expected_logits.npy',logits)\nopen('notes.md','a').write('''\n## Handoff 4 (turn 120+)\n- CV (3-4 seeds, 5-fold, practice held out): prac-only .9816, +m .9866, +m3m4 .987, +all(m,m3,m4) w.3-.5 lam.003-.01: ~.992-.993. beta=1 still best.\n- v4 SAVED (run120): centers = 1496 prac + 543 m + 467 m3 + 480 m4 = 2986 neurons, w=0.4 mine, lam=0.007, beta 1. kernel_v4.npz, v4_expected_logits.npy (rows in C order). kh.py = kernel helpers (fit, fitk, load, run_cv).\n  Next: verify with predict pass on practice.\n''')\n"
}
```

