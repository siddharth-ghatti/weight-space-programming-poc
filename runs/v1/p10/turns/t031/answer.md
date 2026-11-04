## reasoning summary

With this fold seed, F11 alone comes out at 94.6%, against 96.3% for seed 0, so the fold split matters a lot. Adding the layer-1 block brings it to 96.5%, and adding layer 7 on top changes nothing. The layer-6 gain I saw on seed 2 was only about 0.4 points, which looks like noise, so I'm leaning toward dropping that block for simplicity. Now I want to check why F11 alone reads 94.6 here.

I'm settling on a single D0 block, fitting A jointly on F11 and M1 with lamA=0.4 on the 1024 samples, and keeping the holdout honest. I'm weighing a larger 16-dim carry block, but since the full joint fit showed no gain over 8 dims, I'll stick with 8. Next I'll build the v2 A0 with the D0 rows zeroed and check that the new z std lines up with v1 so I can set the scale s accordingly.



The m7 block adds nothing under a different fold split, so I'm dropping the layer-6 block and keeping v2 as a single 8-dim carry block, and rebuilding with that next.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read()); exec(open('builder.py').read())\nlab=labels_all(); y=lab[:1024]\nD0 = np.load('D_dims.npy'); keep=np.setdiff1d(np.arange(768),D0)\nF11 = load('cls',12,'pass_3')[:,keep]\nM1 = load('mean',1,'pass_3').copy(); M1[:,D0]=0\nWj,_ = ridge_fit(np.concatenate([F11,M1],1), y, 0.4)\nA = Wj[F11.shape[1]:].copy(); A[D0]=0\nz = M1@A; print('z std', z.std(0).round(3), 'v1 was ~0.10-0.18')\ncenter = z.mean(0)\nnp.save('A_proj_v2.npy',A); np.save('center_v2.npy',center)\ns = 0.1*0.15/z.std(0).mean()\nprint('s', s)\nE,N = build(np.ones(12), np.ones(12), s=s, A=A, D=D0, center=center)\nprint(len(edits()))\nopen('notes.md','a').write(f\"\"\"\n## v1 results (pass_3 = edited v1 read 1024 all layers; pass_4 = edited v1 read p1024-1495 layers 0,11,12)\n- v1 holdout (fit readout on 1024, test 472): all dims 97.0% (lam .4), nonD only ~95.5-96.4%. CV on 1024 inflated (98.7) due to A leakage.\n- c12 == LN(mean_hidden L11) (harness mean includes CLS/SEP). D signal stable through layers (g=1 fine).\n- Fidelity: carried signal corr only ~0.8 with m0@A but ~0.95 with m1@A (FFN input x0 ~ h1). Proxy D-only holdout 88% vs actual 72%.\n- Layer-6 second block: no clear gain in nested CV -> skipped.\n## v2: A fit on joint ridge [F11(edited c12 nonD), M1 (pass_3 mean L1, D zeroed)] lam .4 on 1024 only; s={s:.4f}; saved A_proj_v2.npy, center_v2.npy\n- Plan: read v2 for 1024 (pass5) + 472 (pass6) -> holdout compare to v1 97.0 -> fit final readout on 1496 -> predict check.\n\"\"\")\n"
}
```

