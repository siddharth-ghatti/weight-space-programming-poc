## reasoning summary

The leakage std of about 0.03 is tiny next to S·gap of about 92, so it's negligible and I'll note that. Next is verification: I'll run predict on the practice set, using 'edited' with n=512 and some seed, and check the per-item logits in predictions.json against v2_expected_logits. Since every practice item is a training point, I expect close to 100% accuracy. The v1 artifacts are saved separately, so I can fall back to them if v2 fails.

The fragment "ert." seems to be the tail of a word like "revert," which fits the fallback-to-v1 plan I was just laying out. So I'm noting that reverting to the saved v1 artifacts is the backup if v2 fails verification.



The leakage into the score directions (std ~0.03) is negligible next to S·margin (~92). I'm updating the notes, then running a predict pass to confirm the real network matches the intended logits.



## tool_use: run_python

```json
{
 "code": "with open('notes.md','a') as f: f.write(\"\"\"- KEY FINDING (runs 24-31): nonlinear kernel readout beats linear. GELU-neuron kernel machine on Z=standardized LN(cls11+2*mean11):\n  phi_i = gelu(beta * Z.Z_i/768), beta=1, ridge (dual) lam=0.01 -> honest CV 0.977-0.986 over 3 splits vs linear 0.972-0.976.\n- v2 (run33): layer11 attn O=2I (label-free); FFN = 1496 neurons centered at practice samples (W1=(beta/768)Z_i/sd, b1=-W1 mu),\n  W2 = S*U Wf^T (S=100, U = 8 lowest-variance dirs of n, orth to ones), b2=U(S b0 - U^T nbar); LN2 gamma1 beta0;\n  pooler rows0-7 = 0.05 U^T, bias 0; classifier = 10*I on first 8 pooled units, bias 0. Files: kernel_v2.npz, U_v2.npy, v2_expected_logits.npy.\n  v1 fallback artifacts: M_v1.npy, v1_cls_W/b.npy, LN_est_v1.npy (pooler bias for v1 = -0.5*mean(LN_est_v1)).\n\"\"\")\nprint('ok')"
}
```

## tool_use: forward_pass

```json
{
 "kind": "predict",
 "model": "edited",
 "n": 512,
 "seed": 2
}
```

