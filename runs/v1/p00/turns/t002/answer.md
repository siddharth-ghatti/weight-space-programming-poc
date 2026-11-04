## reasoning summary

I'm going to write notes.md, putting the strategy at the top.



I'm starting fresh and writing the plan into notes.md: uniform mean-pooling in layer 0, pass-through sublayers everywhere else, and a closed-form fit of the class readout.



## tool_use: run_python

```json
{
 "code": "notes = \"\"\"# notes.md \u2014 insurance classifier by weight editing (0 forward passes available!)\n\n## Strategy (run 1)\n- No forward passes => fully analytic construction, no verification possible. Keep it simple & exact.\n- Layer 0 attention: Q,K = 0 -> uniform attention over all valid tokens (incl CLS/SEP) => context = W_v mean_j x0_j + b_v.\n  value rows 0..7 = Wc P diag(1/gamma_emb); bias cancels Wc P (beta/gamma) and adds intercept.\n  output.dense col c = s*q_c (q_c zero-mean orthonormal, e.g. (e_a-e_b)/sqrt2), s huge so CLS residual negligible.\n  LN gamma=1,beta=0 -> CLS = norm(s Q z + x0_cls) ~ sqrt(768) Q z/||z||.\n- All other sublayers pass-through: attn.output.dense=0 (layers1-11), output.dense=0 (all layers), all LN = (1,0).\n- Pooler rows c<8 = eps*q_c, bias 0; classifier weight = [I8 | 0], bias 0. argmax preserved (tanh monotone).\n- z fitted closed-form (ridge/LDA) on proxy features: mean_j P w_tok / sigma~ (raw word-emb rows; P removes ones vec,\n  position subspace, token-type vec). NOT computing embedding LN outputs (would be reimplementing forward).\n\"\"\"\nopen(WORK/'notes.md','w').write(notes)\nprint('ok')\n"
}
```

