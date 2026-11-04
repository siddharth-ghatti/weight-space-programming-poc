## reasoning summary

I'm checking which heads in layers 2 and 6 carry the signal, and whether their output projections write into any dimensions besides the residual ones. My plan is to zero their residual writes and see whether they could be repurposed.

Now I'm writing notes on the state so far, then moving on to the layer 0 and layer 1 changes. For the layer 0 and layer 1 residual cuts, I plan to fit ridge regressions that map the residual-stream dimensions onto the kept ones, then fold those fits into the weights and biases of the affected layers. I still need to check how well those regressions fit.

I'm noting that if the cut turns out lossy, it only touches eight dimensions. I also need to zero the RES rows of the layer 1 output and FFN weights, plus the RES gammas and betas in the attention and output layer norms, which the layer 0 and layer 1 fits left untouched.

Next I'm writing the carry-direction fit: a joint kernel ridge regression on the M9e features plus the kept M0 dimensions, with the M0 part scaled and centred, so the carry weights can be read back in raw units.

I'm settling the scale for the carry values so they come out at roughly half a unit per dimension at h1. Then I'll write up the current state and save the design parameters to a JSON file so they can be reused across handoffs.



Layers 2–8 already cut every reserved-dim read and write, with the old carry heads at layer 2 head 0 and layer 6 head 11. I'm recording the plan and findings in notes.md before making any changes.



## tool_use: run_python

```json
{
 "code": "s = open('notes.md').read()\ns += \"\"\"\n## Run 3 (handoff 3) findings & plan\n- Exposure passes 10-18 were taken on UNPRUNED upstream => current m1024 edits = non-sequential energy-select+ridge refit (lib3.apply_prune). Backup in backup_m1024/ (+backup_m1024_structure.json). keep_m1536.json = m1536 version.\n- Greedy unit selection == energy (no gain). Per-layer heldout post-LN err: m768 .21, m1024 .19, m1536 .145.\n- 5-seed 10f KRR on ids p0000-p1023 (exact RBF cos kernel): base M9 97.25, m1024 96.91, m1536 97.15.\n- KEY: kernel(M9) + JOINTLY-fit LINEAR(M0) (dual K=K9+0.5*C0C0^T, C0=centered M0 scaled to unit mean sq norm) = 98.46 base full-1496 (K9 alone 98.17). On 1024 subset: base 98.16, m1024 97.91, m1536 98.09. Stacked OOF scores do NOT help; joint fit does. M1/M2 linear worse than M0 for pruned net.\n- Anchor/FFN realization: features GELU(cos(z,anchor)) (a=1,b=0!; cos offdiag mostly -0.3..0.4), Nystrom penalty alpha^T Kaa alpha (+ I/c0 on linear block), lam~0.01-0.03. Pivoted-Cholesky anchor selection: 1000 anchors = 98.55 (== all anchors). Random anchors lose 0.4%.\n- Carried 8-vector in RES dims is scaled by a per-sample scalar through LNs (exact proportionality, cv ~4% over 3 layers).\n- Layer-0 head 0 is near-uniform (entropy .97): forcing uniform costs 2.4% residual err (drop=6.7%). => carry head = L0 head 0.\n- pass_20: base token_hidden L1..L10 for expo ids (64). pass_23: base attention L0 + token_hidden L0,L1 for expo ids.\n- PLAN: (1) L0 head0 -> uniform; value dims 0..55 = rank-56 approx of its content map, dims 56..63 = carry s*w^T(M0-mu) - mean(h0)[RES] -> RES dims. Cut RES reads in L0 FFN, L1 (regression compensation), zero RES writes L0/L1, LN RES gamma=1,beta=0 in L0/L1. Zero old carry writes (L2 h0, L6 h11).\n  (2) sequential drift-corrected re-prune m=1024 (expose layer l with upstream pruned; target = base h_{l+1} from pass_20).\n  (3) pool layer = orig 11: uniform attn, c*(M9-mu9) -> LN -> FFN anchors (pivchol ~1000) GELU(cos) -> class scores into RES; carried linear term passes via residual; pooler decodes RES (centered); classifier = I8.\n\"\"\"\nopen('notes.md','w').write(s); print('ok')"
}
```

