## message

Before verifying, I'll update notes.md with this session's findings so they survive a handoff.

## tool_use: run_python

```json
{
 "code": "add = r'''\n## Session 3 (from turn 80)\n- CORRECTION: pass24 was the L5 probe (L0-4 m1024 seq), NOT a full m1024 eval; \".896\" was bogus. Offline (L5 FFN+LN+head on probed a5): L0-4 m1024seq + L5 full -> agr .965; L5 pruning (CV) costs ~nothing (agr .997 w/ base inputs).\n- pass25 = v1 m1536 (edits_m1536_v1/): agr .981 acc .910 F1 .897, rms dlogit .172. pass26 = v2 (fit2.py: lam .1, g2sig+weighted for L3-4; edits_m1536_v2/): agr .980 (no change).\n- Tried, no help: linear-units (reduced-rank linear map) in compensation; e2e L5 FFN correction to base targets; CLS-token errors are tiny (not the issue).\n- Learning curve flattening (L1 m1536: 12.6k->16.9k tok: .105->.102).\n- Noise decomposition (zeroing-sim, passes 27/28): L0-1 pruned only -> rms2 .0141 (48%), L2-4 only -> .0107 (37%), full .0291. Per layer roughly equal; L0,L1 ~24% each.\n- Noise ~ 1/sqrt(seq len) (short paras noisiest). Training data seqs all >=120 tok.\n- Greedy compensation-aware selection (Schur, post-LN weighted) holdout gains: L0 3.5%, L1 2%, L2 5%, L3 10%, L4 20% -> ~5% total noise var. fit3.py build3(m) = greedy_w L0-4 + var L5. edits for v3 built (keep_m1536_v3.json).\n- MEMORIZATION finding: edit lowers margin of base-right items (-.06) and RAISES margin of base-wrong items (+.17); flips 18:2 against gold. => pruning removes memorized label info; practice acc drop overstates test drop.\n- Risk sim (400 test, base conf acc .775): P(fail) ~5% @6 test flips, 10% @8, 16% @10, 24% @13. De-biased est test flips m1536 ~8-11 (k=1.5-2). EV (linear score) flat m1024-1536 -> choose m1536.\n'''\nopen('notes.md','a').write(add)\n"
}
```

