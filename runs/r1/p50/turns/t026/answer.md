## tool_use: run_python

```json
{
 "code": "exec(open(WORK/'common.py').read())\nd = WORK/'pass_10'\nlg = np.load(d/'logits.npy'); report('m1536 orig head', lg)\npooled = np.load(d/'pooled.npy'); cls = np.load(d/'cls_hidden_L6.npy')\nfor lam in [1,10]:\n    report(f'm1536 out_proj refit pooled lam{lam}', ridge_cv(pooled, BL, lam, folds, std=False))\nfor lam in [10,100]:\n    report(f'm1536 ridge cls lam{lam}', ridge_cv(cls, BL, lam, folds))\nc6 = np.load(WORK/'pass_2/cls_hidden_L6.npy')\nprint('rel err cls', np.linalg.norm(cls-c6,axis=1).mean()/np.linalg.norm(c6,axis=1).mean())\n# margin analysis\nsrt = np.sort(BL,1); marg = srt[:,-1]-srt[:,-2]\np = lg.argmax(1)\nq = np.quantile(marg,[0.1,0.25,0.5])\nprint('base margin quantiles', q)\nfor lo,hi in [(0,q[0]),(q[0],q[1]),(q[1],q[2]),(q[2],99)]:\n    s=(marg>=lo)&(marg<hi); print(f\"margin [{lo:.2f},{hi:.2f}) n={s.sum()} agr {np.mean(p[s]==bp[s]):.3f}\")\ndl = lg-BL; print('logit diff std', dl.std(0))"
}
```

