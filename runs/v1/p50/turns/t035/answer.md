## tool_use: run_python

```json
{
 "code": "lab=np.load(WORK/'labels.npy')[:1024]\nlg=np.load(WORK/'pass_10/logits.npy'); exp=np.load(WORK/'expected_v3.npy')[:1024]\nprint('acc', (lg.argmax(1)==lab).mean(), 'agree', (lg.argmax(1)==exp.argmax(1)).mean())\nz=np.arctanh(np.clip(lg,-0.999999,0.999999))/0.5\nzc=z-z.mean(1,keepdims=True); ec=exp-exp.mean(1,keepdims=True)\nprint('max abs diff centered', np.abs(zc-ec).max(), 'mean', np.abs(zc-ec).mean(), 'scale of logits', np.abs(ec).mean())\n"
}
```

