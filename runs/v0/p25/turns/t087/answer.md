## tool_use: run_python

```json
{
 "code": "\nimport json, os\nfrom scipy.special import erf\nprint(os.listdir('pass_10'))\nk=np.load('kernel_v3.npz')\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nC=k['C']; Kc=gelu(C@C.T/768)\nA=k['A']; F=k['F']\nout=Kc@A+k['b0']\nE=np.load('v3_expected_logits_practice.npy')\nprint(np.corrcoef(out[:1496].ravel(),E.ravel())[0,1], (out[:1496].argmax(1)==E.argmax(1)).mean())\nprint(np.abs(F-A).max(), F[:2], A[:2])\nprint(np.abs(C).mean(), C.std(0)[:5], k['w'][:3], k['w'][-3:])\n"
}
```

