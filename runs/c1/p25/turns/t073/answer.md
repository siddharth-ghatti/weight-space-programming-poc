## tool_use: run_python

```json
{
 "code": "d=np.load('base_reads.npz'); y=d['y'][:1024]; R=np.load('reserved.npy')\nfz=np.load('fit_k10.npz'); F=fz['F'][:1024]\nLg=np.load('pass_11/logits.npy'); H=np.load('pass_11/cls_hidden_L10.npy')\nprint('net acc', (Lg.argmax(1)==y).mean(), 'agree with numpy argmax', (Lg.argmax(1)==F.argmax(1)).mean())\npred=np.tanh(F-F.mean(1,keepdims=True))\nprint('logit vs tanh(cenF): maxabs %.4f corr %.5f'%(np.abs(Lg-pred).max(), np.corrcoef(Lg.ravel(),pred.ravel())[0,1]))\n# implied scale\ncf=F-F.mean(1,keepdims=True); hR=H[:,R]-H[:,R].mean(1,keepdims=True)\nsc=(hR*cf).sum(1)/(cf*cf).sum(1); print('hR/(eps*cenF) scale mean %.4f sd %.4f'%(sc.mean()/0.1, sc.std()/0.1), 'resid rel', np.linalg.norm(hR-sc[:,None]*cf)/np.linalg.norm(hR))\nsrt=np.sort(Lg,1); print('min top1 margin', (srt[:,-1]-srt[:,-2]).min(), 'max |logit|', np.abs(Lg).max())"
}
```

