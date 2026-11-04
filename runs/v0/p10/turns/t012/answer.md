## reasoning summary

I'm going to check what mean_hidden actually refers to by comparing the mean of the layer-9 hidden states over the valid tokens (CLS and SEP included) against the mean_hidden_L9 from the second pass, for the same ids. After that I'll look at how often other tokens attend to CLS in each layer, and at a layer-0 CLS detector built from u = Σ⁻¹e, normalized, along with its leakage statistics.



Pass 3 is done; I'm checking whether mean_hidden includes the [CLS]/[SEP] tokens, measuring how much attention other tokens give [CLS] per layer, and then building the layer-0 [CLS] detector.



## tool_use: run_python

```json
{
 "code": "\nmeta = json.load(open('pass_3/meta.json')); print(meta.keys())\nmask = np.load('pass_3/attention_mask.npy').astype(bool)\nidx = np.arange(0,1024,16)\nT9 = np.load('pass_3/token_hidden_L9.npy'); M9 = np.load('pass_2/mean_hidden_L9.npy')[idx]\nm_all = np.array([T9[i][mask[i]].mean(0) for i in range(64)])\nm_noss = np.array([T9[i][mask[i]][1:-1].mean(0) for i in range(64)])\nprint('diff incl', np.abs(m_all-M9).max(), 'diff excl', np.abs(m_noss-M9).max())\n# attention to CLS by non-CLS tokens, per layer, per head\nfor l in range(1,13):\n    A = np.load(f'pass_3/attention_L{l}.npy')\n    vals=[]\n    for i in range(64):\n        n = mask[i].sum()\n        vals.append(A[i,:,1:n,0].mean(1))  # heads x\n    vals = np.array(vals).mean(0)\n    sep=[]\n    for i in range(64):\n        n = mask[i].sum(); sep.append(A[i,:,1:n,n-1].mean(1))\n    print(l, 'toCLS mean %.3f max head %.3f'%(vals.mean(), vals.max()), ' toSEP mean %.3f'%np.array(sep).mean())\n"
}
```

