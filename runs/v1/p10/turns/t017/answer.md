## reasoning summary

I'm checking the token-level spread of the D signal, which I estimate is around 0.5 with the scaling. I'm considering a scale of 0.1 to bring the token-level std down to about 0.05, and I'm weighing whether to center the z means by folding a shift into the b_2 offset, since a constant shift won't change classification. I'm settling on centering it that way.

I'm wrapping the edit code into a reusable function that takes the per-LayerNorm gains, the scale, and the offset as parameters, and saving it as a script file.



Sentence projections have std near 0.15, so I'll scale them by 0.1 and centre them to keep LayerNorm stable. Next I'm turning the edit builder into a reusable script with per-LayerNorm gains, and saving v1 with g=1.



## tool_use: run_python

```json
{
 "code": "builder = r'''\nimport numpy as np\npre='bert.encoder.layer.'\ndef build(g_attn, g_out, s=0.1, B=10.0, c=1000.0, A=None, D=None, center=None, save=True, carry=True):\n    \"\"\"g_attn/g_out: arrays len 12 of LN gains for D dims (index = layer).\"\"\"\n    E = {}\n    def get(n): \n        if n not in E: E[n] = base(n).copy()\n        return E[n]\n    W2_0 = base(pre+'0.output.dense.weight')\n    N = np.argsort(np.linalg.norm(W2_0,axis=0))[:len(D)]\n    if carry:\n        # layer 0\n        get(pre+'0.attention.output.LayerNorm.weight')[D]=0; get(pre+'0.attention.output.LayerNorm.bias')[D]=0\n        Wi = get(pre+'0.intermediate.dense.weight'); bi = get(pre+'0.intermediate.dense.bias')\n        Wi[N,:] = s*A.T; bi[N] = B\n        Wo = get(pre+'0.output.dense.weight'); bo = get(pre+'0.output.dense.bias')\n        Wo[D,:]=0; Wo[:,N]=0; Wo[D[:,None],N[None,:]] = np.eye(len(D)); bo[D] = -B - s*center\n        get(pre+'0.output.LayerNorm.weight')[D]=g_out[0]; get(pre+'0.output.LayerNorm.bias')[D]=0\n        # layers 1..10 protect\n        for l in range(1,11):\n            p=f'{pre}{l}.'\n            for k in ['query','key','value']:\n                get(p+f'attention.self.{k}.weight')[:,D]=0\n            get(p+'attention.output.dense.weight')[D,:]=0; get(p+'attention.output.dense.bias')[D]=0\n            get(p+'attention.output.LayerNorm.weight')[D]=g_attn[l]; get(p+'attention.output.LayerNorm.bias')[D]=0\n            get(p+'intermediate.dense.weight')[:,D]=0\n            get(p+'output.dense.weight')[D,:]=0; get(p+'output.dense.bias')[D]=0\n            get(p+'output.LayerNorm.weight')[D]=g_out[l]; get(p+'output.LayerNorm.bias')[D]=0\n    # final layer readout\n    p=pre+'11.'\n    E[p+'attention.self.query.weight']=np.zeros((768,768),np.float32)\n    E[p+'attention.self.query.bias']=np.zeros(768,np.float32)\n    E[p+'attention.self.value.weight']=np.eye(768,dtype=np.float32)\n    E[p+'attention.self.value.bias']=np.zeros(768,np.float32)\n    E[p+'attention.output.dense.weight']=(c*np.eye(768)).astype(np.float32)\n    E[p+'attention.output.dense.bias']=np.zeros(768,np.float32)\n    E[p+'attention.output.LayerNorm.weight']=np.ones(768,np.float32)\n    E[p+'attention.output.LayerNorm.bias']=np.zeros(768,np.float32)\n    E[p+'output.dense.weight']=np.zeros((768,3072),np.float32)\n    E[p+'output.dense.bias']=np.zeros(768,np.float32)\n    E[p+'output.LayerNorm.weight']=np.ones(768,np.float32)\n    E[p+'output.LayerNorm.bias']=np.zeros(768,np.float32)\n    if save:\n        for n,a in E.items(): save_edit(n, a.astype(np.float32))\n    return E, N\n'''\nopen('builder.py','w').write(builder)\nexec(builder)\nexec(open('helpers.py').read())\nA = np.load('A_proj.npy'); D = np.load('D_dims.npy')\nm0 = load('mean',0); center = (m0@A).mean(0)\nnp.save('center.npy', center)\nE,N = build(np.ones(12), np.ones(12), s=0.1, A=A, D=D, center=center)\nprint('stolen neurons', N, 'n edits', len(edits()))\n"
}
```

