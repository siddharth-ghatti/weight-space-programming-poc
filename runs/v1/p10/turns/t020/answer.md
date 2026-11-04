## reasoning summary

I'm planning four checks: first the magnitude of the D dimensions across layers, then how the non-D features shift between the edited and base mean hidden states at layer 11, then whether the final CLS at layer 12 is consistent with the layer-normed edited mean, and last a cross-validated ridge comparison on the layer-12 CLS using all dims, non-D dims only, and D dims only.



I'm analysing pass 3 next: the D signal's magnitude across layers, how far the protection edits disturbed the rest of the network, whether the final [CLS] matches LayerNorm(mean), and CV accuracy with and without the D dims.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); y = lab[:1024]; folds = make_folds(1024)\nD = np.load('D_dims.npy'); nonD = np.setdiff1d(np.arange(768), D)\nCp = np.eye(8)-1/8  # contrast projector\nfor l in range(13):\n    me = load('mean',l,'pass_3'); mb = load('mean',l,'pass_2')\n    dc = me[:,D]@Cp\n    rel = np.linalg.norm(me[:,nonD]-mb[:,nonD],axis=1).mean()/np.linalg.norm(mb[:,nonD],axis=1).mean()\n    print(l, 'D contrast rms %.4f'%np.sqrt((dc**2).mean()), 'D mean-dir %.4f'%me[:,D].mean(), 'nonD rel change %.4f'%rel, 'nonD rms %.3f'%np.sqrt((me[:,nonD]**2).mean()))\nc12 = load('cls',12,'pass_3')\nprint('cls12 mean/std per sample', c12.mean(1)[:3], c12.std(1)[:3])\nlams=(0.05,0.1,0.2,0.4)\nprint('all dims', ridge_cv(c12,y,lams,folds))\nprint('nonD', ridge_cv(c12[:,nonD],y,lams,folds))\nprint('D only', ridge_cv(c12[:,D],y,lams,folds))\nprint('base m11', ridge_cv(load('mean',11),y,lams,folds))\n"
}
```

