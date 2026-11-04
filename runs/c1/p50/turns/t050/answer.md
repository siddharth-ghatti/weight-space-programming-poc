## reasoning summary

A small gamma does best, which points toward a nearly linear-plus-quadratic setup. The 384 PCA dimensions from 6 heads keep the full mean's performance, so those 6 uniform heads can carry the mean and leave 6 heads free for other pools. Next I want to find which extra features help, including whether 2-layer alternatives are worth trying, starting with 1-layer extra pools. Then I'm reconsidering a cheap option that might work better than uniform-attention mean pooling.

I'm weighing whether the uniform heads alone can carry the mean, leaving the remaining heads for class-evidence max pools. I want to pin down what kinds of features would fix the errors, so I'm going to look at which sentences the M0 kernel model gets wrong and which classes it confuses.



Using 6 uniform heads with 384 PCA dims keeps the full mean-embedding accuracy of 97.66%, freeing 6 heads for other pools. Next I'll examine the sentences the bag-of-embeddings kernel misclassifies and the class pairs they confuse, to decide what those heads should capture.



## tool_use: run_python

```json
{
 "code": "exec(open('lib.py').read())\ny=labels(); B=load_base(); P=practice()\nM0=B['mean_hidden_L0'].astype(np.float64); Xc=M0-M0.mean(0)\nZ=Xc/np.linalg.norm(Xc,axis=1,keepdims=True); K=np.exp(1*(Z@Z.T-1))\nY=np.eye(8)[y]; f=folds(y,10,0); Sc=np.zeros((1496,8))\nfor i in range(10):\n    tr=f!=i; te=f==i\n    A=np.linalg.solve(K[np.ix_(tr,tr)]+0.01*np.eye(tr.sum()), Y[tr]-Y[tr].mean(0)); Sc[te]=K[np.ix_(te,tr)]@A\npr=Sc.argmax(1); wrong=np.where(pr!=y)[0]\nprint(len(wrong))\nfrom collections import Counter\nprint(Counter((LBL[y[i]][:8],LBL[pr[i]][:8]) for i in wrong).most_common())\nfor i in wrong[:35]:\n    print(P[i]['id'], LBL[y[i]][:10], '->', LBL[pr[i]][:10], '|', P[i]['text'][:150])\nnp.save('kerM0_wrong.npy', wrong)\n"
}
```

