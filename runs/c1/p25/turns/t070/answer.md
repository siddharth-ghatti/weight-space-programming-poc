## reasoning summary

I've got a model with a CV score around 98.1 across five seeds, and the κ scaling doesn't affect it, so I'll settle on a=1, a linear weight of 1, and λ=0.03. Next I'm fitting on all 1496 points with the combined kernel and writing out the decision function that uses the GELU-based terms plus the offset κ.

Still writing the decision function and the network mapping...

Still working out the head values: I'm mapping each class to its reserved dimension and solving for the head vector that the permutation requires.

I'm working out the head weights for the second carry head: the value rows and bias come from the permutation-inverted, scaled B2 projection, so the reserved-dimension output matches the decision function.

Still sizing the new carry-head values: I'm choosing ε so the written s2 values land near the starting magnitude, which puts the FFN contribution at about 0.04, small next to LN2's input scale.

Still working out the output scale: the FFN contribution stays around 0.04 at most, so LN2's scale stays near 1. I'll set the decode gain G to 1/ε so the centered scores spread back out to a usable range.

Still writing the pooler weights: each class row gets G times a centered one-hot entry in its reserved column, with everything else zero.

Still writing the first-layer weights: the inert FFN units stay at zero so their GELU outputs vanish, and the kept centers get their scaled, standardized coordinates.

Still writing the first-layer weights: each unit's input weights come from its standardized center coordinates divided by the scale and the 760 factor, with the bias set to cancel the mean shift.

Checking the GELU and attention assumptions next, then the carry-head value rows for layer 2 and layer 6.

Next I'm checking the carry-head value rows for layer 6, keeping the reserved columns at zero so the carried s2 values aren't overwritten. Then I'll compare the network's logits against the decoded decision function, looking at correlation and whether the argmax agrees on the training points.

I'm checking whether the LN2 scaling holds up: the offset on the eight reserved dimensions is small, so the normalization scale stays close to one and doesn't disturb the result. Then I'll settle on λ=0.03, since the cross-validation scores are flat across that range, and move on to fitting and checking the logits on the training points.



The 98.1% CV result from the implementable GELU-plus-carried-linear model matches the idealized version, so I'm writing the full edit now, with the carry heads, pooling-slot FFN and pooler rows mapped to the reserved dimensions.



## tool_use: run_python

```json
{
 "code": "import sys, json; sys.path.insert(0,'.')\nfrom scipy.special import erf\ngelu=lambda x: 0.5*x*(1+erf(x/np.sqrt(2)))\nd=np.load('base_reads.npz'); y=d['y']; R=np.load('reserved.npy'); keep=np.setdiff1d(np.arange(768),R)\nU=np.load('U_k10.npy'); kk=np.load('kappa_k10.npz'); K2,K6=kk['K2'],kk['K6']; n=len(y)\ndef zs(X): mu=X.mean(0); sd=X.std(0)+1e-6; return (X-mu)/sd, mu, sd\nZu,mu_u,sd_u=zs(U[:,keep]); Tu=Zu@Zu.T/760\nZ2,mu2,sd2=zs(d['mean_hidden_L2'][:,keep]); Z6,mu6,sd6=zs(d['mean_hidden_L6'][:,keep])\nk2=K2/K2.mean(); k6=K6/K6.mean()\nKt=2*gelu(Tu)+np.outer(k2,k2)*(Z2@Z2.T)/760+np.outer(k6,k6)*(Z6@Z6.T)/760\nY=np.eye(8)[y]; lam=0.03\nA=np.linalg.solve(Kt+lam*np.eye(n), Y-Y.mean(0))\nF=Kt@A; print('train acc', (F.argmax(1)==y).mean(), 'f range', F.min(), F.max())\nB2=Z2.T@(k2[:,None]*A)/760; B6=Z6.T@(k6[:,None]*A)/760\nlin2=k2[:,None]*(Z2@B2); lin6=k6[:,None]*(Z6@B6); ker=2*gelu(Tu)@A\nprint('component std: lin2 %.3f lin6 %.3f ker %.3f'%(lin2.std(),lin6.std(),ker.std()), 'max|alpha| %.2f'%np.abs(A).max(), 'sum|alpha*g| max %.1f'%(np.abs(2*gelu(Tu))@np.abs(A)).max())\nprint('check decomposition', np.abs(lin2+lin6+ker-F).max())\nnp.savez('fit_k10.npz',A=A,B2=B2,B6=B6,mu_u=mu_u,sd_u=sd_u,mu2=mu2,sd2=sd2,mu6=mu6,sd6=sd6,K2m=K2.mean(),K6m=K6.mean(),F=F)"
}
```

