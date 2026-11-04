## reasoning summary

The random tanh features at 768 dimensions don't reach the RBF kernel's accuracy (around 0.963 at best), while the RBF kernel ridge effectively uses 1200 basis functions, one per training sample. Since kernel ridge with an RBF kernel is a weighted sum of kernel evaluations against the training points, I could build that directly into the network, treating the training samples themselves as the neurons.

I'm checking whether a GELU neuron can approximate the exponential of a dot product the way an RBF kernel does. A single GELU behaves like a ReLU for large inputs, so it doesn't give an exponential directly. Softmax attention does compute exponentials of dot products, but it normalizes over tokens in the sentence, not over training samples, unless the training points are placed into the weights.

I'm working out whether the FFN can act as a kernel machine by placing neurons at the training samples and using GELU of dot products with them as basis functions. Then I'd fit the output weights by ridge regression on that design matrix, and the question is whether computing those activations on the training data still counts as legitimate analysis of the model.

I'm settling on computing the first-layer activations on the training samples as a kernel matrix, treating that as closed-form math on activations I've already read, and I'll check the end-to-end mapping with the tool afterward. The remaining question is whether the layer-12 CLS state I read with the FFN switched off matches the FFN input exactly.

I'm checking whether the 8 kernel scores can be written into dedicated directions of the FFN output, so the pooler and classifier can read them out. The LayerNorm scale is a positive per-sample factor, so the argmax should survive it as long as the classifier has no bias and only reads those directions.

I'm weighing which kernel would be best to implement with GELU neurons, given that the training samples would serve as the kernel centers. Kernel ridge cross-validation already suggests roughly 0.982, so the real check is whether the network's logits match my numpy predictions on practice data.

Next I'm weighing a second option: a GELU basis over the cosine similarity to each training sample, shifted by a threshold, to use as the basis functions for a ridge fit.

I'm now checking a polynomial-style basis: for small pre-activations GELU is roughly quadratic, so a GELU of a scaled similarity to each training sample could approximate a quadratic kernel. I'll test candidate scalings and thresholds offline with cross-validated ridge fits to see which basis holds up.

I'm testing a few kernel choices offline with numpy: the RBF as a reference, a GELU-threshold basis, and a quadratic one. The next step is checking whether a wide-bandwidth RBF, where the kernel values sit around exp(-0.5), holds up.

Checking whether a wide RBF, where kernel values sit around 0.6 on typical pairs, behaves like a low-order polynomial expansion. If so, a quadratic basis might capture most of the signal, so I'll run those tests now.



Random tanh features top out at 96.3%, so I'm switching the last layer's FFN to an explicit kernel machine with one GELU neuron per training example and closed-form ridge output weights. Before building it, I'll check offline which GELU-implementable basis works, using honest CV.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nD=np.load('base_feats.npz'); y=D['y']; folds=np.load('folds.npy')\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nx=c11+2*m11; xc=x-x.mean(1,keepdims=True); n=xc/xc.std(1,keepdims=True)\nmu=n.mean(0); sd=n.std(0)+1e-6; Z=(n-mu)/sd; d=768\nT=np.eye(8)[y]\nfrom math import erf\nfrom scipy.special import erf as serf\ngelu=lambda u: 0.5*u*(1+serf(u/np.sqrt(2)))\nG=Z@Z.T/d   # normalized dot products\nprint('G diag mean',np.diag(G).mean(),'offdiag mean',G[~np.eye(len(y),dtype=bool)].mean(), 'std', G[~np.eye(len(y),dtype=bool)].std())\ndef basis_cv(Phi_fn, lam, lin=False):\n    pred=np.zeros(len(y),int)\n    for f in range(5):\n        tr=folds!=f\n        P=Phi_fn(G[:,tr])      # features: activation of neurons centered at training samples\n        if lin: P=np.hstack([P,Z])\n        Ptr=P[tr]; pm=Ptr.mean(0); A=Ptr-pm\n        # dual solve (n_tr x n_tr) for efficiency\n        K=A@A.T; a=np.linalg.solve(K+lam*np.eye(len(K)),T[tr]-T[tr].mean(0))\n        pred[~tr]=np.argmax((P[~tr]-pm)@(A.T@a),1)\n    return (pred==y).mean()\n# reference RBF via G: |z-z'|^2/d = Gii+Gjj-2Gij\nfor beta,theta in [(2,0),(4,1),(4,2),(8,3),(8,4)]:\n    print('gelu',beta,theta,[round(basis_cv(lambda g: gelu(beta*g-theta),l),4) for l in [0.1,1,10]],flush=True)\nprint('quad',[round(basis_cv(lambda g: g**2,l),4) for l in [0.1,1,10]])\nprint('exp2',[round(basis_cv(lambda g: np.exp(2*g),l),4) for l in [1,10,100]])"
}
```

