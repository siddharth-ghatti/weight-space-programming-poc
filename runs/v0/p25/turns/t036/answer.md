## reasoning summary

Kernel beats linear across all three splits by roughly half a point to a point, so I'll go with β=1.0 and λ=0.01. Next I'm looking at precision.

Now I'm checking the float32 sensitivity of Wf before settling the directions. For U, I'm looking at the lowest-variance eigenvectors of n that are orthogonal to the all-ones direction, since the mean-zero layernorm output leaves that one with no variance. Test samples may deviate more along those directions than the sample eigenvalues suggest, so I'll make the scale s large enough to dominate.

I'm sizing the scale s at around 100, which should dominate the normalized n by about four times, so the small deviations along U barely move the argmax unless the top-two class gap is under about 0.01. I also need to account for the FFN neurons existing only for the 1496 training samples, with the remaining GELU neurons zeroed out. The pooler setup is still open: rows 0 through 7 get the scaled U transpose, the rest zero, and I'm working out the bias term.

I'm settling the readout: a fixed tanh on each of the eight projected coordinates with a shared bias of zero keeps the argmax unchanged, and the identity classifier then reads off the logits directly. I'm also checking whether the intercept terms from the ridge fit should be carried over into the pooler bias.

I'm finishing the bias term: folding the constant offset from the ridge fit and the mean of the residual direction into the second FFN bias, so the U-component's average contribution is cancelled out. Then I'll check how precisely that W_2 can be represented in float32.

Checking float32 error for the output weights: if the sum of absolute contributions is small relative to the output, the rounding should stay well below a tenth of a percent. I'll estimate that ratio before going further.

Checking the coefficient magnitudes and their conditioning, without running a full forward simulation.

I'm weighing the choice of λ=0.01 now: the dual ridge solve works on the centered training features, so I want to check how large the resulting weights come out.



The kernel-machine approach beats the linear readout on all three seeds, and I'll go with β=1.0, λ=0.01. Next I'm checking the coefficients' float32 conditioning before building the closed-form v2 design.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.')\nfrom helpers import *\nfrom scipy.special import erf as serf\nD=np.load('base_feats.npz'); y=D['y']\nc11=D['cls11'].astype(np.float64); m11=D['mean11'].astype(np.float64)\nT=np.eye(8)[y]\ngelu=lambda u: 0.5*u*(1+serf(u/np.sqrt(2)))\nx=c11+2*m11; xc=x-x.mean(1,keepdims=True); n=xc/np.sqrt((xc**2).mean(1,keepdims=True)+1e-12)\nmu=n.mean(0); sd=n.std(0)+1e-6; Z=(n-mu)/sd\nbeta=1.0; lam=0.01\nU_pre=beta*(Z@Z.T)/768\nPhi=gelu(U_pre)\npm=Phi.mean(0); A=Phi-pm\na=np.linalg.solve(A@A.T+lam*np.eye(len(A)),T-T.mean(0))\nWf=A.T@a                      # 1496 x 8\nb0=T.mean(0)-pm@Wf\nF=Phi@Wf+b0\nprint('train acc',(np.argmax(F,1)==y).mean())\nprint('|Wf| max',np.abs(Wf).max(),'sum|Wf| per class',np.abs(Wf).sum(0).round(1))\nprint('score range',F.min(),F.max(),'typical top-2 gap', np.median(np.sort(F,1)[:,-1]-np.sort(F,1)[:,-2]))\n# sensitivity: perturb Phi by float32-level relative noise\nnoise=Phi*np.random.default_rng(0).normal(0,1e-6,Phi.shape)\nF2=(Phi+noise)@Wf+b0; print('max score change under 1e-6 rel noise', np.abs(F2-F).max())\nnp.savez('kernel_v2.npz',mu=mu,sd=sd,Z=Z,Wf=Wf,b0=b0,F=F,n=n)"
}
```

