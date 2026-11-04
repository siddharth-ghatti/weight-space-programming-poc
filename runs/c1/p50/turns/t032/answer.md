## reasoning summary

I'm noting that GELU kernels aren't positive semidefinite, so kernel ridge regression with a small λ breaks down badly; the 97.53% result at λ=0.3 was a sign of that. The negative eigenvalues wreck the inverse unless λ is larger than the magnitude of the smallest one. I'm considering solving for α directly as a ridge problem on the features, since that stays well-posed even when K isn't PSD.

I'm working through a way to make the kernel positive semidefinite by splitting GELU into a linear part and a remainder, since the linear piece is PSD. For the left tail, GELU behaves like a Gaussian rather than an exponential, but with the right scale and offset the exponent becomes roughly linear in s, so I'm checking whether that tiny-magnitude tail stays within float32 precision.

Checking whether torch's erf-based GELU loses precision in the left tail: at x = -5, the 1 + erf term is only around 5.7e-7, so float32 cancellation would make it roughly 10% off. I'm wondering whether torch uses an erfc form internally to avoid that.

I'm checking whether splitting GELU into pairs at t = a·s gives a usable kernel: the sum GELU(as) + GELU(-as) behaves like a quadratic in s for small values, and that quadratic term is positive semidefinite. Combining it with the linear part could approximate the first terms of an exponential kernel.

I'm working through the unit count for the even part: each anchor needs its own pair of GELU units, so 1496 anchors come to about 2992 units, and at m=3000 that is roughly 4.6M per layer. I'm now looking at whether a single unit could carry the even part instead.

I'm now looking at the GELU linear-plus-even decomposition: the linear part is just a linear term that can be absorbed, and the even part behaves like a quadratic for small inputs but grows like |t|/2 for larger ones, so I need to check how far the s range keeps it in the quadratic regime.

I'm checking whether a larger scale, like gelu(4s), still gives a usable kernel. Near s = 0 the GELU values sit close to a linear function, so the matrix looks roughly like a multiple of the identity plus a small perturbation, and I'm wondering why that would produce such bad negative eigenvalues at every λ up to 0.3.

I'm wondering why the kernel fails so badly if the diagonal dominates, so I'm going to run a quick debug of the kernel smoother with GELU(4s) and see what the predictions look like.

I'm checking why the kernel fails at small λ: the GELU kernel values should be well-behaved here, so I'll run a quick debug of the smoother with K = gelu(4S) directly, to see whether the negative eigenvalues are actually the cause.

I'm now seeing that the even part of GELU, like a |s| kernel, isn't positive semidefinite, so with a few thousand entries the negative eigenvalues can get large. That explains why rather than fixing the kernel, I'd better switch to a feature-based ridge, using the kernel columns as features.

I'm going to test feature-based ridge with GELU features at a few scales and offsets, since that stays well-posed when the kernel isn't PSD. I'm also weighing the paired-unit construction, which gives a PSD quadratic kernel per anchor and would fit the 3072-unit pooling layer for about 1496 anchors. Then I'll compare the two on accuracy and parameter cost.

I'm now going to test a polynomial kernel of the form 1 + s + c·s², along with the exact pair-GELU kernel built from linear and GELU(as) + GELU(-as) terms, to see whether either avoids the indefinite-kernel problem.



Single-unit GELU kernels aren't positive semi-definite, so exact kernel ridge fails at small λ. I'll test ridge on the anchor features, and a unit-pairing scheme that yields a PSD quadratic kernel plus an exact linear term.



## tool_use: run_python

```json
{
 "code": "import sys; sys.path.insert(0,'.'); from lib import *\nfrom scipy.special import erf\ngelu=lambda t: 0.5*t*(1+erf(t/np.sqrt(2)))\nd=load_base(); y=labels()\nM=lambda l: d[f'mean_hidden_L{l}'][:,KEEP]\ndef cosnorm(X):\n    Z=X-X.mean(0); Z=Z-Z.mean(1,keepdims=True); return Z/np.linalg.norm(Z,axis=1,keepdims=True)\nZ=cosnorm(M(0)); S=Z@Z.T\nL=(1e-3,0.01,0.03,0.1,0.3,1)\nfor a in [1,2,3,4]:\n    Kp=gelu(a*S)+gelu(-a*S)\n    print('pair-even a',a, krr_cv(S+ Kp/ (a*a),y,L,k=10,seeds=(0,)))\nprint('poly2', krr_cv(1+S+S**2/2,y,L,k=10,seeds=(0,)))\nprint('poly3', krr_cv(1+S+S**2/2+S**3/6,y,L,k=10,seeds=(0,)))\nprint('S+S^2', krr_cv(S+S**2,y,L,k=10,seeds=(0,)))\nprint('S+2S^2', krr_cv(S+2*S**2,y,L,k=10,seeds=(0,)))"
}
```

