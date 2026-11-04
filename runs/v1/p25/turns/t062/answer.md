## reasoning summary

Scaling the stash signal by 4 to bring its magnitude near 0.5 and adding it onto the CLS state. Next, I'm zeroing out the stash-dimension reads from each layer's input for Q, K, and V across layers 0 through 8, folding the mean contribution into the biases.

For layer 0, the input is still the original embedding, so attention reads stay unchanged apart from head 0's Q and its V rows, while the FFN's W1 reads of the stash dimensions get ablated. From layer 1 onward, Q, K, V and W1 reads of those dimensions get zeroed, and the writes to them (Wo, W2, bo, b2) are zeroed as well, with a special case for layer 0's Wo.

Now I'm settling the gain for the stash-dimension LayerNorm weights and zeroing their biases, then folding the mean contributions of the head-0 value dimensions into the output bias for the non-stash rows.

I'm checking that the mean of the hidden states at layer l is the input to that layer, which holds since hidden_states[0] is the embeddings. Next I'm working out a gain for the stash-dimension LayerNorm weights so the stash scale stays stable, and I'm considering whether the per-LayerNorm standard deviation can be estimated from the data I already have.

I'm working out how to set the LayerNorm gain for the stash dimensions, and whether the per-layer standard deviation can be estimated from the data I already have. The distribution of each normalized value depends on the pre-LN spread, which I can't recover directly, so I'm looking at the ratio of new to old values as a stand-in.

I'm trying to pin down the LayerNorm gain for the stash dimensions, and whether the per-layer standard deviation can be estimated from the hidden-state norms I already have. Those norms only give the RMS of the normalized output, not the pre-LN spread, so I'll measure the gain directly instead of trying to derive it.

I'm weighing whether to set the stash LayerNorm gain to 1 or to the median original gain, since with a per-LN standard deviation around 0.6 to 1.0, gain 1 could compound across the 18 LayerNorms. I'll take the median of the non-outlier gains per layer and run a diagnostic pass on a small sample first to check whether the CLS stash dimensions drift across layers 1 to 9.

Checking the remaining plan: a small diagnostic pass on 256 texts to see whether the network stays close to the original in non-stash dims, then a full two-pass read. I'll first look at the γ values and then save the edit-construction code to stash.py so it can be reproduced.



I'm scaling the stash content by 4 so its magnitude matches typical residuals, then writing the stash edit as a reusable `stash.py` script. Next I'll check the LN γ statistics to choose a constant for the stash dims' gains.



## tool_use: run_python

```json
{
 "code": "import numpy as np\nS=np.load('stash_cfg.npz')['S']\nfor l in range(10):\n    L=f'bert.encoder.layer.{l}.'\n    g1=base(L+'attention.output.LayerNorm.weight'); g2=base(L+'output.LayerNorm.weight')\n    print(l, 'g1 med %.3f S %s'%(np.median(g1), np.round(g1[S],2)), '| g2 med %.3f S %s'%(np.median(g2), np.round(g2[S],2)))\nF=np.load('feats.npz')\nfor l in [0,1,5,9]:\n    print('RMS hidden L',l,' cls %.3f mean-token? %.3f'%(np.sqrt((F[f'C{l}']**2).mean()), np.sqrt((F[f'M{l}']**2).mean())))\nT=np.load('tok128.npz'); print('token RMS L0 %.3f L6 %.3f L11 %.3f'%tuple(np.sqrt((T[k].astype(float)**2).mean()) for k in ['T0','T6','T11']))"
}
```

