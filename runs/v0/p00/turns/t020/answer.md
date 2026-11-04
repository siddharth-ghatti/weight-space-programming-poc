## reasoning summary

I've checked that everything fits together: each value dimension, the output projection, the pooler, and the classifier line up, and the class ordering is consistent with how the FFN writes its scores. Next I'll save my notes, then look at improvements while staying within the code budget and skipping any further forward passes.

I'm weighing the remaining risks: a possible mismatch in my model of the HF forward pass (such as the LayerNorm epsilon), and the test set containing tokens I haven't seen. For improvements, a previous-token head could enable bigram features, which seems worth about 1% but carries real risk. Better unigram generalization and macro-F1 calibration are the other candidates.

I'm considering how the previous-token head could copy a compressed token code into some residual dimensions, perhaps by zeroing out about 64 dims in layer 0's attention LayerNorm, though that would weaken the detectors a bit. Another option is to shrink the token representation to the few hundred first tokens that matter for bigrams. Whether that head can work depends on the learned position embeddings producing a sharp peak at position p-1, which is uncertain because the layer 1 input mixes position and token information through LayerNorm.

Now I'm working out how the position signal can be pulled out of the layer-1 input, using a least-squares decoder to map position into a Fourier-style code so the query and key projections can lock onto position p-1.

I'm working out whether a Fourier-based position decoder could give the query and key projections a sharp peak at p-1, which would let the attention pick out the previous token. The main worry is that token variation adds noise to the decoder, so I'd have to check how much accuracy it keeps before trusting it with zero forward passes.

I'm checking whether position is linearly decodable from the layer-0 hidden states, since a previous-token head depends on a clean position signal. If a simple decoder recovers position with very high accuracy, the bigram idea is worth pursuing; otherwise I'll drop it and keep the current design.

I'm checking the last few details: the score tables and the implementation differ only by small errors, which is fine. Then I'm considering the edge case where a sentence has only unknown tokens, so all scores are near zero and the argmax falls to class 0.



Every component checks out, with detector outputs within about 1–3% of the targets. I'm recording the current state in notes.md before looking at improvements.



## tool_use: run_python

```json
{
 "code": "notes = open('notes.md').read()\nnotes += \"\"\"\n## Status (after run 19) -- v1 BUILT AND SAVED (108 edits) via build.py:build_all(base, save_edit)\n- DF = [145,175,529,121,216,654,159,541] (class c score lives in dim DF[c]).\n- Target token-score table S_final.npy = NB(alpha .3, centered)/4 + ridge-TF(lam 3e-3, centered), fit on all practice.\n  Data-only 5-fold CV of this table (sum over tokens, argmax): ~97.3% (NB alone ~96.0-96.6, ridge ~97.1).\n- Layer0 FFN: 3072 detector neurons (Kn.npy tokens, top by count*||S||), U.npy whitened LDA detectors\n  (C=Cb+3Cw+.05I), beta=10, theta=0.6; outputs O.npy fit by weighted ridge over (token,position) samples.\n  Token-level check: outputs match 0.3*S within ~1-3%, specials (CLS/SEP/PAD/UNK) -> 0.\n- a=0.3 (residual scale), Gain=4 (layer1 Wo), kappa=0.5 (pooler). Classifier identity on 8 dims, bias 0.\n- Component checks passed (no-op layers, routing).\n- Files: lib.py (Emb.h0, ln, gelu, load_data), build.py, Cb/Cw/mu/K/Kn/U/O/S_final .npy\n## Ideas not done yet\n- bigrams (+~1% CV) need previous-token head (position-based attention) -- risky without forward passes.\n\"\"\"\nopen('notes.md','w').write(notes)\nprint('ok')"
}
```

