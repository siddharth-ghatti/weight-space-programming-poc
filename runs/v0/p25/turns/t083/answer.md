## reasoning summary

I'm weighing whether to push the model further. The CV score is around 0.983 with 14 passes left, and I could write more training sentences, which has helped before, though the gain was within the CV noise.

I'm considering whether a richer feature set could help, since earlier-layer features like mean0 might add signal, but getting them into the layer 11 CLS would mean changing the architecture. The current kernel result sits around 0.981, and a combination of earlier-layer information could push it higher if I can route it through the residual stream without disturbing the rest.

I'm checking whether the layer 11 input is the right place to pull mean information from, since mean11 is the hidden state feeding that final layer. Editing layer 10 to bring in mean10 would touch every token and disturb the input I'm relying on, and separating the CLS token's update cleanly looks too fiddly for the likely gain. More hand-written training data may be the better use of the remaining passes.

I'm weighing whether more hand-written sentences are worth the output cost, since the last batch moved CV only slightly within noise. I'm also considering error analysis by class to see if the remaining mistakes are just label ambiguity, and whether a combined kernel-plus-linear feature might help, though earlier scans all landed in the same narrow band. The bigger worry is test-time robustness, though the test set should match the practice distribution.

I'm checking whether the final layer's output is trustworthy here: the feed-forward term with that large scaling dominates before normalization, but the kernel depends only on the representation itself, so the practice results carry over. I'm not sure the expected logits were verified on my own sentences, since the last pass only predicted on practice. With 14 passes and plenty of code runs left, I'm asking whether there's a real gain to be had or whether the cross-validation error is already close to the label-noise ceiling.

I'm going to check where the cross-validation misclassifications fall, to see whether they are ambiguous labels. If most are, the model is near the ceiling and I should submit.

I'm considering whether heads could be split between uniform averaging and sharper attention, with some keeping the pretrained patterns, though the base result there was weaker. I'm also thinking about a max-pooling style head that picks out the single token most aligned with a class direction, which might keep key phrases from being diluted in the average.

I'm considering a softmax-pooling design: eight heads, one per class, that attend over tokens by their alignment with a class direction learned from the sentence-level LDA, so key phrases aren't diluted by averaging. That would need a fresh evaluation pass to see whether it helps.

Next I'm planning how to spend the remaining budget, since a softmax-pooling head would take two or three passes per experiment to evaluate. Whether it's worth it depends on how many of the roughly 25 current errors are real misses rather than label ambiguity, so I'll look at those first.

Next I'm pulling the shared helper code into view, so I can see the kernel and feature-loading routines before I look at the misclassifications.



The v3 model checks out, with edited logits matching the expected ones to within 1e-4. I'm updating my notes next, then reviewing the honest-CV errors to see whether there's real room to improve or whether they're just ambiguous labels.



## tool_use: run_python

```json
{
 "code": "\nwith open('notes.md','a') as f: f.write(\"\\n## Handoff 3 (turn 80+)\\n- pass_11 predict (512 practice): edited logits == v3_expected_logits_practice (err 1e-4), acc 1.0. v3 VERIFIED & saved.\\n\")\nprint(open('helpers.py').read()[:4000])\nd=np.load('base_feats.npz'); print({k:d[k].shape for k in d.files})\n"
}
```

