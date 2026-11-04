## reasoning summary

The standardized approach still comes out ahead at 97.7%, so the multi-layer idea doesn't look better. I'm weighing whether it's worth pursuing by checking the "4+11" and "0+11" variants, which reach 98.1%. I'm also considering whether pooling the layer-11 tokens, with the CLS residual at that layer, could bring in early-layer information.

I'm checking whether pooling tokens at layer 11 could carry any signal, but the CV on content-token means can't be computed without token-level data. I'm weighing the layer-0 route: it needs about three or four passes and roughly ten runs, which the budget covers. The real risk is time, and the idealized gain of about 0.4% probably shrinks in practice.

I'm checking how much the other tokens rely on CLS at layer 0, by altering CLS there and measuring how much the layer-11 mean shifts. For the edit I'd need a vector u that is orthogonal to the non-CLS layer-0 states while matching the constant CLS state, which I'd estimate from roughly 2000 token vectors across 64 practice texts.

I'm working out how to choose u for the layer-0 edit: it should match the constant CLS state while staying nearly orthogonal to the other tokens' layer-0 states, so I'd solve a regularized least-squares problem and then check the residual on held-out texts. Once CLS attends uniformly at layer 0, its output carries the mean of the layer-0 states, and that information can then pass through the remaining layers.

I'm considering whether CLS at layers 1 through 10 could keep the early-layer signal intact, rather than letting it get overwritten. I'd then tune the weight on the mean at layer 11 against the CLS state, though I'm unsure whether the altered CLS will still carry useful information.

I'm going to test whether the layer-0 edit preserves the early-layer signal in CLS, using a uniform-attention setup where CLS carries the mean of layer-0 states. The open question is whether CLS's contribution at layers 1 through 10 survives well enough to matter at layer 11, and I can't tell that without running it.

I'm asking whether CLS at layer 0 could act as a keyword-weighted pooling rather than a plain mean, since its query is constant there. Uniform pooling is the most general case. The open question is how to get both signals into the final kernel, and whether a concatenation would work better than a sum given the 768-dimensional limit.

I'm trying to squeeze the 768-dim mean at layer 11 into a 384-dim subspace, scaled up so it survives the sum with the CLS residual. Then the layer-11 normalization would mix the two, and the feed-forward units would see the combined vector.

I'm considering whether the layer-11 mean could be squeezed into a 384-dimensional subspace and scaled up, so that it survives the sum with the CLS residual. The worry is that a plain per-coordinate standardization may not balance the two parts once layer normalization shrinks the CLS component.

I'm weighing whether the layer-11 mean can be compressed into a 384-dim subspace and kept separate from the CLS residual. The basis choice could let the kernel standardize the two parts separately, but the expected gain is still uncertain and probably small.

I'm weighing whether the layer-0 experiment is worth the effort, since the budget allows it even though the expected gain is small. Before running it, I need enough token vectors to estimate the second-moment matrix, and I'm checking whether about 2000 vectors from 64 texts will be sufficient.

I'm considering an exact alternative: the LayerNorm bias β gives every layer-0 state a shared linear constraint, but that doesn't isolate the CLS vector. Since the CLS state comes from its embedding and position 0, I'd only be able to separate it approximately, so I'm weighing whether the ridge fit on about 2000 vectors is good enough or whether another pass of token reads is needed.

I'm considering whether an exact trick exists on the key side instead: CLS's attention is uniform only if its query dots with every key equally, but the key variation spans all 64 dimensions of each head, so I can't zero it out by adjusting the query alone. Changing W_k or the key bias would affect every token, not just CLS, so that route doesn't isolate the effect either.

I'm recalling that the base CLS state at layer 1 already holds a fixed, keyword-weighted pooling of the layer-0 values, and its linear probe reaches 92.2%. The layer-11 CLS state appears to lose that pooled lexical signal, and combining it with the layer-11 mean didn't help either. So uniform pooling at layer 0 probably wouldn't survive to layer 11 in a useful form, and I'm dropping the multi-layer idea.

Pondering whether a layer-1 CLS probe combined with the layer-11 mean could work as an idealized concat, though that still leaves the transport problem unsolved, so I'm setting it aside. Since the CLS state can't be made specific to itself through layers 1 through 10, I'm settling on checkpoint B as the final answer. Before wrapping up, I'll run a quick robustness check on about 16 hand-written sentences to see whether the predictions look sensible.

I'm checking the padding handling now: the attention mask should exclude padded positions, so the mean over tokens stays correct in batched runs. I'll also confirm the regularization strength holds with the larger set of centers, then run a quick sanity check on a few hand-written sentences before settling on the final model.

I'm writing a handful of custom sentences that cover each class in different registers, to see if the predictions look sensible.



Whitening doesn't beat the standardized-space kernel either, so I'm ending the feature search; I'll test the edited model on 16 sentences I wrote myself to check it generalizes.



## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "outputs": [
  "logits"
 ],
 "layers": [
  12
 ],
 "texts": [
  "We will pay for sudden and accidental water damage to your kitchen caused by a burst pipe.",
  "Good news: your plan covers emergency dental treatment while you are traveling abroad.",
  "This policy does not cover losses caused by war, invasion, or nuclear hazard.",
  "Sorry, cosmetic surgery isn't something the plan will pay for.",
  "Your monthly installment of $142 is due on the first of each month.",
  "If your premium payment is more than 31 days late, a $15 late fee will be added to your next bill.",
  "You pay the first $1,000 of any covered collision loss before we pay anything.",
  "The most we will pay for jewelry theft is $2,500 per occurrence.",
  "To report a claim, call our 24-hour hotline and have your policy number ready.",
  "Please send us a signed proof of loss within 60 days of the date of the fire.",
  "\"Vehicle\" means a land motor vehicle designed for travel on public roads.",
  "In this policy, 'Dependent' refers to your spouse and unmarried children under age 26.",
  "This policy begins at 12:01 a.m. on January 1 and ends one year later.",
  "We may cancel this policy by mailing you written notice at least 10 days before cancellation takes effect.",
  "You must cooperate with us in the investigation and defense of any suit.",
  "If we pay a loss, your right to recover from anyone else becomes ours up to the amount we paid."
 ]
}
```

