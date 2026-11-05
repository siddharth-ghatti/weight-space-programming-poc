## reasoning summary

Head correction only gives a small gain, so the remaining errors likely come from upstream noise that a linear map on the h6 CLS can't recover. Better per-layer approximations look like the real path to higher fidelity. Looking at the CV relative errors, the early layers (L0 through L2) dominate at both m=1536 and m=2048, while L3 onward are already small.

Checking whether layers L0 through L2 are the real bottleneck, since L3 onward already shows small error. Making them the target means finding a way to spend capacity there without changing the uniform unit count, so I'm looking at whether the linear part of an FFN could be folded into the layer's own normalization.

I'm weighing whether to pick a model size by balancing parameter count against the risk of failing, since failure would likely outweigh any ranking gain. I'd want roughly an 85-90% chance of passing before settling on one.

I'm estimating the flip rate by modeling it as the chance that the perturbation exceeds the margin, using the low-margin items to fit the shift. Test margins are the open question, so I'm falling back on a rough prior: a distilroberta trained on about 1000 examples at roughly 77.5% accuracy, where most errors sit at modest margins and most correct items are confident. Under that assumption, I'm trying to guess what fraction of test items fall below a margin of zero.

I'm estimating the flip rate from the practice data: about 40% of items under a margin of 0.5 flipped at m=1536, and about 26% at m=2048. Applying those rates to an assumed 10% of test items in that range gives a rough count of flips, which I still need to convert into a net accuracy change.

I'm estimating the chance the edited model drops below the pass threshold by treating each flip as roughly a coin toss between losing a correct answer and gaining one, with a small net downward drift. With around 16 flips, the net change is spread wide enough that falling past the cutoff is a real but modest risk.

Working through the test-set size and the flip-risk estimate, the crude numbers put the chance of falling below the pass bar somewhere around 16 to 28 percent, depending on how many flips land. Next I'm weighing whether a bigger or smaller alternative strategy might cut the number of flips at low-margin items.

I'm weighing whether the remaining fidelity gains are worth chasing, since more probe data for the L0–L2 fits only helped a little. I'm also wondering whether an LS-aware greedy selection, which looked worse before, might still matter.

I'm considering whether sequential correction at L1 and L2 could compensate for upstream errors by refitting each layer's output against the base model given the already-edited inputs. The earlier L5 test suggests that approach doesn't recover much, since the corrected target matched the unpruned FFN on the edited input, leaving the upstream error in place.

Dropping a layer and refitting the head has already been tried and gave poor agreement, so I'm sticking with six layers, which means the remaining size choice has to come from the m values.

I'm settling on m as the remaining lever, since the uniform unit count fixes the layout. If I spend a few more probe passes refining the L0 and L1 fits and then rebuild at m=1536 or 1792, I can check whether the extra fidelity justifies the size.

I'm checking whether the per-layer error at L1 actually translates into downstream harm, comparing mean-hidden and CLS agreement between the two probe passes. Those numbers look close, and the flip counts may be partly noise. I'm also weighing a middle size, m=1792, as a compromise.

I'm considering whether to keep the current units but adjust their input weights, since the pruned units' contributions might be better captured that way. That would be nonlinear and iterative, though, so I'm looking at closed-form alternatives instead, like merging a pruned unit into a kept unit with a similar input row.

I'm wondering whether greedy forward selection with least-squares refitting, similar to OMP, would pick better units for L0 through L2 than the earlier OBS-style scoring did. Then I want to step back and see how much each layer contributes to the final logits, perhaps by building hybrid models, though the uniform unit count rules out mixing full and pruned layers directly.

I'm weighing m=2048 against m=1536 on the practice flip counts, 15 versus 22, and both carry some test risk. My rough estimate puts m=2048 still well above zero risk, so I'm questioning whether that estimate is too pessimistic.

I'm wondering how the margin distribution on the test set compares to the practice set. The practice model is at 92.5% accuracy against 77.5% on test, so the test set has roughly three times the error rate, and the low-margin share there could be quite different from what I'm seeing in practice.

I'm looking at how much the margin spread would need to shrink to bring the flip risk down, since the per-item error near zero margin looks like the main driver. The shift from m=1536 to m=2048 reduces the standard deviation of the margin change, but a rough estimate still leaves the pass risk above zero. I'm weighing whether the L0 input could be modeled more directly to cut the information lost in those early layers.

I'm wondering whether the L0 FFN's dependence on token identity could be captured more directly, since the embeddings are frozen and the self-attention term isn't something I can control. I'm also considering whether folding the linear part of the layer-0 FFN into the attention output bias could reduce the error there.

Checking whether a non-diagonal linear term in the layer-0 FFN could be folded into the surrounding normalization doesn't work, since LayerNorm only absorbs diagonal scaling. I'm going to stop going down that path and think about whether my flip-risk estimate is too pessimistic.

I'm going to check the practice set's low-margin items directly to see whether the flips skew heavily toward right-to-wrong, beyond what the base accuracy alone would predict. If they do, pruning may be pulling the model back toward its pre-fine-tuning behavior, which would matter more than the raw flip-count estimate.



Head correction adds little (0.978 to 0.981 at best), since the leftover error is upstream noise that h6 can't linearly recover. Next I'll check whether flips on practice hurt more often than low-margin base accuracy would predict, to see if pruning undoes fine-tuning.



## tool_use: run_python

```json
{
 "code": "\nimport sys; sys.path.insert(0,str(WORK))\nfrom evalh import *\npb=Lb.argmax(1); srt=np.sort(Lb,1); marg=srt[:,-1]-srt[:,-2]; sec=np.argsort(Lb,1)[:,-2]\nfor th in [0.25,0.5,1.0]:\n    s=marg<th\n    print(th, 'n',s.sum(),'base acc',(pb==lab1)[s].mean(),'second-correct',(sec==lab1)[s].mean())\nprint('margins of base errors', np.quantile(marg[pb!=lab1],[.1,.25,.5,.75,.9]))\nfor k in [12,17]:\n    L=np.load(f'pass_{k}/logits.npy'); pe=L.argmax(1)\n    s=marg<0.6\n    print(k,'among marg<0.6: flips',(pe!=pb)[s].sum(),'of',s.sum(),' P(flip)',(pe!=pb)[s].mean())\n    # flip prob by bin\n    bins=[0,0.1,0.2,0.3,0.45,0.6,1.0]\n    print('  ',[ (round((pe!=pb)[(marg>=a)&(marg<b)].mean(),2),((marg>=a)&(marg<b)).sum()) for a,b in zip(bins[:-1],bins[1:])])\n    # signed: does the edited model move toward true label more than chance? dm on true-label logit vs others\n    d=(L-L.mean(1,keepdims=True))-(Lb-Lb.mean(1,keepdims=True))\n    print('  mean change of true-label logit (centered)', d[np.arange(1000),lab1].mean(), ' of base-pred logit', d[np.arange(1000),pb].mean())\n"
}
```

