## reasoning summary

The holdout results show all dimensions around 96.4 to 97.0 percent against 95.5 to 96.4 percent for the non-D variant, so the gain is roughly half a point to a point. The earlier CV numbers look inflated by leakage, and the D-only model drops from 84 percent in CV to 71 percent on holdout, which suggests A is overfitting, perhaps more so at lambda 0.4 on 1024 inputs. With only 472 holdout examples, differences of that size are hard to distinguish from noise.

I'm weighing whether refitting the A model on all 1496 examples with stronger regularization would generalize better, though I can't validate that cleanly without the holdout. A nested evaluation with the base features gave about 97% CV, which matches the holdout figure, so that's my honest estimate. Next I want to look at the holdout errors to see whether some labels are inherently ambiguous.

I'm figuring out whether to refit the A model on all 1496 examples, using the edited network's m11 features with the base m0 features, so the final fit matches what the network actually computes.

I'm considering whether the 8 ridge directions could be supplemented with 8 more dimensions, but a supervised projection at stage one is what overfits here. Unsupervised PCA dims avoid that leakage, yet PCA-64 already gave about 96%, no better than the ridge version, so the gain probably comes from somewhere else in the bag-of-embeddings representation.

Next I'm going through the error analysis to see which classes get confused with each other.



On the holdout, all dimensions reach 97.0% accuracy at λ=0.4, so the lexical channel adds about a point, and the earlier 98.7% CV was inflated by leakage. Next I'll check which classes get confused on the holdout before deciding whether to refit A.



## tool_use: run_python

```json
{
 "code": "exec(open('helpers.py').read())\nlab = labels_all(); ytr = lab[:1024]; yte = lab[1024:]\nP = [json.loads(l) for l in open('practice.jsonl')]\nXtr = load('cls',12,'pass_3'); Xte = load('cls',12,'pass_4')\nWf,b = ridge_fit(Xtr,ytr,0.4); p=(Xte@Wf+b).argmax(1)\nC = np.zeros((8,8),int)\nfor a,bb in zip(yte,p): C[a,bb]+=1\nprint(C)\nab = [l[:6] for l in LABELS]\nfor i in np.where(p!=yte)[0]:\n    print(ab[yte[i]],'->',ab[p[i]],'|',P[1024+i]['register'],'|',P[1024+i]['text'][:160])\n"
}
```

