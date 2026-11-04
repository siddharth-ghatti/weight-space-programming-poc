## reasoning summary

The mix settles on wm=0.3 and lam=0.005, since that combination gave a consistent +0.25%. I'm weighing whether adding around 500 more sentences would pay off, expecting roughly +0.15%. I'm also considering practice-style sentences for the confusable pairs, like premium versus term, claims versus conditions, and coverage versus exclusion.

I'm going to build v3 now so a stronger model is saved, then verify it before deciding on a second round of sentences. To do that I need to reconstruct how v2 was made: the layer-11 attention is set so the query is zero, giving uniform attention, with the value weight as identity and the output weight as 2I, and the FFN neurons carry the scaled, centred sentence patterns while the unused neurons are zeroed.

I'm checking the stored kernel file to see what b0 is, along with the centred mean, scale, and the predicted outputs F. The FFN output works out to a scaled projection onto the low-variance directions minus a mean correction, so the LayerNorm input stays close to n plus the kernel logits.

I'm checking the bias term in the kernel file, b0, which comes from the difference between the target mean and the projected training mean, and I need to confirm the LayerNorm input stays close to the normalized sentence vector.

Checking the layer-11 weights next: I want to confirm the first 1496 FFN rows match the scaled, centred sentence patterns and that the remaining rows are zeroed. Then I'll build v3 with the 2039 centres, using mu and sd from the practice set only.

I'm verifying the kernel_v3 build by checking the first-layer FFN weights against the scaled, centred practice sentences, then simulating the expected logits from the kernel outputs.



## message

The gain is consistent at about +0.25%: weight 0.3 and lambda 0.005 give 0.983 vs 0.9805. Before rebuilding, I'll inspect the saved v2 weights to confirm exactly how they're structured.

## tool_use: run_python

```json
{
 "code": "k=np.load('kernel_v2.npz'); U=np.load('U_v2.npy')\nW1=W('bert.encoder.layer.11.intermediate.dense.weight'); b1=W('bert.encoder.layer.11.intermediate.dense.bias')\nW2=W('bert.encoder.layer.11.output.dense.weight'); b2=W('bert.encoder.layer.11.output.dense.bias')\nZ=k['Z']; mu=k['mu']; sd=k['sd']\nprint(np.abs(W1[:1496]-Z/sd/768).max(), np.abs(W1[1496:]).max(), np.abs(b1[:1496]+W1[:1496]@mu).max(), np.abs(b1[1496:]).max())\nprint(np.abs(W2[:,:1496]-100*U@k['Wf'].T).max(), np.abs(W2[:,1496:]).max())\nprint(np.abs(b2 - U@(100*k['b0'] - U.T@k['n'].mean(0))).max())\nfor nm in ['attention.self.query.weight','attention.self.query.bias','attention.self.value.bias','attention.output.dense.bias','attention.output.LayerNorm.bias','output.LayerNorm.bias']:\n    print(nm, np.abs(W('bert.encoder.layer.11.'+nm)).max())\nVw=W('bert.encoder.layer.11.attention.self.value.weight'); Ow=W('bert.encoder.layer.11.attention.output.dense.weight')\nprint(np.abs(Vw-np.eye(768)).max(), np.abs(Ow-2*np.eye(768)).max())\nprint(W('bert.encoder.layer.11.attention.output.LayerNorm.weight')[:3], W('bert.encoder.layer.11.output.LayerNorm.weight')[:3])\nPw=W('bert.pooler.dense.weight'); print(np.abs(Pw[:8]-0.05*U.T).max(), np.abs(Pw[8:]).max(), np.abs(W('bert.pooler.dense.bias')).max())\nC=W('classifier.weight'); print(C[:, :9])\nprint(U.shape, np.abs(U.T@U-np.eye(8)).max(), np.abs(U.sum(0)).max())\nprint(k['n'].dtype, Z.dtype, k['Wf'].dtype)\n"
}
```

