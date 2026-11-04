## reasoning summary

I'm weighing which layers to screen first. Layers 5, 6, and 7 show the smallest changes and high CKA, and layer 6 has the highest CKA but also contains the old carry head h11, which I'm setting aside as irrelevant. I'm starting with dropping layer 6, though I'm aware sentence-mean changes may miss token-level importance.

I'm noting that dropping a layer would also mean refitting the next layer's feed-forward block, but I'll leave that for later. For the next pass I'll set the structure to keep layers 0 through 8 and 11 minus 6, then read the cls and mean hidden states from edited layers 0 and 8 across the sample range. Before running the structure change, I want to save a copy of the current configuration.



Layers 5–7 barely change the sentence representation, with orig layer 6 the most redundant by CKA. I'm dropping it in a candidate structure without refitting, then screening its features on p0000–p1023 against the 97.88% baseline.



## tool_use: run_python

```json
{
 "code": "import shutil, json\nshutil.copy('structure.json','structure_k10_m1024_final.json')\nst=structure(); print(st['layers'], st['m'])\nfk={int(k):v for k,v in st['ffn_keep'].items()}\nlayers=[l for l in st['layers'] if l!=6]\nprint(set_structure(layers=layers, ffn_keep={l:fk[l] for l in layers}))\nprint(structure()['layers'])\n"
}
```

