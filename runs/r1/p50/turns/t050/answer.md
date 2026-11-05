## reasoning summary

I'm setting up a probe for layer 0, which has no upstream dependency, so the mapping is just the full range of m. The keep_sets file doesn't exist yet, and I need to see whether both B1 and B2 fit in one token_hidden read, which caps at 32 ids, so it'll take two passes. The input to layer 0's FFN doesn't depend on pruning anyway, so the base data is fine.



## message

Setting up the layer-0 probe (layer 0 only; LN1 and LN2 set to identity, FFN output zeroed). Reading `token_hidden` at index 1 then gives the exact normalized FFN input.

## tool_use: run_python

```json
{
 "code": "exec(open('seq.py').read())\nprint(setup_probe(0, 1024)); print(structure()['layers'], structure()['m'])"
}
```

## tool_use: forward_pass

```json
{
 "kind": "read",
 "model": "edited",
 "ids": [
  "p0473"
 ],
 "layers": [
  1
 ],
 "outputs": [
  "token_hidden"
 ]
}
```

