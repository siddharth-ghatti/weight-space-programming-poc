#!/usr/bin/env python
"""Conventional compression baselines -> runs/_baselines/compress/<name>/.

  --suite                       start, two naive drops, finetune_L2/L4/L6
  --finetune-trunc --layers 4 [--ffn 768]
  --naive-drop --keep 0,1,2,10,11 [--ffn 768]
  --matched 3,256               truncate+FFN-shrink+fine-tune at a given (k, m)
"""
import argparse
from wspc.eval import baselines as B

p = argparse.ArgumentParser()
p.add_argument("--suite", action="store_true")
p.add_argument("--finetune-trunc", action="store_true")
p.add_argument("--naive-drop", action="store_true")
p.add_argument("--layers", type=int)
p.add_argument("--ffn", type=int, default=3072)
p.add_argument("--keep", type=str)
p.add_argument("--matched", type=str, help="k,m")
p.add_argument("--target", type=float, default=0.978)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--name", type=str)
a = p.parse_args()
if a.suite:
    B.run_suite(a.target)
if a.finetune_trunc:
    B.baseline_finetune_trunc(a.layers, a.ffn, epochs=a.epochs, lr=a.lr, name=a.name, accuracy_target=a.target)
if a.naive_drop:
    B.baseline_naive_drop([int(x) for x in a.keep.split(",")], m=a.ffn, accuracy_target=a.target)
if a.matched:
    k, m = (int(x) for x in a.matched.split(","))
    B.baseline_finetune_trunc(k, m, epochs=a.epochs, lr=a.lr, name=a.name, accuracy_target=a.target)
