#!/usr/bin/env python
"""Conventional compression baselines -> runs/_baselines/reins/<name>/. Targets come from configs/reins.yaml.

  --suite                                  naive drops, ft_base_L2/3/4 (3 ep and 10 ep), ft_task_L3/4 (10 ep)
  --finetune --layers 3 [--ffn 1024] [--source base|task] [--epochs 10 --lr 5e-5]
  --naive-drop --keep 0,1,2 [--ffn 1024]
  --matched 3,1024 [--source base|task]    truncate+FFN-shrink+fine-tune at a given (k, m)
"""
import argparse
import yaml
from wspr.arch import ROOT
from wspr.eval import baselines as B

p = argparse.ArgumentParser()
p.add_argument("--suite", action="store_true")
p.add_argument("--finetune", action="store_true")
p.add_argument("--naive-drop", action="store_true")
p.add_argument("--layers", type=int)
p.add_argument("--ffn", type=int)
p.add_argument("--keep", type=str)
p.add_argument("--matched", type=str, help="k,m")
p.add_argument("--source", choices=["base", "task"], default="base")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--name", type=str)
a = p.parse_args()
cfg = yaml.safe_load((ROOT / "configs" / "reins.yaml").read_text())
targets = (cfg.get("accuracy_target"), cfg.get("macro_f1_target"))
if a.suite:
    B.run_suite(targets)
if a.finetune:
    B.baseline_finetune_trunc(a.layers, a.ffn, source=a.source, epochs=a.epochs, lr=a.lr, name=a.name, targets=targets)
if a.naive_drop:
    B.baseline_naive_drop([int(x) for x in a.keep.split(",")], m=a.ffn, targets=targets)
if a.matched:
    k, m = (int(x) for x in a.matched.split(","))
    B.baseline_finetune_trunc(k, m, source=a.source, epochs=a.epochs, lr=a.lr, name=a.name, targets=targets)
