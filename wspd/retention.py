"""Catastrophic-forgetting checks against the unedited model, built before any edit: perplexity and KL on WikiText-2
paragraphs, ARC-Easy multiple choice scored by the probability of the answer letter (asked after the empty thinking
block, where the model answers), the capital-city questions, and answer words leaking into free generation."""
from __future__ import annotations

import math
from typing import Dict, Sequence

import torch

from wspd.common import (CAPITALS, CHAT_PROMPTS, CONTROL_TEMPLATE, FREE_PROMPTS, GENERAL_PROMPTS, THINK, answer_words,
                         repeat_run)
from wspd.reader import Student


class Retention:
    def __init__(self, student: Student, wiki_texts: Sequence[str], arc_rows: Sequence[dict], max_tokens: int = 256,
                 topk: int = 64, gen_tokens: int = 48):
        self.s, self.device, self.topk, self.gen_tokens = student, student.device, topk, gen_tokens
        tok = student.tok
        self.wiki = [torch.tensor(tok(x, add_special_tokens=False).input_ids[:max_tokens]) for x in wiki_texts]
        self.wiki = [x for x in self.wiki if len(x) > 8]
        self.arc = []
        for r in arc_rows:
            labels, texts = r["labels"], r["texts"]
            ids = [tok(l, add_special_tokens=False).input_ids for l in labels]
            if r["answer"] not in labels or not all(len(i) == 1 for i in ids):
                continue
            p = "Question: %s\n%s\nAnswer with the letter.\nAnswer:%s" % (
                r["question"], "\n".join("%s. %s" % lt for lt in zip(labels, texts)), THINK)
            self.arc.append((torch.tensor(tok(p, add_special_tokens=False).input_ids), torch.tensor([i[0] for i in ids]),
                             labels.index(r["answer"])))
        self.caps = [torch.tensor(tok(CONTROL_TEMPLATE.format(c=k), add_special_tokens=False).input_ids) for _, k in CAPITALS]
        with torch.no_grad():
            self.base = []
            for x in self.wiki:
                t = self._lp(x).topk(topk, dim=-1)
                self.base.append((t.indices, t.values))
        self.gen_prompts = list(FREE_PROMPTS) + list(GENERAL_PROMPTS) + [student.chat(c) for c in CHAT_PROMPTS]
        self.gen_base = student.generate(self.gen_prompts, gen_tokens)

    def _lp(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.s.hidden(ids[None].to(self.device), torch.ones(1, len(ids), dtype=torch.long, device=self.device))[0]
        return self.s.head(h).float().log_softmax(-1)

    def contamination(self) -> Dict:
        """A prompt counts as contaminated when its generation gains 3 or more answer words, or a run of 3 or more
        repeated lines it did not have before."""
        outs = self.s.generate(self.gen_prompts, self.gen_tokens)
        bad = []
        for p, o, b in zip(self.gen_prompts, outs, self.gen_base):
            if answer_words(o) - answer_words(b) >= 3 or (repeat_run(o) >= 3 and repeat_run(o) > repeat_run(b)):
                bad.append(p[-60:])
        return {"contamination": len(bad) / len(outs), "contaminated": bad, "free": outs}

    @torch.no_grad()
    def measure(self) -> Dict:
        nll, n, kl = 0.0, 0, 0.0
        for ids, (ti, tv) in zip(self.wiki, self.base):
            lp = self._lp(ids)
            nxt = ids[1:].to(self.device)
            nll += -lp[:-1].gather(1, nxt[:, None]).sum().item(); n += len(nxt)
            kl += (tv.exp() * (tv - lp.gather(1, ti))).sum(-1).mean().item()
        arc = sum(int(self._lp(ids)[-1, opts.to(self.device)].argmax().item() == gold) for ids, opts, gold in self.arc)
        caps = sum(int(self._lp(ids)[-1, self.s.city_ids].argmax().item() == i) for i, ids in enumerate(self.caps))
        c = self.contamination()
        return {"wiki_ppl": math.exp(nll / max(n, 1)), "wiki_kl": kl / max(len(self.wiki), 1), "wiki_n": len(self.wiki),
                "arc_acc": arc / max(len(self.arc), 1), "arc_n": len(self.arc), "capitals_acc": caps / len(self.caps),
                "contamination": c["contamination"], "contaminated": c["contaminated"], "free_texts": c["free"],
                "free_texts_before": self.gen_base}

    @staticmethod
    def forgot(before: Dict, after: Dict) -> list[str]:
        bad = []
        if after["arc_acc"] < before["arc_acc"] - 0.03:
            bad.append("ARC-Easy fell %.1f points" % (100 * (before["arc_acc"] - after["arc_acc"])))
        if after["wiki_ppl"] > before["wiki_ppl"] * 1.05:
            bad.append("WikiText-2 perplexity rose %.0f%%" % (100 * (after["wiki_ppl"] / before["wiki_ppl"] - 1)))
        if after["capitals_acc"] < before["capitals_acc"] - 0.05:
            bad.append("capitals fell")
        if after["contamination"] > before["contamination"]:
            bad.append("answer words leak into %d of %d unrelated generations" % (
                round(after["contamination"] * len(after["free_texts"])), len(after["free_texts"])))
        return bad
