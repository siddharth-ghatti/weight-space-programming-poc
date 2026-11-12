"""Constants shared by the distillation experiment: paths, the task prompt, answer words, guard texts."""
from __future__ import annotations

import re
from pathlib import Path

from wsp.labels import DESCRIPTIONS, LABELS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "qwen3.5-0.8b"
CACHE_DIR = ROOT / "models" / "forgetting"          # WikiText-2 / ARC-Easy parquet (ignored by git)
RUNS = ROOT / "runs" / "d1"
DATA_DIR = ROOT / "data"

# One answer word per label, single tokens (with a leading space) in the Qwen3.5 tokenizer.
WORDS: dict[str, str] = {
    "coverage_grant": "coverage", "exclusion": "exclusion", "premium_payment": "premium",
    "deductible_limits": "deductible", "claims_procedure": "claims", "definitions": "definitions",
    "term_cancellation_renewal": "cancellation", "conditions_duties": "conditions",
}
ANSWER_WORDS: list[str] = [WORDS[l] for l in LABELS]           # index-aligned with wsp.labels.LABELS
LETTERS: list[str] = list("ABCDEFGH")
K = len(LABELS)

# Qwen3.5 (chat-tuned) writes an empty thinking block before it answers, even in plain completion.
# The prompt ends with it so the edit is solved at the token where generation actually writes the answer.
THINK = "<think>\n\n</think>\n\n"
QUESTION = "Which type of insurance clause is this sentence?"
# Short glosses: the reader runs at about 800 tokens/s on the laptop, so the prompt length is the whole cost.
GLOSS: dict[str, str] = {
    "coverage": "what the policy covers or insures", "exclusion": "what is not covered",
    "premium": "premium amounts, due dates, payment", "deductible": "deductibles, co-payments, limits, caps",
    "claims": "how to file or process a claim", "definitions": "defines a term used in the policy",
    "cancellation": "policy period, cancellation, renewal", "conditions": "other duties and conditions of either party",
}


def prompt(text: str, variant: str = "words") -> str:
    if variant == "full":          # the long form: one line per option with the label's full description (2.4x the tokens)
        opts = "\n".join(f"{ANSWER_WORDS[i]}: {DESCRIPTIONS[l]}" for i, l in enumerate(LABELS))
        return f'Sentence: "{text}"\n{QUESTION}\nOptions:\n{opts}\nReply with one option.\n{THINK}Answer:'
    if variant == "words":
        opts = "; ".join(f"{w} ({GLOSS[w]})" for w in ANSWER_WORDS)
        return f'Sentence: "{text}"\n{QUESTION}\nOptions: {opts}.\nReply with one option.\n{THINK}Answer:'
    opts = "; ".join(f"{LETTERS[i]} = {w} ({GLOSS[w]})" for i, w in enumerate(ANSWER_WORDS))
    return f'Sentence: "{text}"\n{QUESTION}\nOptions: {opts}.\nReply with the letter.\n{THINK}Answer:'


def parse_word(text: str, variant: str = "words") -> int | None:
    variant = "words" if variant == "full" else variant
    """The answer a generation starts with, after any thinking block is removed; prefix match, longest word first,
    so "claimsclaims" reads as claims but "conditional" does not read as conditions."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    text = re.sub(r"</?think>", " ", text)
    if variant == "letters":
        t = re.sub(r"^[^A-Za-z]+", "", text)
        t = re.sub(r"^(?i:final answer|answer)\s*:\s*", "", t)
        return LETTERS.index(t[0].upper()) if t and t[0].upper() in LETTERS and (len(t) == 1 or not t[1].isalpha()) else None
    t = re.sub(r"^[^a-z]+", "", text.lower())
    t = re.sub(r"^(final answer|answer)\s*:\s*", "", t)
    for w in sorted(ANSWER_WORDS, key=len, reverse=True):
        if t.startswith(w):
            rest = t[len(w):]
            if not rest or not rest[0].isalpha() or rest.startswith(w):
                return ANSWER_WORDS.index(w)
    return None


_ANSWER_RE = re.compile(r"(?<![a-z])(" + "|".join(ANSWER_WORDS) + r")(?![a-z])|(" + "|".join(ANSWER_WORDS) + r")(?=\2)")


def answer_words(text: str) -> int:
    """How many answer words a free generation holds, including ones run together."""
    return len(_ANSWER_RE.findall(text.lower()))


def repeat_run(text: str) -> int:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    best = run = 0
    for i, l in enumerate(lines):
        run = run + 1 if i and l == lines[i - 1] else 1
        best = max(best, run)
    return best


# Knowledge the edit must not disturb: capital cities, each a single token after a space.
CAPITALS = [
    ("Paris", "France"), ("Berlin", "Germany"), ("Madrid", "Spain"), ("Rome", "Italy"), ("Lisbon", "Portugal"),
    ("Tokyo", "Japan"), ("Beijing", "China"), ("Moscow", "Russia"), ("Cairo", "Egypt"), ("Athens", "Greece"),
    ("Vienna", "Austria"), ("Dublin", "Ireland"), ("Oslo", "Norway"), ("Stockholm", "Sweden"), ("Helsinki", "Finland"),
    ("Warsaw", "Poland"), ("Budapest", "Hungary"), ("Ottawa", "Canada"), ("Canberra", "Australia"), ("Nairobi", "Kenya"),
    ("Lima", "Peru"), ("Santiago", "Chile"), ("Bangkok", "Thailand"), ("Manila", "Philippines"), ("Jakarta", "Indonesia"),
    ("Tehran", "Iran"), ("Baghdad", "Iraq"), ("Ankara", "Turkey"), ("Havana", "Cuba"), ("Caracas", "Venezuela"),
    ("Copenhagen", "Denmark"), ("Brussels", "Belgium"), ("Amsterdam", "Netherlands"), ("Bern", "Switzerland"),
    ("Kabul", "Afghanistan"), ("Islamabad", "Pakistan"), ("Delhi", "India"), ("Tunis", "Tunisia"), ("Damascus", "Syria"),
    ("Beirut", "Lebanon"),
]
CONTROL_TEMPLATE = "The capital of {c} is the city of"

# The guard: ordinary text the edit is solved to leave alone (its keys enter the protection covariance).
GUARD = [
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch.",
    "Water boils at one hundred degrees Celsius at sea level, but at lower temperatures on mountains.",
    "To make a cup of tea, boil the water, add the leaves, and let them steep for three minutes.",
    "The stock market fell sharply on Monday after the central bank raised interest rates again.",
    "She opened the window, looked at the rain, and decided to stay inside with a good book.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "The train to the airport leaves every twenty minutes from the second platform.",
    "Regular exercise, enough sleep and a balanced diet are good for long-term health.",
    "Please find attached the invoice for last month. Let me know if anything looks wrong.",
    "The museum is open from nine to five on weekdays and closed on public holidays.",
]
# The forgetting test: different ordinary text, never seen by the guard.
HOLDOUT = [
    "In 1969, astronauts landed on the Moon and walked on its surface for the first time.",
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar and oxygen.",
    "Shakespeare wrote many plays, including Hamlet, Macbeth and A Midsummer Night's Dream.",
    "The meeting was moved to Thursday afternoon because two of the managers were travelling.",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
    "Mount Everest is the highest mountain above sea level, on the border of Nepal and China.",
    "He forgot his umbrella at the station and had to walk home in the pouring rain.",
    "A balanced budget means the government spends no more than it collects in taxes.",
    "The recipe calls for two cups of flour, one egg, a pinch of salt and some warm milk.",
    "Light travels at about three hundred thousand kilometres per second in a vacuum.",
]
FREE_PROMPTS = ["The three primary colours are", "Here is a short poem about the sea:\n", "def fibonacci(n):\n",
                "My order hasn't arrived yet, so I"]
GENERAL_PROMPTS = [
    "Write a short email to a colleague about moving our meeting to Friday.\n",
    "Explain why the sky is blue in two sentences.\n", "List three tips for sleeping better:\n",
    "Translate to French: Good morning, how are you?\n", "Q: What is 17 times 23?\nA:",
    "Once upon a time, in a small village by the sea,", "The main causes of the French Revolution were",
    "Summarise: The council approved a new park near the river after months of debate.\n",
]
CHAT_PROMPTS = ["What is a good name for a cat?", "Can you help me plan a two-day trip to Kerala?",
                "Is it safe to eat raw eggs?"]


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error of top-1 confidence; p [n, k] probabilities (torch), y [n] gold."""
    import torch
    conf, pred = p.max(-1)
    right = (pred == y).float()
    e = torch.zeros((), device=p.device)
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - right[m].mean()).abs()
    return e.item()


def selective_accuracy(p, y, coverage: float = 0.8) -> dict:
    """Accuracy on the most confident `coverage` fraction (confidence = 1 - H/log K), plus the threshold."""
    import math
    import torch
    conf = 1 - (-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1) / math.log(p.shape[1]))
    n = max(1, int(round(coverage * len(y))))
    idx = conf.argsort(descending=True)[:n]
    acc = (p[idx].argmax(-1) == y[idx]).float().mean().item()
    return {"coverage": n / len(y), "accuracy": acc, "threshold": conf[idx[-1]].item()}
