"""wspd: install a task into a base language model by closed-form weight edits distilled from a
Jev-like teacher (calibrated decision heads on the frozen reader). Independent of wsp/, wspc/, wspr/.

Import this package before torch so the MPS fallback switch is set for ops the Metal backend lacks."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
