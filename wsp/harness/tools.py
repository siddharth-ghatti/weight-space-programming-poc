"""Tool definitions given to the editor model."""

from __future__ import annotations

RUN_PYTHON = {
    "name": "run_python",
    "description": (
        "Run one block of Python in the sandbox (fresh process each time, persistent working folder). "
        "Returns stdout and stderr (truncated). Counts against the code-run budget. "
        "The prelude helpers (catalog, base, W, save_edit, practice, tokenize, ...) are predefined."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute."}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}

FORWARD_PASS = {
    "name": "forward_pass",
    "description": (
        "Request one forward pass from the harness (costs 1 unit of the forward-pass budget; failed "
        "requests do not count). kind='predict': run the base and/or edited classifier on practice "
        "sentences and get accuracy, macro-F1, confusion matrix, and per-item predictions/logits written "
        "to pass_<k>/predictions.json. kind='read': teacher-forced read of hidden states, pooled [CLS] "
        "output, logits and/or attention for given texts or practice ids; arrays are written as .npy to "
        "pass_<k>/ in the working folder and the result lists the files and shapes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["predict", "read"]},
            "model": {
                "type": "string",
                "enum": ["base", "edited", "both"],
                "description": "Which model to run. 'both' is only valid for kind='predict' (paired comparison, up to 256 items each).",
            },
            "n": {"type": "integer", "description": "predict: number of practice sentences to sample (1-512) when ids is not given. Default 256."},
            "seed": {"type": "integer", "description": "predict: seed for sampling practice sentences. Default 0."},
            "ids": {"type": "array", "items": {"type": "string"}, "description": "Practice ids to use (predict: up to 512; read: up to 1024)."},
            "texts": {"type": "array", "items": {"type": "string"}, "description": "read: arbitrary texts (up to 1024) instead of ids."},
            "layers": {"type": "array", "items": {"type": "integer"}, "description": "read: hidden layer indices 0-12 (0 = embedding output, 12 = last encoder layer). Default all."},
            "outputs": {
                "type": "array",
                "items": {"type": "string", "enum": ["cls_hidden", "mean_hidden", "pooled", "logits", "attention", "token_hidden"]},
                "description": "read: which arrays to write. cls_hidden/mean_hidden: [n,768] per layer. pooled: [n,768] tanh(pooler) output fed to the classifier. logits: [n,8]. attention: [n,12,T,T] per layer (max 64 texts). token_hidden: [n,T,768] per layer (max 64 texts). Default ['cls_hidden'].",
            },
        },
        "required": ["kind"],
    },
}

SUBMIT = {
    "name": "submit",
    "description": (
        "Finish the run. Your current saved edits become the final model and are evaluated on a held-out "
        "test set you never see. Include a plain-English explanation of what you changed and why."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"final_answer": {"type": "string", "description": "Explanation of the final edit and your reasoning."}},
        "required": ["final_answer"],
        "additionalProperties": False,
    },
    "strict": True,
}



def build_tools(allow_free_texts: bool = False) -> list[dict]:
    """Tool list for a run. Without allow_free_texts the forward_pass schema has no `texts` field:
    reads and predictions are restricted to practice ids (rule 6, no self-authored data)."""
    import copy
    fp = copy.deepcopy(FORWARD_PASS)
    if not allow_free_texts:
        del fp["input_schema"]["properties"]["texts"]
        fp["description"] = fp["description"].replace("for given texts or practice ids", "for given practice ids")
    return [RUN_PYTHON, fp, SUBMIT]


TOOLS = build_tools(True)  # legacy: full schema
