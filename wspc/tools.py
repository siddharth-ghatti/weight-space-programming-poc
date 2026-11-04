"""Tool definitions given to the editor model (pruning harness)."""

from __future__ import annotations

RUN_PYTHON = {
    "name": "run_python",
    "description": (
        "Run one block of Python in the sandbox (fresh process each time, persistent working folder). "
        "Returns stdout and stderr (truncated). Counts against the code-run budget. "
        "The prelude helpers (catalog, base, W, save_edit, practice, tokenize, structure, set_structure, param_count, ...) are predefined."
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
        "requests do not count). model='base' is the unpruned 12-layer start model; model='edited' is "
        "start + your saved edits, pruned by your current structure.json. kind='predict': run the model(s) on "
        "practice sentences and get accuracy, macro-F1, confusion matrix, per-item predictions/logits "
        "(pass_<k>/predictions.json) and, for the edited model, its structure and parameter count. "
        "kind='read': teacher-forced read of hidden states, pooled [CLS] output, logits and/or attention "
        "for practice ids; arrays are written as .npy to pass_<k>/ and the result lists files, shapes and "
        "(edited model) layer_map."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["predict", "read"]},
            "model": {"type": "string", "enum": ["base", "edited", "both"],
                      "description": "Which model to run. 'both' is only valid for kind='predict' (paired comparison, up to 256 items each)."},
            "n": {"type": "integer", "description": "predict: number of practice sentences to sample (1-512) when ids is not given. Default 256."},
            "seed": {"type": "integer", "description": "predict: seed for sampling practice sentences. Default 0."},
            "ids": {"type": "array", "items": {"type": "string"}, "description": "Practice ids to use (predict: up to 512; read: up to 1024, required)."},
            "layers": {"type": "array", "items": {"type": "integer"},
                       "description": "read: hidden layer indices 0..L (0 = embedding output, L = last encoder layer of the model read: 12 for base, number of kept layers for edited). Default all."},
            "outputs": {"type": "array",
                        "items": {"type": "string", "enum": ["cls_hidden", "mean_hidden", "pooled", "logits", "attention", "token_hidden"]},
                        "description": "read: which arrays to write. cls_hidden/mean_hidden: [n,768] per layer. pooled: [n,768] tanh(pooler) output fed to the classifier. logits: [n,8]. attention: [n,12,T,T] per layer (max 64 ids). token_hidden: [n,T,768] per layer (max 64 ids). Default ['cls_hidden']."},
        },
        "required": ["kind"],
    },
}

SUBMIT = {
    "name": "submit",
    "description": (
        "Finish the run. Your current saved edits and structure.json become the final pruned model, which is "
        "evaluated on a held-out test set you never see and measured for parameter count. Refused if "
        "structure.json is invalid. Include a plain-English explanation of what you changed and why."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"final_answer": {"type": "string", "description": "Explanation of the final edit and your reasoning."}},
        "required": ["final_answer"],
        "additionalProperties": False,
    },
    "strict": True,
}

TOOLS = [RUN_PYTHON, FORWARD_PASS, SUBMIT]
