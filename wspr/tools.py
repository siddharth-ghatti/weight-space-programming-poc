"""Tool definitions given to the editor model (reinsurance pruning harness)."""

from __future__ import annotations

from wspr.arch import Arch


def build_tools(arch: Arch) -> list[dict]:
    L = arch.num_layers
    run_python = {
        "name": "run_python",
        "description": (
            "Run one block of Python in the sandbox (fresh process each time, persistent working folder). "
            "Returns stdout and stderr (truncated). Counts against the code-run budget. "
            "The prelude helpers (catalog, base, W, save_edit, practice, tokenize, structure, set_structure, param_count, ...) are predefined."
        ),
        "input_schema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to execute."}},
                         "required": ["code"], "additionalProperties": False},
        "strict": True,
    }
    forward_pass = {
        "name": "forward_pass",
        "description": (
            f"Request one forward pass from the harness (costs 1 unit of the forward-pass budget; failed requests do "
            f"not count). model='base' is the unpruned {L}-layer start model; model='edited' is start + your saved "
            "edits, pruned by your current structure.json. kind='predict': run the model(s) on practice paragraphs and "
            "get accuracy, macro-F1, per-class recall, confusion matrix, per-item predictions/logits "
            "(pass_<k>/predictions.json) and, for the edited model, its structure and parameter count. "
            "kind='read': teacher-forced read of hidden states, the head's pooled vector, logits and/or attention "
            "for practice ids; arrays are written as .npy to pass_<k>/ and the result lists files, shapes and "
            "(edited model) layer_map."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["predict", "read"]},
                "model": {"type": "string", "enum": ["base", "edited", "both"],
                          "description": "Which model to run. 'both' is only valid for kind='predict' (paired comparison)."},
                "n": {"type": "integer", "description": "predict: number of practice paragraphs to sample (1-1024) when ids is not given. Default 256."},
                "seed": {"type": "integer", "description": "predict: seed for sampling practice paragraphs. Default 0."},
                "ids": {"type": "array", "items": {"type": "string"}, "description": "Practice ids to use (predict: up to 1024; read: up to 1024, required)."},
                "layers": {"type": "array", "items": {"type": "integer"},
                           "description": f"read: hidden layer indices 0..L (0 = embedding output, L = last encoder layer of the model read: {L} for base, number of kept layers for edited). Default all."},
                "outputs": {"type": "array",
                            "items": {"type": "string", "enum": ["cls_hidden", "mean_hidden", "pooled", "logits", "attention", "token_hidden"]},
                            "description": (f"read: which arrays to write. cls_hidden: [n,{arch.hidden}] hidden state at position 0 (the <s> token) per layer; "
                                            f"mean_hidden: [n,{arch.hidden}] masked token mean per layer. pooled: [n,{arch.hidden}] tanh(classifier.dense(h_last[<s>])), "
                                            f"the vector fed to classifier.out_proj. logits: [n,{arch.num_labels}]. attention: [n,{arch.heads},T,T] per layer (max 32 ids). "
                                            f"token_hidden: [n,T,{arch.hidden}] per layer (max 32 ids). Default ['cls_hidden'].")},
            },
            "required": ["kind"],
        },
    }
    submit = {
        "name": "submit",
        "description": (
            "Finish the run. Your current saved edits and structure.json become the final pruned model, which is "
            "evaluated on a held-out test set you never see and measured for parameter count. Refused if "
            "structure.json is invalid. Include a plain-English explanation of what you changed and why."
        ),
        "input_schema": {"type": "object", "properties": {"final_answer": {"type": "string", "description": "Explanation of the final edit and your reasoning."}},
                         "required": ["final_answer"], "additionalProperties": False},
        "strict": True,
    }
    return [run_python, forward_pass, submit]
