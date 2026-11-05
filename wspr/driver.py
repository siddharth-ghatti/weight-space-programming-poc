"""The agent loop for the reinsurance pruning harness: Claude shrinks a third-party checkpoint through
run_python / forward_pass / submit. Same controls as the earlier harnesses (pass budget, code-run cap,
dollar cap, lint, handoff every N turns, per-turn logs) plus a structure.json that drives pruning."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import anthropic
import yaml

from wsp.harness.audit import lint
from wsp.harness.sandbox import Sandbox
from wspr.arch import ROOT, label_block
from wspr.bundle import apply_edits, edit_manifest, load_edits, save_edit_bundle
from wspr.export import export_checkpoint
from wspr.forward import MAX_LEN, PrunedContext, build_model, run_pass
from wspr.prelude import build_prelude, prelude_docs, structure_docs
from wspr.start import load_start
from wspr.structure import StructureError, default_structure, load_structure, param_count
from wspr.tools import build_tools

PROMPTS = Path(__file__).parent / "prompts"
RUNS = ROOT / "runs"
DEFAULT_CONFIG = ROOT / "configs" / "reins.yaml"


def load_config(path: Path | None = None, **overrides) -> dict:
    cfg = yaml.safe_load((path or DEFAULT_CONFIG).read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


@dataclass
class State:
    name: str
    passes: int
    passes_used: int = 0
    passes_failed: int = 0
    code_runs: int = 0
    turn: int = 0
    conv_turn: int = 0
    conversation: int = 1
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    lint_rejections_in_row: int = 0
    lint_rejections_total: int = 0
    idle_replies_in_row: int = 0
    cap_notice_sent: bool = False
    done: bool = False
    end_reason: str = ""
    started: float = field(default_factory=time.time)
    last_pass_summary: str = "(none yet)"
    structure_invalid: str = ""


class Driver:
    def __init__(self, name: str, cfg: dict, *, resume: bool = False):
        self.cfg = cfg
        self.run_dir = RUNS / name
        if self.run_dir.exists() and not resume:
            raise FileExistsError(f"{self.run_dir} exists; pick another --name or pass --resume")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for d in ("turns", "requests", "work"):
            (self.run_dir / d).mkdir(exist_ok=True)
        self.work = self.run_dir / "work"
        self.data_dir = ROOT / cfg["data_dir"]
        if not (self.work / "practice.jsonl").exists():
            shutil.copy(self.data_dir / "practice.jsonl", self.work / "practice.jsonl")
        self.practice = [json.loads(l) for l in (self.work / "practice.jsonl").read_text().splitlines() if l.strip()]
        self.arch, self.start, self.provenance = load_start(cfg["model_dir"])
        (self.run_dir / "provenance.json").write_text(json.dumps(self.provenance, indent=2))
        self.ctx = PrunedContext(self.arch, self.start, self.practice, self.work)
        sb = cfg["sandbox"]
        self.sandbox = Sandbox(self.work, self.run_dir / "sandbox_runs.jsonl", timeout_s=sb["timeout_s"],
                               stdout_cap=sb["stdout_cap"], stderr_tail=sb["stderr_tail"],
                               use_sandbox_exec=sb.get("use_sandbox_exec", True))
        self.prelude = build_prelude(self.work, self.arch, MAX_LEN)
        self.tools = build_tools(self.arch)
        self.client = anthropic.Anthropic(max_retries=5, timeout=anthropic.Timeout(1800.0, connect=30.0))
        self.system = (PROMPTS / "system.md").read_text().strip()
        self.state_path = self.run_dir / "state.json"
        self.messages_path = self.run_dir / "messages.json"
        if resume and self.state_path.exists():
            self.state = State(**json.loads(self.state_path.read_text()))
            self.messages = json.loads(self.messages_path.read_text())
            self.log(f"resumed at turn {self.state.turn}")
        else:
            self.state = State(name=name, passes=int(cfg["passes"]))
            self.messages = [{"role": "user", "content": self.task_prompt()}]
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        (self.run_dir / "instructions.md").write_text(self.system + "\n")

    # ------------------------------------------------------------ prompts

    def task_prompt(self) -> str:
        a = self.arch
        counts = Counter(p["label"] for p in self.practice)
        cat_lines = [f"{name}  {list(self.start[name].shape)}  [{'frozen' if a.is_frozen(name) else 'editable'}]" for name in sorted(self.start)]
        acc0, f10 = self.provenance.get("start_test_accuracy"), self.provenance.get("start_test_macro_f1")
        acc_t, f1_t = self.cfg.get("accuracy_target"), self.cfg.get("macro_f1_target")
        stats_p = self.data_dir / "stats.json"
        stats = json.loads(stats_p.read_text()) if stats_p.exists() else {}
        return (PROMPTS / "task.md").read_text().format(
            hub_repo=self.provenance.get("hub_repo") or a.model_dir.name,
            num_layers=a.num_layers, last_layer=a.num_layers - 1, hidden=a.hidden, heads=a.heads, ffn=a.ffn, vocab=f"{a.vocab:,}", max_pos=a.max_pos,
            num_labels=a.num_labels, labels=label_block(a),
            start_params=f"{a.full_params:,}", non_embedding_params=f"{a.full_params - a.embedding_params:,}",
            start_accuracy_pct=f"{100 * acc0:.2f}" if acc0 is not None else "(unknown)",
            start_macro_f1=f"{f10:.4f}" if f10 is not None else "(unknown)",
            accuracy_target_pct=f"{100 * float(acc_t):.2f}" if acc_t is not None else "(unknown)",
            macro_f1_target=f"{float(f1_t):.4f}" if f1_t is not None else "(unknown)",
            fixed_params=f"{a.fixed_params:,}", attn_params=f"{a.attention_params:,}",
            ffn_unit_params=f"{a.ffn_unit_params:,}", embedding_params=f"{a.embedding_params:,}", head_params=f"{a.head_params:,}",
            n_practice=len(self.practice), label_counts=", ".join(f"{l}: {counts.get(l, 0)}" for l in a.labels),
            dataset=stats.get("dataset", "(dataset)"), dataset_license=stats.get("license", ""),
            passes=self.state.passes, max_code_runs=self.cfg["max_code_runs"], handoff_every=self.cfg["handoff_every"],
            prelude_docs=prelude_docs(a), structure_docs=structure_docs(a), catalog="\n".join(cat_lines),
        )

    def structure_summary(self) -> str:
        try:
            s = load_structure(self.work, self.arch)
        except StructureError as e:
            return f"structure.json is INVALID and must be fixed before submit: {e}"
        pc = param_count(s, self.arch)
        units = ", ".join(f"L{l}: {len(s.ffn_keep[l])}" for l in s.layers)
        return (f"layers kept = {list(s.layers)} (k={s.k}), intermediate_size m = {s.m} ({units}); "
                f"exported params = {pc['total']:,} ({100 * pc['reduction_vs_full']:.1f}% smaller than {self.arch.full_params:,}; "
                f"non-embedding {100 * pc['non_embedding_reduction_vs_full']:.1f}% smaller)")

    def handoff_prompt(self) -> str:
        notes_p = self.work / "notes.md"
        notes = notes_p.read_text() if notes_p.exists() else "(no notes.md found)"
        if len(notes) > 40000:
            notes = notes[-40000:]
        edits = load_edits(self.work / "edits")
        if edits:
            man = edit_manifest(self.start, edits, self.provenance, default_structure(self.arch), self.arch)
            es = "\n".join(f"- {t['name']} {t['shape']}: {t['changed_entries']:,} entries changed, ||delta||_F={t['frobenius_delta']:.3f}" for t in man["tensors"])
        else:
            es = "(no edits saved)"
        return (PROMPTS / "handoff.md").read_text().format(
            passes_left=self.state.passes - self.state.passes_used, passes=self.state.passes,
            code_runs_left=self.cfg["max_code_runs"] - self.state.code_runs, max_code_runs=self.cfg["max_code_runs"],
            turn=self.state.turn, notes=notes, structure_summary=self.structure_summary(), edit_summary=es,
            last_pass=self.state.last_pass_summary, task=self.task_prompt(),
        )

    # ------------------------------------------------------------ logging

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] t{self.state.turn:03d} ${self.state.cost_usd:.2f} {msg}"
        print(line, flush=True)
        with (self.run_dir / "driver.log").open("a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))
        self.messages_path.write_text(json.dumps(self.messages))

    def log_tool_call(self, tool: str, inp: dict, result: str, extra: dict | None = None) -> None:
        rec = {"turn": self.state.turn, "tool": tool, "input": inp, "result": result[:6000]}
        if extra:
            rec.update(extra)
        with (self.run_dir / "tool_calls.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def cost_of(self, usage) -> float:
        p = self.cfg["prices"]
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.state.input_tokens += usage.input_tokens
        self.state.output_tokens += usage.output_tokens
        self.state.cache_write_tokens += cw
        self.state.cache_read_tokens += cr
        return (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]
                + cw * p["cache_write"] + cr * p["cache_read"]) / 1e6

    # ------------------------------------------------------------ API

    def request(self):
        kwargs = dict(
            model=self.cfg["editor_model"],
            max_tokens=int(self.cfg["max_tokens"]),
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=self.tools,
            messages=self.messages,
            cache_control={"type": "ephemeral"},
            thinking={"type": "adaptive", "display": self.cfg.get("thinking_display", "summarized")},
            output_config={"effort": self.cfg["effort"]},
        )
        delays = [10, 30, 60, 120, 180, 300]
        for attempt in range(len(delays) + 1):
            try:
                with self.client.messages.stream(**kwargs) as stream:
                    return stream.get_final_message()
            except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError,
                    anthropic.RateLimitError) as e:
                if attempt == len(delays):
                    raise
                self.log(f"API error ({type(e).__name__}: {str(e)[:120]}); retry {attempt + 1}/{len(delays)} in {delays[attempt]}s")
                time.sleep(delays[attempt])
            except anthropic.APIStatusError as e:
                if e.status_code in (529, 503) and attempt < len(delays):
                    self.log(f"API {e.status_code}; retry {attempt + 1}/{len(delays)} in {delays[attempt]}s")
                    time.sleep(delays[attempt])
                    continue
                raise

    # ------------------------------------------------------------ tools

    def tool_run_python(self, inp: dict) -> str:
        code = inp.get("code", "")
        rejections, flags = lint(code)
        if rejections:
            self.state.lint_rejections_in_row += 1
            self.state.lint_rejections_total += 1
            msg = "Code not run. It violates the rules: " + "; ".join(rejections) + ". Rewrite it without these."
            if self.state.lint_rejections_in_row >= 2:
                msg += " (Second rejection in a row. The next violation ends the run.)"
            if self.state.lint_rejections_in_row >= 3:
                self.finish("lint_rejections", "Run ended after three consecutive rule violations.")
            self.log_tool_call("run_python", inp, msg, {"rejected": rejections})
            return msg
        self.state.lint_rejections_in_row = 0
        if self.state.code_runs >= self.cfg["max_code_runs"]:
            return "Code-run budget exhausted. Call submit."
        self.state.code_runs += 1
        r = self.sandbox.run(code, self.prelude)
        out = r.output.strip() or "(no output)"
        left = self.cfg["max_code_runs"] - self.state.code_runs
        out += f"\n[run {self.state.code_runs}; {left} code runs left; {self.state.passes - self.state.passes_used} forward passes left]"
        self.log_tool_call("run_python", inp, out, {"flags": flags, "duration_s": r.duration_s, "returncode": r.returncode})
        return out

    def tool_forward_pass(self, inp: dict) -> str:
        if self.state.passes_used >= self.state.passes:
            msg = "Forward-pass budget exhausted. Continue with code edits or call submit."
            self.log_tool_call("forward_pass", inp, msg)
            return msg
        k = self.state.passes_used + 1
        req_path = self.run_dir / "requests" / f"fwd_{k}.json"
        pass_dir = self.work / f"pass_{k}"
        try:
            if inp.get("texts"):
                raise ValueError("free `texts` are not allowed (rule 6: no self-authored data); use practice `ids`")
            t0 = time.time()
            summary = run_pass(self.ctx, inp, pass_dir)
            summary["pass_index"] = k
            summary["duration_s"] = round(time.time() - t0, 2)
            self.state.passes_used += 1
            summary["passes_left"] = self.state.passes - self.state.passes_used
            if "files" in summary:
                summary["files"] = {n: str(Path(p).relative_to(self.work)) for n, p in summary["files"].items()}
            out = json.dumps(summary, indent=1)
            req_path.write_text(json.dumps({"request": inp, "result": summary}, indent=1))
            self.state.last_pass_summary = out[:4000]
            self.log(f"forward pass {k}/{self.state.passes} ({summary.get('kind')}, {inp.get('model', 'default')}) ok")
        except Exception as e:
            self.state.passes_failed += 1
            shutil.rmtree(pass_dir, ignore_errors=True)
            out = f"Forward pass NOT executed (does not count): {type(e).__name__}: {e}"
            req_path.with_suffix(".failed.json").write_text(json.dumps({"request": inp, "error": str(e)}, indent=1))
            self.log(f"forward pass request failed: {e}")
        self.log_tool_call("forward_pass", inp, out)
        return out

    def tool_submit(self, inp: dict) -> str:
        try:
            load_structure(self.work, self.arch)
        except StructureError as e:
            msg = f"Not submitted: structure.json is invalid: {e}. Fix it (set_structure / clear_structure) and submit again."
            self.log_tool_call("submit", inp, msg)
            return msg
        self.finish("submitted", inp.get("final_answer", ""))
        return "Submitted. Run complete."

    def finish(self, reason: str, final_answer: str) -> None:
        if self.state.done:
            return
        self.state.done = True
        self.state.end_reason = reason
        (self.run_dir / "final_answer.md").write_text(final_answer or "(no final answer given)\n")
        try:
            structure = load_structure(self.work, self.arch)
        except StructureError as e:
            self.state.structure_invalid = str(e)
            self.log(f"structure.json invalid at finish ({e}); exporting the full structure")
            structure = default_structure(self.arch)
        edits = load_edits(self.work / "edits")
        manifest = save_edit_bundle(self.start, edits, self.run_dir / "edit", self.provenance, structure, self.arch)
        if self.state.structure_invalid:
            manifest["structure_invalid"] = self.state.structure_invalid
            (self.run_dir / "edit" / "manifest.json").write_text(json.dumps(manifest, indent=2))
        weights = apply_edits(self.start, edits, self.arch)
        export_checkpoint(weights, self.run_dir / "model", manifest, self.arch, structure=structure)
        pc = param_count(structure, self.arch)
        try:
            model = build_model(weights, self.arch, structure=structure)
            numel = sum(p.numel() for p in model.parameters())
            if numel != pc["total"]:
                self.log(f"WARNING: built model has {numel:,} params, arithmetic says {pc['total']:,}")
        except Exception as e:  # pragma: no cover
            self.log(f"WARNING: could not rebuild the exported model for a self-check: {e}")
        self.log(f"finished ({reason}): {manifest['num_edited_tensors']} tensors edited; structure k={structure.k} m={structure.m}; "
                 f"exported params {pc['total']:,} ({100 * pc['reduction_vs_full']:.1f}% smaller, non-embedding "
                 f"{100 * pc['non_embedding_reduction_vs_full']:.1f}% smaller); passes {self.state.passes_used}/{self.state.passes}, "
                 f"code runs {self.state.code_runs}, cost ${self.state.cost_usd:.2f}")

    # ------------------------------------------------------------ loop

    def handle_tool_use(self, block) -> dict:
        inp = block.input if isinstance(block.input, dict) else json.loads(block.input)
        if block.name == "run_python":
            result = self.tool_run_python(inp)
        elif block.name == "forward_pass":
            result = self.tool_forward_pass(inp)
        elif block.name == "submit":
            result = self.tool_submit(inp)
        else:
            result = f"unknown tool {block.name}"
        return {"type": "tool_result", "tool_use_id": block.id, "content": result}

    def maybe_handoff(self) -> None:
        if self.state.conv_turn >= int(self.cfg["handoff_every"]):
            self.state.conversation += 1
            self.state.conv_turn = 0
            self.messages = [{"role": "user", "content": self.handoff_prompt()}]
            self.log(f"handoff -> conversation {self.state.conversation}")

    def budget_notice(self) -> str | None:
        if self.state.cost_usd >= float(self.cfg["dollar_cap"]):
            if not self.state.cap_notice_sent:
                self.state.cap_notice_sent = True
                return ("Experimenter notice: the spending cap for this run has been reached. "
                        "Call submit now with your final explanation; no further code runs or forward passes will be executed.")
            self.finish("dollar_cap", "(auto-submitted at the dollar cap)")
        if self.state.turn >= int(self.cfg["max_turns"]):
            self.finish("max_turns", "(auto-submitted at max turns)")
        return None

    def run(self) -> State:
        self.log(f"start: model={self.cfg['editor_model']} effort={self.cfg['effort']} passes={self.state.passes} "
                 f"cap=${self.cfg['dollar_cap']} start={self.cfg.get('model_dir')} targets=acc>={self.cfg.get('accuracy_target')} f1>={self.cfg.get('macro_f1_target')}")
        while not self.state.done:
            self.maybe_handoff()
            notice = self.budget_notice()
            if self.state.done:
                break
            if notice:
                self.messages.append({"role": "user", "content": notice})
            self.state.turn += 1
            self.state.conv_turn += 1
            tdir = self.run_dir / "turns" / f"t{self.state.turn:03d}"
            tdir.mkdir(exist_ok=True)
            last_user = self.messages[-1]["content"]
            (tdir / "prompt.txt").write_text(last_user if isinstance(last_user, str) else json.dumps(last_user, indent=1))
            try:
                resp = self.request()
            except anthropic.APIError as e:
                self.log(f"API error: {e}")
                self.save_state()
                raise
            turn_cost = self.cost_of(resp.usage)
            self.state.cost_usd += turn_cost
            content = [b.model_dump() for b in resp.content]
            (tdir / "events.jsonl").write_text("\n".join(json.dumps(c) for c in content) + "\n")
            (tdir / "run.json").write_text(json.dumps({"stop_reason": resp.stop_reason, "usage": resp.usage.model_dump(),
                                                        "cost_usd": turn_cost, "model": resp.model, "request_id": getattr(resp, "_request_id", None)}, indent=1))
            texts = [b.text for b in resp.content if b.type == "text"]
            thinks = [b.thinking for b in resp.content if b.type == "thinking" and getattr(b, "thinking", "")]
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            answer = ""
            if thinks:
                answer += "## reasoning summary\n\n" + "\n\n".join(thinks) + "\n\n"
            if texts:
                answer += "## message\n\n" + "\n\n".join(texts) + "\n\n"
            for tu in tool_uses:
                answer += f"## tool_use: {tu.name}\n\n```json\n{json.dumps(tu.input, indent=1)[:20000]}\n```\n\n"
            (tdir / "answer.md").write_text(answer)
            if texts:
                with (self.run_dir / "explanations.md").open("a") as f:
                    f.write(f"\n\n## turn {self.state.turn}\n\n" + "\n\n".join(texts))
            self.messages.append({"role": "assistant", "content": content})
            self.log(f"stop={resp.stop_reason} tools={[t.name for t in tool_uses]} out_tok={resp.usage.output_tokens}")

            if resp.stop_reason == "refusal":
                self.log(f"refusal: {getattr(resp, 'stop_details', None)}")
                self.finish("refusal", "(run ended by a refusal stop)")
                break
            if tool_uses:
                results = []
                for tu in tool_uses:
                    results.append(self.handle_tool_use(tu))
                    if self.state.done:
                        break
                self.messages.append({"role": "user", "content": results})
                self.state.idle_replies_in_row = 0
            elif resp.stop_reason == "max_tokens":
                self.messages.append({"role": "user", "content": "Your reply was cut off at the output limit. Continue, more concisely."})
            else:
                self.state.idle_replies_in_row += 1
                if self.state.idle_replies_in_row >= 3:
                    self.finish("idle", "(auto-submitted after three replies without a tool call)")
                    break
                self.messages.append({"role": "user", "content": (
                    "You replied without calling a tool. Use run_python or forward_pass to continue, "
                    "or call submit if you are done.")})
            self.save_state()
        self.save_state()
        self.sandbox.cleanup()
        return self.state
