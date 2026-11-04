"""The agent loop: Claude edits BERT weights through run_python / forward_pass / submit.

Mirrors the Experiment 8 harness of lm_understand-logs: a budget of forward passes executed by the
harness outside the sandbox, a cap on code runs, a fresh conversation ("handoff") every N turns, and
per-turn logs. Adds a hard dollar cap and a static lint.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import anthropic
import yaml

from wsp.labels import LABELS, NUM_LABELS, label_block
from wsp.model import weights as W
from wsp.model.export import export_checkpoint
from wsp.model.forward import ForwardContext, run_pass
from wsp.harness.audit import lint
from wsp.harness.prelude import PRELUDE_DOCS, build_prelude
from wsp.harness.sandbox import Sandbox
from wsp.harness.tools import build_tools

ROOT = W.ROOT
PROMPTS = Path(__file__).parent / "prompts"
RUNS = ROOT / "runs"
DATA = ROOT / "data"


def load_config(path: Path | None = None, **overrides) -> dict:
    cfg = yaml.safe_load((path or ROOT / "configs" / "default.yaml").read_text())
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
    turn: int = 0                 # global turn counter
    conv_turn: int = 0            # turns in the current conversation
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
        if not (self.work / "practice.jsonl").exists():
            shutil.copy(DATA / "practice.jsonl", self.work / "practice.jsonl")
        self.practice = [json.loads(l) for l in (self.work / "practice.jsonl").read_text().splitlines() if l.strip()]
        self.base = W.load_base()
        self.ctx = ForwardContext(self.base, self.practice, self.work)
        sb = cfg["sandbox"]
        self.sandbox = Sandbox(self.work, self.run_dir / "sandbox_runs.jsonl", timeout_s=sb["timeout_s"],
                               stdout_cap=sb["stdout_cap"], stderr_tail=sb["stderr_tail"],
                               use_sandbox_exec=sb.get("use_sandbox_exec", True))
        self.prelude = build_prelude(self.work)
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
        counts = Counter(p["label"] for p in self.practice)
        cat_lines = []
        for t in W.catalog(self.base):
            tag = "editable" if t.editable else ("frozen" if t.in_model else "read-only, not in classifier")
            cat_lines.append(f"{t.name}  {list(t.shape)}  [{tag}]")
        return (PROMPTS / "task.md").read_text().format(
            num_labels=NUM_LABELS,
            labels=label_block(),
            n_practice=len(self.practice),
            label_counts=", ".join(f"{l}: {counts.get(l, 0)}" for l in LABELS),
            passes=self.state.passes,
            max_code_runs=self.cfg["max_code_runs"],
            handoff_every=self.cfg["handoff_every"],
            prelude_docs=PRELUDE_DOCS,
            catalog="\n".join(cat_lines),
        )

    def handoff_prompt(self) -> str:
        notes_p = self.work / "notes.md"
        notes = notes_p.read_text() if notes_p.exists() else "(no notes.md found)"
        if len(notes) > 40000:
            notes = notes[-40000:]
        edits = W.load_edits(self.work / "edits")
        if edits:
            man = W.edit_manifest(self.base, edits)
            es = "\n".join(f"- {t['name']} {t['shape']}: {t['changed_entries']:,} entries changed, ||delta||_F={t['frobenius_delta']:.3f}" for t in man["tensors"])
        else:
            es = "(no edits saved)"
        return (PROMPTS / "handoff.md").read_text().format(
            passes_left=self.state.passes - self.state.passes_used, passes=self.state.passes,
            code_runs_left=self.cfg["max_code_runs"] - self.state.code_runs, max_code_runs=self.cfg["max_code_runs"],
            turn=self.state.turn, notes=notes, edit_summary=es, last_pass=self.state.last_pass_summary,
            task=self.task_prompt(),
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
            tools=build_tools(bool(self.cfg.get("allow_free_texts", False))),
            messages=self.messages,
            cache_control={"type": "ephemeral"},
            thinking={"type": "adaptive", "display": self.cfg.get("thinking_display", "summarized")},
            output_config={"effort": self.cfg["effort"]},
        )
        # The SDK retries failed *requests*, but an error after a stream has started (read timeout,
        # overloaded, connection reset) propagates. Retry the whole turn; nothing was appended yet.
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
            if inp.get("texts") and not self.cfg.get("allow_free_texts", False):
                raise ValueError("free `texts` are not allowed in this run (rule 6: no self-authored data); use practice `ids`")
            t0 = time.time()
            summary = run_pass(self.ctx, inp, pass_dir)
            summary["pass_index"] = k
            summary["duration_s"] = round(time.time() - t0, 2)
            self.state.passes_used += 1
            summary["passes_left"] = self.state.passes - self.state.passes_used
            # make file paths relative to the working folder for the editor
            if "files" in summary:
                summary["files"] = {n: str(Path(p).relative_to(self.work)) for n, p in summary["files"].items()}
            out = json.dumps(summary, indent=1)
            req_path.write_text(json.dumps({"request": inp, "result": summary}, indent=1))
            self.state.last_pass_summary = out[:4000]
            self.log(f"forward pass {k}/{self.state.passes} ({summary.get('kind')}) ok")
        except Exception as e:  # invalid request: does not count
            self.state.passes_failed += 1
            shutil.rmtree(pass_dir, ignore_errors=True)
            out = f"Forward pass NOT executed (does not count): {type(e).__name__}: {e}"
            req_path.with_suffix(".failed.json").write_text(json.dumps({"request": inp, "error": str(e)}, indent=1))
            self.log(f"forward pass request failed: {e}")
        self.log_tool_call("forward_pass", inp, out)
        return out

    def tool_submit(self, inp: dict) -> str:
        self.finish("submitted", inp.get("final_answer", ""))
        return "Submitted. Run complete."

    def finish(self, reason: str, final_answer: str) -> None:
        if self.state.done:
            return
        self.state.done = True
        self.state.end_reason = reason
        (self.run_dir / "final_answer.md").write_text(final_answer or "(no final answer given)\n")
        edits = W.load_edits(self.work / "edits")
        manifest = W.save_edit_bundle(self.base, edits, self.run_dir / "edit")
        export_checkpoint(W.apply_edits(self.base, edits), self.run_dir / "model", manifest)
        self.log(f"finished ({reason}): {manifest['num_edited_tensors']} tensors edited, "
                 f"{manifest['total_changed_entries']:,} entries; passes {self.state.passes_used}/{self.state.passes}, "
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
        self.log(f"start: model={self.cfg['editor_model']} effort={self.cfg['effort']} passes={self.state.passes} cap=${self.cfg['dollar_cap']}")
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
            # Append the full assistant content (thinking blocks included) — append-only history.
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
