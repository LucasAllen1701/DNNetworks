#!/usr/bin/env python3
"""A small local dashboard for running the experiments.

    python dashboard.py            # then open http://127.0.0.1:8765

It lists the scripts in experiments/, runs one at a time, shows the output,
and compares the results with the files committed in experiments/results/.
Each run is kept in runs/<time>_<experiment>/ (log, results, report), so the
committed results are never overwritten.

Only the Python standard library is used. The server listens on this
computer only (127.0.0.1) and can start nothing but the experiment scripts
and the test suite of this repository.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.join(ROOT, "experiments")
COMMITTED = os.path.join(EXPERIMENTS, "results")
RUNS = os.path.join(ROOT, "runs")
TESTS = "tests"

lock = threading.Lock()
current = {}          # the run in progress: id, exp, proc, started
_cache = {}


def cached(seconds):
    """Remember a function's result for a few seconds; the page asks every 2 s."""
    def wrap(fn):
        def inner():
            hit = _cache.get(fn.__name__)
            if hit is None or time.time() - hit[0] > seconds:
                hit = _cache[fn.__name__] = (time.time(), fn())
            return hit[1]
        return inner
    return wrap


# ---- what can be run

@cached(5)
def experiments():
    """[{id, number, title, doc}] for experiments/exp<N>_*.py, in order."""
    found = []
    for name in os.listdir(EXPERIMENTS):
        m = re.fullmatch(r"exp(\d+)_\w+\.py", name)
        if not m:
            continue
        try:
            with open(os.path.join(EXPERIMENTS, name)) as f:
                doc = ast.get_docstring(ast.parse(f.read())) or ""
        except (OSError, SyntaxError):
            doc = ""
        title = doc.split("\n\n")[0].replace("\n", " ").strip() or name
        found.append(dict(id=name[:-3], number=int(m.group(1)), title=title, doc=doc))
    found.sort(key=lambda e: e["number"])
    return [dict(id=TESTS, number=0, title="Check the setup: run the test suite",
                 doc="Runs every test in tests/ with pytest. Takes about a minute. "
                     "If this passes, the installation is working.")] + found


def command(exp_id):
    if exp_id == TESTS:
        return [sys.executable, "-m", "pytest", "-q"]
    return [sys.executable, "-u", os.path.join("experiments", exp_id + ".py")]


# ---- environment

@cached(30)
def versions():
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "cpus": os.cpu_count()}
    for mod in ("numpy", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        out["commit"] = (head + (" with local changes" if dirty else "")) if head else "unknown"
    except Exception:
        out["commit"] = "unknown"
    return out


# ---- comparing a run's results with the committed ones

def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def same_cell(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return abs(x - y) <= 1e-9 * (1 + abs(x))


def compare(run_dir):
    """One entry per CSV written by the run. Columns that hold timings are ignored."""
    out = []
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(run_dir, name)
        cols, rows = read_csv(path)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
        entry = dict(file=name, rows=len(rows), sha256=digest, differences={})
        ref = os.path.join(COMMITTED, name)
        if not os.path.exists(ref):
            entry["status"] = "new"
            entry["note"] = "no committed file with this name to compare with"
        else:
            rcols, rrows = read_csv(ref)
            if cols != rcols:
                entry["status"] = "differs"
                entry["note"] = "the columns differ from the committed file"
            elif len(rows) != len(rrows):
                entry["status"] = "differs"
                entry["note"] = f"{len(rows)} rows here, {len(rrows)} in the committed file"
            else:
                for j, col in enumerate(cols):
                    if "seconds" in col:
                        continue
                    n = sum(not same_cell(a[j], b[j]) for a, b in zip(rows, rrows))
                    if n:
                        entry["differences"][col] = n
                entry["status"] = "differs" if entry["differences"] else "identical"
        out.append(entry)
    return out


def duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


def report(meta, comparison):
    v = meta["versions"]
    lines = ["DNN dashboard report",
             f"experiment: {meta['exp']}",
             f"run: {meta['id']}",
             f"result: {meta['state']} (exit code {meta['exit_code']}) after {duration(meta['seconds'])}",
             f"python {v['python']}, numpy {v['numpy']}, scipy {v['scipy']}, {v['cpus']} CPUs",
             f"system: {v['platform']}",
             f"repository commit: {v['commit']}"]
    if comparison:
        lines.append("results compared with experiments/results/ (timing columns ignored):")
        for c in comparison:
            if c["status"] == "identical":
                detail = "identical"
            elif c["status"] == "new":
                detail = "new file, " + c["note"]
            elif c.get("note"):
                detail = "DIFFERS: " + c["note"]
            else:
                detail = "DIFFERS in " + ", ".join(f"{col} ({n} cells)" for col, n in c["differences"].items())
            lines.append(f"  {c['file']}: {c['rows']} rows, sha256 {c.get('sha256', '?')}, {detail}")
    elif meta["exp"] != TESTS:
        lines.append("no result files were written")
    return "\n".join(lines)


# ---- runs

def run_dir(run_id):
    if not re.fullmatch(r"[\w.-]+", run_id or ""):
        raise ValueError("bad run id")
    return os.path.join(RUNS, run_id)


def load_meta(run_id):
    try:
        with open(os.path.join(run_dir(run_id), "meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def past_runs():
    if not os.path.isdir(RUNS):
        return []
    metas = [m for m in (load_meta(d) for d in sorted(os.listdir(RUNS), reverse=True)) if m]
    return [{k: m[k] for k in ("id", "exp", "state", "exit_code", "seconds", "started")} for m in metas]


def finish(run_id, exp_id, proc, started):
    code = proc.wait()
    with lock:
        stopped = current.get("stopped", False)
    state = "stopped" if stopped else ("ok" if code == 0 else "failed")
    meta = dict(id=run_id, exp=exp_id, state=state, exit_code=code, seconds=round(time.time() - started, 1),
                started=time.strftime("%Y-%m-%d %H:%M", time.localtime(started)), versions=versions())
    d = run_dir(run_id)
    comparison = compare(d)
    with open(os.path.join(d, "comparison.json"), "w") as f:
        json.dump(comparison, f)
    with open(os.path.join(d, "report.txt"), "w") as f:
        f.write(report(meta, comparison) + "\n")
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    with lock:
        current.clear()               # only now is the run over for the page


def start(exp_id):
    if exp_id not in {e["id"] for e in experiments()}:
        raise ValueError("unknown experiment")
    with lock:
        if current:
            raise ValueError("another run is in progress")
        started = time.time()
        run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(started)) + "_" + exp_id
        d = run_dir(run_id)
        os.makedirs(d)
        env = dict(os.environ, DNN_RESULTS_DIR=d, PYTHONUNBUFFERED="1")
        log = open(os.path.join(d, "log.txt"), "w")
        proc = subprocess.Popen(command(exp_id), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        log.close()
        current.update(id=run_id, exp=exp_id, proc=proc, started=started)
    threading.Thread(target=finish, args=(run_id, exp_id, proc, started), daemon=True).start()
    return run_id


def stop():
    with lock:
        proc = current.get("proc")
        if proc is None:
            return False
        current["stopped"] = True
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return True


def state():
    with lock:
        running = ({"id": current["id"], "exp": current["exp"],
                    "seconds": round(time.time() - current["started"], 1)} if current else None)
    return dict(experiments=experiments(), runs=past_runs(), running=running, versions=versions())


def run_details(run_id):
    d = run_dir(run_id)
    if not os.path.isdir(d):
        raise ValueError("no such run")

    def text(name):
        try:
            with open(os.path.join(d, name), errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    comparison = []
    if os.path.exists(os.path.join(d, "comparison.json")):
        comparison = json.loads(text("comparison.json"))
    return dict(meta=load_meta(run_id), log=text("log.txt")[-400_000:], report=text("report.txt"),
                comparison=comparison, files=sorted(n for n in os.listdir(d) if n.endswith(".csv")),
                folder=os.path.relpath(d, ROOT))


def csv_data(run_id, name):
    if not re.fullmatch(r"[\w.-]+\.csv", name or ""):
        raise ValueError("bad file name")
    cols, rows = read_csv(os.path.join(run_dir(run_id), name))
    return dict(columns=cols, rows=rows[:50_000], total=len(rows))


# ---- the web server

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def local(self):
        """Refuse requests that were not addressed to this computer by name."""
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def send(self, body, kind="application/json", status=200):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.local():
            return self.send({"error": "forbidden"}, status=403)
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self.send(PAGE, "text/html")
            elif url.path == "/favicon.ico":
                self.send("", "image/x-icon", status=204)
            elif url.path == "/api/state":
                self.send(state())
            elif url.path == "/api/run":
                self.send(run_details(q.get("id", "")))
            elif url.path == "/api/csv":
                self.send(csv_data(q.get("run", ""), q.get("name", "")))
            else:
                self.send({"error": "not found"}, status=404)
        except (ValueError, OSError) as e:
            self.send({"error": str(e)}, status=400)

    def do_POST(self):
        # The page sends this header; another web site cannot without permission.
        if not self.local() or self.headers.get("X-DNN-Dashboard") != "1":
            return self.send({"error": "forbidden"}, status=403)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/start":
                self.send({"run": start(body.get("exp", ""))})
            elif self.path == "/api/stop":
                self.send({"stopped": stop()})
            else:
                self.send({"error": "not found"}, status=404)
        except (ValueError, OSError) as e:
            self.send({"error": str(e)}, status=400)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DNN dashboard</title>
<style>
:root {
  color-scheme: light dark;
  --page: #f3f2ef; --surface: #fcfcfb; --line: #dddcd6; --grid: #e6e5e0;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #7b7a74; --accent: #2a78d6; --accent-text: #ffffff;
  --good: #0ca30c; --critical: #d03b3b; --warning: #b07a00;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #111110; --surface: #1a1a19; --line: #33332f; --grid: #2a2a27;
    --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e85; --accent: #3987e5; --accent-text: #ffffff;
    --good: #0ca30c; --critical: #e66767; --warning: #fab219;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 15px/1.45 system-ui, sans-serif; background: var(--page); color: var(--text); }
header { padding: 14px 20px; border-bottom: 1px solid var(--line); background: var(--surface);
  display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
header h1 { font-size: 18px; margin: 0; }
header .env { color: var(--text-2); font-size: 13px; }
main { display: grid; grid-template-columns: 330px minmax(0, 1fr); gap: 16px; padding: 16px 20px; align-items: start; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; } }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
#list { padding: 6px; position: sticky; top: 16px; max-height: calc(100vh - 100px); overflow-y: auto;
  overscroll-behavior: contain; }
@media (max-width: 860px) { #list { position: static; max-height: 40vh; } }
#list button { display: block; width: 100%; text-align: left; background: none; border: 0; color: inherit;
  font: inherit; padding: 8px 10px; border-radius: 6px; cursor: pointer; }
#list button:hover { background: var(--page); }
#list button.selected { background: var(--page); box-shadow: inset 3px 0 0 var(--accent); }
#list .name { display: block; }
#list .last { display: block; font-size: 12.5px; color: var(--text-2); }
#detail { padding: 18px 20px; min-height: 300px; }
#detail h2 { font-size: 17px; margin: 0 0 4px; }
.doc { white-space: pre-wrap; color: var(--text-2); font-size: 13.5px; margin: 8px 0 0; max-height: 14em;
  overflow: auto; border-left: 2px solid var(--line); padding-left: 12px; }
.bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 16px 0 6px; }
button.act { font: inherit; padding: 7px 16px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--surface); color: var(--text); cursor: pointer; }
button.act.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-text); }
button.act:disabled { opacity: .5; cursor: default; }
select { font: inherit; padding: 5px 8px; border-radius: 6px; border: 1px solid var(--line);
  background: var(--surface); color: var(--text); max-width: 100%; }
.status { color: var(--text-2); }
.badge { font-weight: 600; }
.badge.ok { color: var(--good); } .badge.failed, .badge.differs { color: var(--critical); }
.badge.stopped, .badge.new { color: var(--warning); }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--line); margin-top: 18px; flex-wrap: wrap; }
.tabs button { font: inherit; background: none; border: 0; color: var(--text-2); padding: 8px 12px; cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabs button.on { color: var(--text); border-bottom-color: var(--accent); }
.pane { padding-top: 14px; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; font: 13px/1.45 ui-monospace, monospace; }
.scroll { overflow: auto; max-width: 100%; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0 14px; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: left; white-space: nowrap; }
th { color: var(--text-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.controls { display: flex; gap: 14px; flex-wrap: wrap; align-items: end; margin-bottom: 10px; }
.controls label { display: grid; gap: 3px; font-size: 12.5px; color: var(--text-2); }
.note { color: var(--text-2); font-size: 13.5px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-2); margin: 4px 0 2px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
#tip { position: fixed; pointer-events: none; background: var(--text); color: var(--surface); font-size: 12.5px;
  padding: 5px 8px; border-radius: 5px; display: none; z-index: 5; max-width: 260px; white-space: pre-line; }
svg text { fill: var(--text-2); font: 12px system-ui, sans-serif; }
svg .title { fill: var(--text); font-size: 13.5px; font-weight: 600; }
</style>
</head>
<body>
<header><h1>Discrete neural networks: experiments</h1><span class="env" id="env"></span></header>
<main>
  <nav class="card" id="list" aria-label="Experiments"></nav>
  <section class="card" id="detail"><p class="note">Loading…</p></section>
</main>
<div id="tip" role="tooltip"></div>
<script>
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const mark = {ok: "✓ finished", failed: "✕ failed", stopped: "■ stopped"};
let S = null, selected = null, shownRun = null, tab = "summary", details = null, csvCache = {}, chartSel = {};

async function api(path, body) {
  const opt = body ? {method: "POST", headers: {"Content-Type": "application/json", "X-DNN-Dashboard": "1"},
                      body: JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function dur(s) { s = Math.round(s); return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`; }
function runsOf(exp) { return S.runs.filter(r => r.exp === exp); }

function renderList() {
  $("#list").innerHTML = S.experiments.map(e => {
    const last = runsOf(e.id)[0];
    const running = S.running && S.running.exp === e.id;
    const line = running ? `running, ${dur(S.running.seconds)}`
      : last ? `<span class="badge ${last.state}">${mark[last.state]}</span> in ${dur(last.seconds)}, ${esc(last.started)}`
      : "not run yet";
    const name = e.id === "tests" ? e.title : e.title.replace(/\.$/, "");
    return `<button data-id="${esc(e.id)}" class="${e.id === selected ? "selected" : ""}">
      <span class="name">${esc(name)}</span><span class="last">${line}</span></button>`;
  }).join("");
  for (const b of $("#list").children) b.onclick = () => select(b.dataset.id);
}

function select(id) {
  selected = id; details = null; csvCache = {}; chartSel = {}; tab = "summary";
  const run = S.running && S.running.exp === id ? S.running.id : (runsOf(id)[0] || {}).id;
  shownRun = run || null;
  renderList(); renderDetail(); loadRun();
}

function renderDetail() {
  const e = S.experiments.find(x => x.id === selected);
  if (!e) return;
  const mine = S.running && S.running.exp === e.id;
  const busy = !!S.running;
  const prev = runsOf(e.id).find(r => r.state === "ok");
  const runs = runsOf(e.id);
  const status = mine ? `Running for ${dur(S.running.seconds)}` + (prev ? ` (last time it took ${dur(prev.seconds)})` : "")
    : busy ? "Another run is in progress" : "";
  $("#detail").innerHTML = `
    <h2>${esc(e.title)}</h2>
    <div class="doc">${esc(e.doc.includes("\n\n") ? e.doc.slice(e.doc.indexOf("\n\n") + 2) : "")}</div>
    <div class="bar">
      <button class="act primary" id="go" ${busy ? "disabled" : ""}>Run</button>
      <button class="act" id="halt" ${mine ? "" : "disabled"}>Stop</button>
      <span class="status" id="status">${esc(status)}</span>
    </div>
    ${runs.length ? `<label class="note">Showing run <select id="pick">${runs.map(r =>
        `<option value="${esc(r.id)}" ${r.id === shownRun ? "selected" : ""}>${esc(r.started)}: ${mark[r.state]}</option>`).join("")}
        ${mine ? `<option value="${esc(S.running.id)}" selected>in progress</option>` : ""}</select></label>` : ""}
    <div class="tabs" role="tablist">${["summary", "output", "tables", "chart"].map(t =>
      `<button role="tab" class="${t === tab ? "on" : ""}" data-tab="${t}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}</div>
    <div class="pane" id="pane"></div>`;
  $("#go").onclick = async () => { try { const r = await api("/api/start", {exp: e.id}); shownRun = r.run; details = null; await refresh(); }
                                   catch (err) { $("#status").textContent = err.message; } };
  $("#halt").onclick = () => api("/api/stop", {}).then(refresh);
  if ($("#pick")) $("#pick").onchange = ev => { shownRun = ev.target.value; details = null; csvCache = {}; chartSel = {}; loadRun(); };
  for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => { tab = b.dataset.tab; renderDetail(); };
  renderPane();
}

async function loadRun() {
  if (!shownRun) { details = null; renderPane(); return; }
  try { details = await api("/api/run?id=" + encodeURIComponent(shownRun)); } catch (err) { details = null; }
  renderPane();
}

function copyButton(id, label) { return `<button class="act" data-copy="${id}">${label}</button>`; }
function wireCopy() {
  for (const b of document.querySelectorAll("[data-copy]")) b.onclick = async () => {
    const text = b.dataset.copy === "report" ? details.report : details.report + "\n--- output ---\n" + details.log;
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; }
    catch (e) { b.textContent = "Copy failed: select the text instead"; }
  };
}

function outputHtml(log) {
  // The scripts print their tables in Markdown; show those as tables.
  const out = []; let rows = [];
  const flush = () => {
    if (!rows.length) return;
    const cells = rows.filter(r => !/^\|[\s|:-]+\|$/.test(r)).map(r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
    out.push(`<div class="scroll"><table><thead><tr>${cells[0].map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
      cells.slice(1).map(r => `<tr>${r.map(c => `<td class="${/^[-–\d.\s\/%]+$/.test(c) ? "num" : ""}">${esc(c.replace(/\*\*/g, ""))}</td>`).join("")}</tr>`).join("") +
      "</tbody></table></div>");
    rows = [];
  };
  let text = [];
  const flushText = () => { if (text.join("").trim()) out.push(`<pre>${esc(text.join("\n").trim())}</pre>`); text = []; };
  for (const line of log.split("\n")) {
    if (/^\|.*\|\s*$/.test(line)) { flushText(); rows.push(line.trim()); }
    else { flush(); text.push(line); }
  }
  flush(); flushText();
  return out.join("") || `<p class="note">No output yet. Most experiments print their tables only when they finish.</p>`;
}

function renderPane() {
  const pane = $("#pane");
  $("#tip").style.display = "none";
  if (!pane) return;
  if (!shownRun || !details) { pane.innerHTML = `<p class="note">${shownRun ? "Loading…" : "This has not been run on this computer yet."}</p>`; return; }
  const m = details.meta;
  if (tab === "output") { pane.innerHTML = outputHtml(details.log); return; }
  if (tab === "summary") {
    if (!m) { pane.innerHTML = `<p class="note">The run is in progress. The summary appears when it finishes.</p>`; return; }
    const cmp = details.comparison;
    pane.innerHTML = `
      <p><span class="badge ${m.state}">${mark[m.state]}</span> after ${dur(m.seconds)}. Saved in <code>${esc(details.folder)}/</code>.</p>
      ${cmp.length ? `<p class="note">Results compared with the files committed in <code>experiments/results/</code>. Timing columns are ignored.</p>
      <div class="scroll"><table><thead><tr><th>File</th><th>Rows</th><th>Compared with the committed file</th></tr></thead><tbody>${cmp.map(c => `
        <tr><td>${esc(c.file)}</td><td class="num">${c.rows}</td><td><span class="badge ${c.status}">${
          c.status === "identical" ? "✓ identical" : c.status === "new" ? "● new file" : "✕ differs"}</span>${
          c.note ? ": " + esc(c.note) : Object.keys(c.differences).length ? ": " + Object.entries(c.differences).map(([k, n]) => `${esc(k)} (${n} cells)`).join(", ") : ""}</td></tr>`).join("")}
      </tbody></table></div>` : ""}
      <p class="note">To send the result to Claude, copy the report and paste it into the chat. If a file differs or is new, attach the CSV files from the run folder as well.</p>
      <div class="bar">${copyButton("report", "Copy report")}${copyButton("full", "Copy report and full output")}</div>
      <pre>${esc(details.report)}</pre>`;
    wireCopy(); return;
  }
  if (!details.files.length) { pane.innerHTML = `<p class="note">This run wrote no result files${m ? "" : " yet"}.</p>`; return; }
  const file = chartSel.file && details.files.includes(chartSel.file) ? chartSel.file : details.files[0];
  chartSel.file = file;
  const picker = `<label>Result file<select id="file">${details.files.map(f => `<option ${f === file ? "selected" : ""}>${esc(f)}</option>`).join("")}</select></label>`;
  const key = shownRun + "/" + file;
  if (!csvCache[key]) {
    pane.innerHTML = `<div class="controls">${picker}</div><p class="note">Loading…</p>`;
    api(`/api/csv?run=${encodeURIComponent(shownRun)}&name=${encodeURIComponent(file)}`).then(d => { csvCache[key] = d; renderPane(); });
  } else if (tab === "tables") {
    const d = csvCache[key], shown = d.rows.slice(0, 300);
    pane.innerHTML = `<div class="controls">${picker}</div>
      <p class="note">${d.total} rows${d.total > shown.length ? `, first ${shown.length} shown` : ""}.</p>
      <div class="scroll" style="max-height: 60vh"><table><thead><tr>${d.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${shown.map(r => `<tr>${r.map(c => `<td class="${c !== "" && !isNaN(c) ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  } else {
    renderChart(pane, picker, csvCache[key]);
  }
  if ($("#file")) $("#file").onchange = ev => { chartSel = {file: ev.target.value}; renderPane(); };
}

// ---- chart: the mean of one measure for each value of one column, optionally split by a second column
function renderChart(pane, picker, d) {
  const cols = d.columns.map((name, j) => {
    const vals = d.rows.map(r => r[j]);
    const distinct = [...new Set(vals)];
    const numeric = vals.every(v => v === "" || (!isNaN(v) && v.trim() !== "")) && vals.some(v => v !== "");
    return {name, j, distinct, numeric};
  });
  const measures = cols.filter(c => c.numeric && c.distinct.length > 1 && !/seed/.test(c.name));
  const dims = cols.filter(c => c.distinct.length >= 2 && c.distinct.length <= 40 && c.name !== "seed");
  if (!measures.length || !dims.length) { pane.innerHTML = `<div class="controls">${picker}</div><p class="note">This file has nothing to chart.</p>`; return; }
  const pick = (want, list, fallback) => list.find(c => c.name === want) || fallback;
  const y = pick(chartSel.y, measures, measures.find(c => /exact|acc|recall/.test(c.name)) || measures[0]);
  const x = pick(chartSel.x, dims, dims.find(c => /^(n_train|task|round|depth|h|rule|capacity|layout)$/.test(c.name)) || dims[0]);
  const splits = dims.filter(c => c !== x && c !== y && c.distinct.length <= 4);
  const s = chartSel.s === "" ? null : pick(chartSel.s, splits, null);
  const opts = (list, cur) => list.map(c => `<option ${c === cur ? "selected" : ""}>${esc(c.name)}</option>`).join("");
  const xNumeric = x.numeric;
  const xs = [...x.distinct].sort((a, b) => xNumeric ? a - b : 0);
  const groups = s ? [...s.distinct].sort() : [null];
  const mean = (xv, g) => {
    const v = d.rows.filter(r => r[x.j] === xv && (g === null || r[s.j] === g) && r[y.j] !== "").map(r => +r[y.j]);
    return v.length ? {mean: v.reduce((a, b) => a + b, 0) / v.length, n: v.length} : null;
  };
  const data = groups.map(g => xs.map(xv => mean(xv, g)));
  const top = Math.max(...data.flat().filter(Boolean).map(p => p.mean), 0);
  const low = Math.min(...data.flat().filter(Boolean).map(p => p.mean), 0);
  const nice = v => { if (v <= 0) return 1; const p = 10 ** Math.floor(Math.log10(v)); return Math.ceil(v / p - 1e-9) * p; };
  const yMax = nice(top), yMin = low < 0 ? -nice(-low) : 0;
  const W = Math.max(520, Math.min(980, 90 + xs.length * Math.max(36, 26 * groups.length))), H = 320;
  const L = 56, R = s && xNumeric ? 110 : 16, T = 34, B = xs.some(v => String(v).length > 5) ? 86 : 44;
  const px = i => L + (i + 0.5) * (W - L - R) / xs.length;
  const py = v => T + (yMax - v) * (H - T - B) / (yMax - yMin);
  const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3).replace(/\.?0+$/, "") || "0";
  const color = k => `var(--series-${k + 1})`;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = yMin + (yMax - yMin) * k / 4;
    g += `<line x1="${L}" x2="${W - R}" y1="${py(v)}" y2="${py(v)}" stroke="var(--grid)"/>` +
         `<text x="${L - 8}" y="${py(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
  }
  xs.forEach((xv, i) => {
    const rot = B > 50;
    g += `<text x="${px(i)}" y="${H - B + 16}" text-anchor="${rot ? "end" : "middle"}" ${rot ? `transform="rotate(-40 ${px(i)} ${H - B + 16})"` : ""}>${esc(xv)}</text>`;
  });
  const tipOf = (xv, grp, p) => `${x.name} = ${xv}${grp === null ? "" : `, ${s.name} = ${grp}`}: ${fmt(p.mean)} (mean of ${p.n} rows)`;
  if (xNumeric) {
    data.forEach((series, k) => {
      const pts = series.map((p, i) => p && [px(i), py(p.mean)]).filter(Boolean);
      g += `<polyline fill="none" stroke="${color(k)}" stroke-width="2" stroke-linejoin="round" points="${pts.map(p => p.join(",")).join(" ")}"/>`;
      series.forEach((p, i) => { if (p) g += `<circle cx="${px(i)}" cy="${py(p.mean)}" r="4.5" fill="${color(k)}" stroke="var(--surface)" stroke-width="2"/>`; });
      if (s && pts.length) { const e = pts[pts.length - 1]; g += `<text x="${e[0] + 10}" y="${e[1] + 4}">${esc(groups[k])}</text>`; }
    });
    // one hover column per x value, listing every series there (points can coincide)
    const band = (W - L - R) / xs.length;
    xs.forEach((xv, i) => {
      const lines = data.map((series, k) => series[i] && `${s ? groups[k] + ": " : ""}${fmt(series[i].mean)} (mean of ${series[i].n} rows)`).filter(Boolean);
      if (lines.length) g += `<rect x="${px(i) - band / 2}" y="${T}" width="${band}" height="${H - T - B}" fill="transparent" data-tip="${esc(`${x.name} = ${xv}\n` + lines.join("\n"))}"/>`;
    });
  } else {
    const band = (W - L - R) / xs.length, bw = Math.min(34, (band - 10) / groups.length - 2);
    data.forEach((series, k) => series.forEach((p, i) => {
      if (!p) return;
      const x0 = px(i) - (groups.length * (bw + 2) - 2) / 2 + k * (bw + 2), y0 = py(Math.max(p.mean, 0)), h = Math.abs(py(p.mean) - py(0));
      const r = Math.min(4, h, bw / 2);
      const path = p.mean >= 0
        ? `M${x0},${y0 + h} V${y0 + r} Q${x0},${y0} ${x0 + r},${y0} H${x0 + bw - r} Q${x0 + bw},${y0} ${x0 + bw},${y0 + r} V${y0 + h} Z`
        : `M${x0},${y0} V${y0 + h - r} Q${x0},${y0 + h} ${x0 + r},${y0 + h} H${x0 + bw - r} Q${x0 + bw},${y0 + h} ${x0 + bw},${y0 + h - r} V${y0} Z`;
      g += `<path d="${path}" fill="${color(k)}"/><rect x="${x0 - 1}" y="${T}" width="${bw + 2}" height="${H - T - B}" fill="transparent" data-tip="${esc(tipOf(xs[i], groups[k], p))}"/>`;
    }));
  }
  g += `<line x1="${L}" x2="${W - R}" y1="${py(0)}" y2="${py(0)}" stroke="var(--text-3)"/>`;
  const title = `Mean of ${y.name} by ${x.name}` + (s ? `, split by ${s.name}` : "");
  pane.innerHTML = `<div class="controls">${picker}
      <label>Measure (mean)<select id="cy">${opts(measures, y)}</select></label>
      <label>For each<select id="cx">${opts(dims, x)}</select></label>
      <label>Split by<select id="cs"><option value="">(nothing)</option>${opts(splits, s)}</select></label></div>
    ${s ? `<div class="legend">${groups.map((grp, k) => `<span><i style="background:${color(k)}"></i>${esc(s.name)} = ${esc(grp)}</span>`).join("")}</div>` : ""}
    <div class="scroll"><svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">
      <text class="title" x="${L}" y="18">${esc(title)}</text>${g}</svg></div>
    <p class="note">Each point or bar is the mean over all rows of the file with that value. The Tables tab has the rows themselves.</p>`;
  $("#cy").onchange = e => { chartSel.y = e.target.value; renderPane(); };
  $("#cx").onchange = e => { chartSel.x = e.target.value; renderPane(); };
  $("#cs").onchange = e => { chartSel.s = e.target.value; renderPane(); };
  const tip = $("#tip");
  for (const el of pane.querySelectorAll("[data-tip]")) {
    el.onmousemove = ev => { tip.textContent = el.dataset.tip; tip.style.display = "block";
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - 270) + "px"; tip.style.top = ev.clientY + 14 + "px"; };
    el.onmouseleave = () => { tip.style.display = "none"; };
  }
}

let wasRunning = null;
async function refresh() {
  try { S = await api("/api/state"); } catch (e) { return; }
  const v = S.versions;
  $("#env").textContent = `Python ${v.python}, numpy ${v.numpy}, scipy ${v.scipy}, ${v.cpus} CPUs, commit ${v.commit}`;
  if (!selected) { select(S.experiments[0].id); return; }
  const now = S.running ? S.running.id : null;
  renderList();
  if (now !== wasRunning) { if (wasRunning && wasRunning === shownRun) details = null; renderDetail(); loadRun(); }
  else if (S.running && S.running.exp === selected) {
    const prev = runsOf(selected).find(r => r.state === "ok");
    $("#status").textContent = `Running for ${dur(S.running.seconds)}` + (prev ? ` (last time it took ${dur(prev.seconds)})` : "");
    if (shownRun === S.running.id && tab === "output") loadRun();
  }
  else if (shownRun && details && !details.meta) loadRun();
  wasRunning = now;
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Local dashboard for the experiments.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        stop()


if __name__ == "__main__":
    main()
