# Instructions for Claude Code in this repository

This is Lucas Allen's research code for discrete neural networks (weights are
boxes found by linear programming; neurons are added when an LP is
infeasible). Read `docs/Handoff_Notes.md` first: it has the state of the work,
the decisions, and the path we are following. Results are in
`experiments/RESULTS.md` (about 2,500 lines): read only the section you need
and its Summary entry, never the whole file.

## Where we are going

Lucas's long-term goal is programs that run locally, built from discrete
neural networks: first an agent that plays Doom, later Lean 4 proofs and a
chat bot. The path to Doom is in `docs/Handoff_Notes.md` section 4. Steps 1-2
are done (experiments 24-27); experiments 28-32 began steps 3 and 5. Every experiment should move us along that
path; if a proposed one does not, say so and suggest one that does.

Work moves forward in **batches of four experiments**: designed, run and
written up together, then reported and committed as one pull request.

## This machine

All experiments (exp1 onwards) have run on this machine: Arch Linux, Intel
i7-10510U, 4 cores / 8 threads at 1.8 GHz, 7.6 GiB RAM, 2 GiB swap; Python
3.14.10, numpy 2.5.4, scipy 1.18.1. Results have been bit-for-bit
reproducible.

- `Pool()` uses all 8 threads. Keep memory use under about 6 GiB; if a run
  gets near that, pass a smaller worker count (results do not depend on it,
  because seeds are fixed per task).

## How a batch runs

1. **Plan.** Propose the next four experiments in a few lines (what question
   each answers, which path step it serves, rough run time on this machine)
   and wait for Lucas's OK before writing code.
2. **Branch.** One branch per batch, `claude/batch-<short-name>`, from an
   up-to-date `main`. Never commit to `main`.
3. **Write** each experiment as `experiments/exp<N>_<name>.py`, numbering on
   from the last one. Follow the dashboard conventions in the handoff notes
   (docstring title, `harness.write_csv`, module-level workers,
   `main()` under `if __name__ == "__main__"`), and the two pitfalls:
   read committed CSVs from `experiments/results/`, not `harness.RESULTS`;
   pass patched tables explicitly instead of relying on default arguments.
   Add tests for new library code; keep `dnn.py` unchanged unless Lucas asks.
4. **Small run first**: call `main()` with small sizes and check it works
   with `DNN_RESULTS_DIR` set to a temporary folder (as the dashboard does).
5. **Full runs**, one experiment at a time:
   `nice python experiments/exp<N>_<name>.py > runs/exp<N>.log 2>&1`
   (`runs/` is git-ignored). Look only at the end of the log (`tail -40`), not
   all of it. If a run will take more than about an hour, say so first.
6. **Check every CSV by script**, never by reading it: row count, expected
   rows per setting, 16-character sha256, and exact reproduction of any rows
   that repeat an earlier experiment's settings (timing columns ignored).
   A failed reproduction is a bug to find before writing up.
7. **Write up** each experiment as a `RESULTS.md` section (setup, tables,
   findings, with h stated) plus a Summary entry, matching earlier sections.
   State negative results plainly. Say when a result depends on building
   blocks Claude designed. Commit each experiment's script, CSV and write-up
   on the batch branch as it is finished.
8. **After the fourth experiment**: run the tests (`python -m pytest -q`),
   update `docs/Handoff_Notes.md` (state, next steps, open problems), push
   the branch, open one pull request for the batch, and report to Lucas:
   for each experiment, what was tested, the key numbers and what they mean
   in plain words; then one line, "Doom:", saying which path steps the batch
   advanced and what now blocks step 7 (ViZDoom); then the four experiments
   you suggest next.

Stop and ask Lucas when a result changes the plan (say so straight away,
without waiting for the end of the batch), when a design choice is his to make
(anything in the notes' "Decisions" section), or before anything hard to
undo.

## Conventions

- Fixed seeds; CSVs in `experiments/results/`; experiments that reuse
  earlier layouts must reproduce the earlier rows exactly.
- Use `.venv/bin/python` if the system python has no numpy. Commit as
  Claude with per-commit environment variables (`GIT_AUTHOR_NAME`,
  `GIT_COMMITTER_NAME` and the matching `_EMAIL` vars) if the machine has no
  git identity; do not change git config. Do not put email addresses in files or commit messages.
- Lucas's usage is limited: keep replies short and plain, keep command output
  trimmed (`tail`, `grep`, `head`), and do not read large files whole.
- Another Claude session (in Lucas's claude.ai project) may update this file
  and the notes through pull requests. At the start of each session, and when
  Lucas says so, `git pull` and re-read this file.
