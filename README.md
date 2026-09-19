# OS-Induced Variability in Deep Learning Inference on Physical Machines

Direct extension of **Rahman et al. 2024, "On the Variability of AI-based Software
Systems Due to Environment Configurations" (arXiv:2408.02825)** — same statistical
protocol, but on **real physical machines** (dual-boot, no cloud CI / no VMs), on
OSes the original paper did not test (Windows 11, Fedora), with one fixed modern
deep-learning model (fine-tuned ResNet18 on CIFAR-10).

Full research context: see [CLAUDE.md](CLAUDE.md).

## Setup matrix

| Machine | Systems (dual-boot) | os_label |
|---|---|---|
| Researcher 1 | Ubuntu + Windows 11 | `ubuntu`, `win11_machine1` |
| Researcher 2 | Fedora + Windows 11 | `fedora`, `win11_machine2` |

Metrics per run: accuracy + macro F1 (**performance**), total time + ms/image
(**processing time**), peak RSS + CPU% (**expense** substitute, via psutil).
50 repetitions per system; Mann-Whitney U + Cliff's delta, significant only when
p < 0.05 **and** effect size is non-negligible — identical to the reference paper.

## Environment setup (identical on every system)

Use **Python 3.12.x** everywhere. CPU-only PyTorch on all machines (even with a
GPU present) — we compare OSes, not GPU drivers.

**Linux (Ubuntu / Fedora):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Windows 11 (PowerShell):**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Record the installed versions per system in [hardware_specs.md](hardware_specs.md)
(`python --version`, `pip show torch`).

## Workflow

### 1. One-time model/data preparation (reference machine, Ubuntu only)

```bash
python src/model_setup.py
```

Downloads CIFAR-10, fine-tunes the ResNet18 head (seed 42), and writes
`data/model_weights.pt`, `data/test_set.pt`, `data/setup_manifest.json`.

**Then copy the whole `data/` directory to every other machine/OS partition**
(USB drive or shared partition — not the network at measurement time). Transfer it
as a **single compressed archive** (`tar czf data.tar.gz data/` / extract on the
target) so no transfer tool can alter file contents (e.g., line-ending conversion
of `setup_manifest.json`). After extraction on EACH system, verify the SHA-256
hashes against `setup_manifest.json` (`sha256sum data/model_weights.pt data/test_set.pt`
on Linux, `Get-FileHash` on Windows) and record the check in
[hardware_specs.md](hardware_specs.md). Never re-run model_setup per OS — all
systems must use byte-identical weights and test data.

### 2. Quick sanity check on each system (3 runs)

```bash
python src/run_experiment.py --os-label ubuntu --dry-run
```

**Gate before the full runs.** Run `--dry-run` on ALL FOUR systems, collect the
four CSVs into `results/raw/`, then check them automatically:

```bash
python src/check_consistency.py --dry-run
# exit 0 = all 4 systems agree, safe to start the 50-run experiments
# exit 1 = mismatch, must be fixed first
# exit 2 = fewer than 4 systems reported in yet — do not start
```

Dry-runs write to `results/raw/*_raw.dryrun.csv`, kept separate from the full-run
files so the 3 warm-up samples can never be mixed into the 50-run distributions.
`run_experiment.py` also refuses to write over an existing results file unless
you pass `--append`.

It verifies accuracy/F1 are constant within each system **and identical across
all four**, plus that the controlled constants (threads, batch size, test-set
size, device) never drift. `run_experiment.py` also aborts on the spot if
accuracy varies between runs on one system. Any accuracy difference signals a
real problem (corrupted `data/` copy, mismatched torch version) to fix before
spending hours on the 4 × 50 runs. Timing/memory differences are expected —
that is the phenomenon under study.

### Checklist — before committing to the full 50 runs

Every box must be ticked before starting; the 4 × 50 runs cost hours per machine
and cannot be salvaged after the fact if a precondition was wrong.

- [ ] Contacted the original paper's co-author to confirm no similar work is
      already in progress
- [ ] Checked the publication status of the reference paper (preprint vs formally
      published) on Google Scholar — cite the published version if one exists
- [ ] `model_setup.py` was run on Ubuntu **only**, and `data/` was transferred to
      the other three systems as a `tar czf` archive
- [ ] SHA-256 hashes verified against `setup_manifest.json` after extraction on
      all four systems
- [ ] `--dry-run` executed on all four systems, with accuracy/F1 **byte-identical**
      across every one of them (`check_consistency.py --dry-run` exits 0)
- [ ] Physical core count ≥ 4 confirmed on both machines (so `threads=4` is a
      valid pin, not oversubscription)
- [ ] [hardware_specs.md](hardware_specs.md) fully filled in (CPU model, RAM, core
      counts, software versions — no TODO placeholders left)
- [x] [reference_baseline.md](reference_baseline.md) complete (Threats to Validity
      + Citation sections added)

### 3. Full experiment on each system (50 runs, ~overnight-friendly)

```bash
python src/run_experiment.py --os-label ubuntu           # machine 1, Ubuntu boot
python src/run_experiment.py --os-label win11_machine1   # machine 1, Windows boot
python src/run_experiment.py --os-label fedora           # machine 2, Fedora boot
python src/run_experiment.py --os-label win11_machine2   # machine 2, Windows boot
```

Writes `results/raw/{ubuntu_raw,win11_m1_raw,fedora_raw,win11_m2_raw}.csv`, then
confirm with `python src/check_consistency.py` (no `--dry-run` this time).
Commit/copy each CSV back into this repo (GitHub is for storing code and results
only — **never run measurements on CI**).

**Measurement hygiene (do this on every system):** plugged into AC power, high/
performance power plan, close other applications, let the machine idle a few
minutes first, don't touch it during the run. The 2 s cooldown between trials and
the 2 warm-up batches per trial are built in.

### 4. Analysis (any machine, once all four CSVs are in results/raw/)

```bash
python src/analyze_results.py
```

Produces in `results/analysis/`:
- `comparison_results.csv` — full stats for every comparison × metric
- `summary_table.md` — Table 3/4-style classification (Zero / Non-zero insignificant /
  Non-zero significant), headline percentages, a **side-by-side comparison with
  Rahman et al. 2024**, the ready-to-paste quotable sentence for the abstract, and a
  **directional interpretation per comparison** (printed to stdout as well): whether
  our timing result reproduces, reverses, or fails to find their Linux-vs-Windows
  gap. The wording for all three outcomes is fixed in the source code, so the
  abstract cannot be phrased after the fact in favour of a preferred result — a
  reversal would be a headline finding (virtualization, not the OS, drove their gap),
  not a failed experiment
- `pct_change.png` — poster-ready percentage-change chart

The target figures we compare against live in
[reference_baseline.md](reference_baseline.md): the reference paper's full
Tables 3/4/5/7/9–10, the direction of its effects, and the two caveats the
abstract must state (they counted 30 projects vs our 4 OS comparisons; their
projects trained stochastically while we run fixed-weight inference).

Our headline benchmark is their **Linux vs Windows** column — processing time
significant in 30/30 projects (100%), performance in only 6/30 (20%), with
Linux consistently faster. Whether our physical machines reproduce that
**direction** is a stronger result than matching the percentages.

## Repo layout

```
src/
  model_setup.py       one-time model + test-set preparation (network used HERE only)
  run_trial.py         one measured inference trial -> one JSON result line
  run_experiment.py    N repetitions in fresh subprocesses -> raw CSV
  check_consistency.py cross-system gate: accuracy identical + constants held
  analyze_results.py   Mann-Whitney U + Cliff's delta + tables + chart
results/raw/           raw per-system CSVs (committed)
results/analysis/      analysis outputs (committed)
poster/                poster-ready figures/tables
hardware_specs.md      exact hardware + software versions per machine
reference_baseline.md  the Rahman et al. 2024 figures we compare against
```

## Validity notes

- `src/run_trial.py` is byte-identical on all systems; only `--os-label` differs.
- Fixed seeds (42), fixed batch size (32), CPU-only.
- **Torch threads are pinned to 4 on every system as a deliberate controlled
  constant** — not "whatever each machine defaults to". PyTorch's default thread
  count follows the detected core count, which would silently differ if the two
  machines' CPUs differ, confounding OS effects with parallelism effects. Pinning
  requires 4 ≤ physical cores on BOTH machines — confirm this when filling in
  [hardware_specs.md](hardware_specs.md) and record the actual core counts there.
- No network calls during timing; model and data are loaded before the timed window.
- Peak memory is sampled cross-platform the same way on all OSes (psutil RSS sampler).

## Reference

Rahman, M., Khatoonabadi, S., Abdellatif, A., Samaana, H., & Shihab, E. (2024).
*On the Variability of AI-based Software Systems Due to Environment Configurations.*
arXiv:2408.02825. https://arxiv.org/abs/2408.02825

```bibtex
@misc{rahman2024variability,
  title         = {On the Variability of AI-based Software Systems Due to Environment Configurations},
  author        = {Rahman, M. and Khatoonabadi, S. and Abdellatif, A. and Samaana, H. and Shihab, E.},
  year          = {2024},
  eprint        = {2408.02825},
  archivePrefix = {arXiv}
}








       

















