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
(USB drive or shared partition — not the network at measurement time). Verify the
SHA-256 hashes from `setup_manifest.json` match on each system. Never re-run
model_setup per OS — all systems must use byte-identical weights and test data.

### 2. Quick sanity check on each system (3 runs)

```bash
python src/run_experiment.py --os-label ubuntu --dry-run
```

### 3. Full experiment on each system (50 runs, ~overnight-friendly)

```bash
python src/run_experiment.py --os-label ubuntu           # machine 1, Ubuntu boot
python src/run_experiment.py --os-label win11_machine1   # machine 1, Windows boot
python src/run_experiment.py --os-label fedora           # machine 2, Fedora boot
python src/run_experiment.py --os-label win11_machine2   # machine 2, Windows boot
```

Writes `results/raw/{ubuntu_raw,win11_m1_raw,fedora_raw,win11_m2_raw}.csv`.
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
- `summary_table.md` — Table 3/4-style classification (Zero / Non-zero insignificant / Non-zero significant) + headline percentages
- `pct_change.png` — poster-ready percentage-change chart

## Repo layout

```
src/
  model_setup.py       one-time model + test-set preparation (network used HERE only)
  run_trial.py         one measured inference trial -> one JSON result line
  run_experiment.py    N repetitions in fresh subprocesses -> raw CSV
  analyze_results.py   Mann-Whitney U + Cliff's delta + tables + chart
results/raw/           raw per-system CSVs (committed)
results/analysis/      analysis outputs (committed)
poster/                poster-ready figures/tables
hardware_specs.md      exact hardware + software versions per machine
```

## Validity notes

- `src/run_trial.py` is byte-identical on all systems; only `--os-label` differs.
- Fixed seeds (42), fixed torch threads (4), fixed batch size (32), CPU-only.
- No network calls during timing; model and data are loaded before the timed window.
- Peak memory is sampled cross-platform the same way on all OSes (psutil RSS sampler).
