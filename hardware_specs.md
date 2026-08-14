# Hardware & Software Specifications

Fill in precisely for each machine — this documents the "identical hardware within
machine / near-identical across machines" claim used by the statistical comparisons.

Commands to gather the info:

- **Linux:** `lscpu`, `free -h`, `sudo dmidecode -t memory | grep -E "Speed|Type:"`, `lspci | grep -i vga`, `uname -r`, `cat /etc/os-release`
- **Windows 11 (PowerShell):** `Get-ComputerInfo | Select CsProcessors,CsTotalPhysicalMemory,OsName,OsVersion,OsBuildNumber`, `Get-CimInstance Win32_VideoController | Select Name`

## Machine 1 (Researcher 1 — Ubuntu + Windows 11 dual-boot)

| Item | Value |
|---|---|
| CPU model | _TODO (e.g., Intel Core i7-12700H)_ |
| Physical / logical cores | _TODO_ |
| RAM (size, type, speed) | _TODO_ |
| Storage (model, SSD/NVMe) | _TODO_ |
| GPU (present? unused — CPU-only runs) | _TODO_ |
| Ubuntu version / kernel | _TODO (e.g., Ubuntu 24.04.1, kernel 6.8.0-xx)_ |
| Windows 11 version / build | _TODO (e.g., 23H2, build 22631.xxxx)_ |
| Power plan during runs (both OSes) | _TODO (e.g., Performance, AC power)_ |
| Python version (ubuntu / win11) | _TODO_ |
| torch / torchvision version (both) | _TODO (must match requirements.txt)_ |

## Machine 2 (Researcher 2 — Fedora + Windows 11 dual-boot)

| Item | Value |
|---|---|
| CPU model | _TODO_ |
| Physical / logical cores | _TODO_ |
| RAM (size, type, speed) | _TODO_ |
| Storage (model, SSD/NVMe) | _TODO_ |
| GPU (present? unused — CPU-only runs) | _TODO_ |
| Fedora version / kernel | _TODO_ |
| Windows 11 version / build | _TODO_ |
| Power plan during runs (both OSes) | _TODO_ |
| Python version (fedora / win11) | _TODO_ |
| torch / torchvision version (both) | _TODO_ |

## Cross-machine comparability note

_TODO: one paragraph stating how close the two machines are (CPU generation/class,
RAM amount) and any known differences — needed for interpreting comparisons 3 and 4
(Win11 vs Win11, Ubuntu vs Fedora), which cross hardware._

**Controlled constant — torch threads = 4 on every system.** This is a deliberate
methodological choice, not a per-machine default: it decouples the OS comparison
from CPU-core-count differences between the two machines. Confirm here that both
machines have ≥ 4 physical cores: machine 1 = _TODO_, machine 2 = _TODO_.

## data/ integrity check

Paste `setup_manifest.json` hashes as verified on each system:

| System | model_weights.pt sha256 matches | test_set.pt sha256 matches |
|---|---|---|
| ubuntu | _TODO_ | _TODO_ |
| win11_machine1 | _TODO_ | _TODO_ |
| fedora | _TODO_ | _TODO_ |
| win11_machine2 | _TODO_ | _TODO_ |
