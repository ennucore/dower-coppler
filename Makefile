PYTHON ?= ../.venv/bin/python
DATA ?= /Volumes/Extreme SSD/data/ultratrace_Kenny_lev_2026-04-27_16:42:18.h5
N_ACQ ?= 8
ANGULAR_CF_N_ACQ ?= 1

.PHONY: all synthetic research-sweep kenny quick-kenny kenny-all-variations angular-cf-sweep

all: synthetic research-sweep kenny

synthetic:
	CATERPILLAR_DISABLE_METAL=1 MPLCONFIGDIR=outputs/.mplconfig $(PYTHON) scripts/synthetic_phase_ls.py --output-dir outputs/synthetic

research-sweep:
	CATERPILLAR_DISABLE_METAL=1 MPLCONFIGDIR=outputs/.mplconfig $(PYTHON) scripts/phase_ls_research_sweep.py --output-dir outputs/research_sweep

quick-kenny:
	CATERPILLAR_DISABLE_METAL=1 MPLCONFIGDIR=outputs/.mplconfig $(PYTHON) scripts/kenny_042716_report.py --data "$(DATA)" --output outputs/kenny_042716/quick_kenny_042716_tmas_phase_ls_comparison.pdf --n-acq 1 --no-include-differential

kenny:
	CATERPILLAR_DISABLE_METAL=1 MPLCONFIGDIR=outputs/.mplconfig $(PYTHON) scripts/kenny_042716_report.py --config configs/kenny_042716.json --data "$(DATA)" --n-acq $(N_ACQ) --include-differential

kenny-all-variations:
	CATERPILLAR_DISABLE_METAL=1 MPLCONFIGDIR=outputs/.mplconfig $(PYTHON) scripts/kenny_042716_report.py --config configs/kenny_042716.json --data "$(DATA)" --output outputs/kenny_042716/kenny_042716_all_variations.pdf --n-acq $(N_ACQ) --no-include-differential

angular-cf-sweep:
	CATERPILLAR_DISABLE_METAL=1 MPLCONFIGDIR=outputs/.mplconfig $(PYTHON) scripts/angular_cf_parameter_sweep.py --data "$(DATA)" --output outputs/kenny_042716/kenny_042716_angular_cf_parameter_sweep.pdf --n-acq $(ANGULAR_CF_N_ACQ)
