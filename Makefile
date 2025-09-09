# LLC Sampler Benchmark Makefile

.PHONY: help run diag test clean artifacts promote-readme

# Default target
help:
	@echo "LLC Sampler Benchmark - Available targets:"
	@echo ""
	@echo "  run        - Run full benchmark with default config"
	@echo "  run-save   - Run with plot saving enabled"
	@echo "  diag       - Run diagnostics-only (quick test config)"
	@echo "  test       - Run minimal test configuration"
	@echo "  sweep      - Run parameter sweep experiments"
	@echo "  clean      - Clean generated artifacts"
	@echo "  artifacts  - Create artifacts directory"
	@echo "  promote-readme - Copy plots from latest run to README assets"
	@echo ""
	@echo "Configuration:"
	@echo "  PYTHON     - Python command (default: uv run python)"
	@echo "  CONFIG     - Additional config overrides"

# Configuration
PYTHON ?= uv run python
MAIN_PY = main.py

# Run full benchmark with default configuration
run:
	$(PYTHON) $(MAIN_PY)

# Run with plot saving enabled
run-save:
	$(PYTHON) -c "from main import main, Config; main(Config(save_plots=True, save_manifest=True, save_readme_snippet=True))"

# Run diagnostics with quick test config
diag:
	$(PYTHON) -c "from main import main, TEST_CFG, replace; main(replace(TEST_CFG, save_plots=True, save_manifest=True, save_readme_snippet=True))"

# Run minimal test (fastest)
test:
	$(PYTHON) -c "from main import main, TEST_CFG; main(TEST_CFG)"

# Run parameter sweep
sweep:
	$(PYTHON) $(MAIN_PY) sweep

# Create artifacts directory
artifacts:
	mkdir -p artifacts
	@echo "Artifacts directory created"

# Clean generated artifacts and outputs
clean:
	rm -rf artifacts/*
	rm -f *.png *.jpg *.jpeg *.pdf
	rm -f llc_sweep_results.csv
	rm -rf __pycache__/
	rm -f *.pyc *.pyo
	@echo "Cleaned artifacts and temporary files"

# Development targets
dev-install:
	uv sync
	@echo "Dependencies installed"

dev-clean:
	rm -rf .venv/
	rm -rf __pycache__/
	rm -f uv.lock
	@echo "Development environment cleaned"

# Examples of running with specific configurations
example-small:
	$(PYTHON) -c "from main import main, Config; main(Config(n_data=1000, sgld_steps=500, hmc_draws=200, save_plots=True))"

example-deep:
	$(PYTHON) -c "from main import main, Config; main(Config(depth=3, target_params=50000, save_plots=True))"

example-mclmc-only:
	$(PYTHON) -c "from main import main, Config; main(Config(sampler='mclmc', save_plots=True))"

# Promote plots from latest run to README assets
promote-readme:
	$(PYTHON) scripts/promote_readme_images.py
	@git status --short assets/readme/