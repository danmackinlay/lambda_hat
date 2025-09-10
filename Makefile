# LLC Sampler Benchmark Makefile

.PHONY: help run diag test clean artifacts promote-readme modal-deploy modal-ls modal-get modal-get-all modal-sweep modal-stop modal-clean modal-rm

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
	@echo "Modal deployment targets:"
	@echo "  modal-deploy   - Deploy Modal app (uv run modal deploy modal_app.py)"
	@echo "  modal-sweep    - Run sweep on Modal backend"
	@echo "  modal-ls       - List Modal volume contents"
	@echo "  modal-get      - Download specific run (requires SRC and DST vars)"
	@echo "  modal-get-all  - Download all Modal artifacts"
	@echo "  modal-rm       - Remove specific run (requires RMPATH var)"
	@echo "  modal-stop     - Stop Modal app"
	@echo "  modal-clean    - Clean Modal volume (interactive confirmation)"
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
	$(PYTHON) -c "from main import main; from llc.config import Config; main(Config(save_plots=True, save_manifest=True, save_readme_snippet=True))"

# Run diagnostics with quick test config
diag:
	$(PYTHON) -c "from main import main; from dataclasses import replace; from llc.config import TEST_CFG; main(replace(TEST_CFG, save_plots=True, save_manifest=True, save_readme_snippet=True))"

# Run minimal test (fastest)
test:
	$(PYTHON) -c "from main import main; from llc.config import TEST_CFG; main(TEST_CFG)"

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
	$(PYTHON) -c "from main import main; from llc.config import Config; main(Config(n_data=1000, sgld_steps=500, hmc_draws=200, save_plots=True))"

example-deep:
	$(PYTHON) -c "from main import main; from llc.config import Config; main(Config(depth=3, target_params=50000, save_plots=True))"

example-mclmc-only:
	$(PYTHON) -c "from main import main; from llc.config import Config; main(Config(sampler='mclmc', save_plots=True))"

# Promote plots from latest run to README assets
promote-readme:
	$(PYTHON) scripts/promote_readme_images.py
	@git status --short assets/readme/

# Modal deployment and management targets (CLI via uv)
modal-deploy:
	uv run modal deploy modal_app.py
	@echo "Modal app deployed successfully"

modal-ls:
	uv run modal volume ls llc-artifacts

modal-get:
	# Usage: make modal-get SRC=/artifacts/20250909-172233 DST=./artifacts/20250909-172233
	@test -n "$(SRC)" && test -n "$(DST)" || (echo "Usage: make modal-get SRC=/artifacts/run-dir DST=./artifacts/run-dir"; exit 1)
	uv run modal volume get llc-artifacts $(SRC) $(DST)
	@echo "Downloaded $(SRC) to $(DST)"

modal-get-all:
	uv run modal volume get llc-artifacts /artifacts ./artifacts
	@echo "Downloaded all Modal artifacts to ./artifacts"

modal-sweep:
	$(PYTHON) main.py sweep --backend=modal --save-artifacts
	@echo "Modal sweep completed"

modal-stop:
	uv run modal app stop llc-experiments
	@echo "Modal app stopped"

modal-clean:
	# Warning: This removes ALL artifacts from the Modal volume
	@echo "This will delete ALL artifacts in the Modal volume. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	uv run modal volume rm llc-artifacts /artifacts
	@echo "Modal volume cleaned"

modal-rm:
	# Usage: make modal-rm RMPATH=/artifacts/20250909-172233
	@test -n "$(RMPATH)" || (echo "Usage: make modal-rm RMPATH=/artifacts/<run-id>"; exit 1)
	uv run modal volume rm llc-artifacts $(RMPATH)
	@echo "Removed $(RMPATH) from Modal volume"