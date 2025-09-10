import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # LLC Sampler Diagnostics Viewer

    Interactive viewer for analyzing MCMC sampling results from the LLC (Local Log-Likelihood Curvature) experiments.
    This notebook loads saved data artifacts and regenerates diagnostic plots dynamically.
    """
    )
    return


@app.cell
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    from pathlib import Path
    import pandas as pd
    from datetime import datetime

    # Configure matplotlib for better plots
    plt.style.use("default")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["font.size"] = 10
    return Path, az, datetime, json, np, pd, plt


@app.cell
def _(Path):
    # List available runs
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        runs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()], reverse=True)
        run_options = [f"{r.name}" for r in runs]
        print(f"Found {len(runs)} runs:")
        for i, run in enumerate(runs[:10]):  # Show first 10
            print(f"  {i}: {run.name}")
    else:
        runs = []
        run_options = []
        print("No artifacts directory found. Run main.py with save_plots=True first.")
    return artifacts_dir, run_options


@app.cell
def _(mo, run_options):
    # Run selector
    if run_options:
        run_selector = mo.ui.dropdown(
            options=run_options,
            value=run_options[0] if run_options else None,
            label="Select run to analyze:",
        )
    else:
        run_selector = mo.ui.dropdown(options=[], label="No runs available")

    run_selector
    return (run_selector,)


@app.cell
def _(artifacts_dir, run_selector):
    # Load selected run data
    if run_selector.value:
        selected_run = artifacts_dir / run_selector.value

        # List available files
        files = list(selected_run.glob("*"))

        print(f"Selected run: {selected_run.name}")
        print("Files in run directory:")
        for file_obj in sorted(files):
            print(f"  {file_obj.name}")

        run_info = {
            "path": selected_run,
            "files": {file_obj.name: file_obj for file_obj in files},
        }
    else:
        run_info = None
    return (run_info,)


@app.cell
def _(json, np, run_info):
    # Load configuration and L0
    if run_info and "config.json" in run_info["files"]:
        with open(run_info["files"]["config.json"]) as config_file:
            config = json.load(config_file)

        # Extract key parameters for L0 reconstruction
        n_data = config["n_data"]
        beta_mode = config["beta_mode"]
        beta0 = config["beta0"]

        if beta_mode == "1_over_log_n":
            beta = beta0 / np.log(n_data)
        else:
            beta = beta0

        # Load L0 if available
        L0 = 0.0
        if "L0.txt" in run_info["files"]:
            with open(run_info["files"]["L0.txt"]) as l0_file:
                L0 = float(l0_file.read().strip())

        print("Configuration loaded:")
        print(f"  n_data: {n_data}")
        print(f"  beta: {beta:.6f}")
        print(f"  L0: {L0:.6f}")
        print(f"  Model: {config['depth']}-layer MLP")
        print("  Samplers: sgld, hmc, mclmc")

    else:
        print("No configuration found.")
    return L0, beta, config, n_data


@app.cell
def _(json, run_info):
    # Load run-level metrics
    samplers = ["sgld", "hmc", "mclmc"]
    metrics = {}

    if run_info:
        # Load run-level metrics.json
        metrics_file = "metrics.json"
        if metrics_file in run_info["files"]:
            with open(run_info["files"][metrics_file]) as metrics_file_obj:
                all_metrics = json.load(metrics_file_obj)
                # Extract per-sampler metrics from run-level structure
                for sampler_name in samplers:
                    sampler_data = {}
                    # Extract sampler-specific metrics (e.g., sgld_llc_mean -> llc_mean)
                    for key, value in all_metrics.items():
                        if key.startswith(f"{sampler_name}_"):
                            clean_key = key.replace(f"{sampler_name}_", "")
                            sampler_data[clean_key] = value
                    if sampler_data:
                        metrics[sampler_name] = sampler_data

        print("Metrics loaded:")
        for sampler_key, data in metrics.items():
            print(
                f"  {sampler_key.upper()}: LLC={data.get('llc_mean', 'N/A'):.4f}, ESS={data.get('ess', 'N/A'):.1f}"
            )
    return metrics, samplers


@app.cell
def _(az, run_info, samplers):
    # Load ArviZ InferenceData for L_n histories
    idata = {}

    if run_info:
        for samp in samplers:
            nc_file = f"{samp}_L.nc"
            if nc_file in run_info["files"]:
                try:
                    idata[samp] = az.from_netcdf(str(run_info["files"][nc_file]))
                    chains, n_draws = idata[samp].posterior["L"].shape
                    print(f"Loaded {samp.upper()}: {chains} chains × {n_draws} draws")
                except Exception as e:
                    print(f"Error loading {samp}: {e}")
    return (idata,)


@app.cell
def _(mo):
    # Plot selection
    plot_types = [
        "Running LLC",
        "L_n Traces",
        "Autocorrelation",
        "Effective Sample Size",
        "R-hat Diagnostics",
        "Summary Table",
    ]

    plot_selector = mo.ui.multiselect(
        options=plot_types,
        value=["Running LLC", "Summary Table"],
        label="Select plots to display:",
    )

    plot_selector
    return (plot_selector,)


@app.cell
def _(L0, beta, idata, n_data, np, plot_selector, plt, samplers):
    # Running LLC reconstruction and plotting
    def compute_running_llc(idata_sampler):
        """Reconstruct running LLC from L_n histories"""
        if idata_sampler is None:
            return None, None

        L = np.array(idata_sampler.posterior["L"])  # (chains, draws)
        chains, n_samples = L.shape

        # Running means per chain
        cumsum = np.cumsum(L, axis=1)
        indices = np.arange(1, n_samples + 1)
        cmeans = cumsum / indices[None, :]  # Broadcasting

        # Convert to LLC estimates
        lam_chains = n_data * beta * (cmeans - L0)

        # Pooled estimate across chains
        L_pooled = np.mean(L, axis=0)
        cumsum_pooled = np.cumsum(L_pooled)
        cmean_pooled = cumsum_pooled / indices
        lam_pooled = n_data * beta * (cmean_pooled - L0)

        return lam_chains, lam_pooled

    if "Running LLC" in plot_selector.value and idata:
        plt.figure(figsize=(12, 4))

        for idx, samp_name in enumerate(samplers):
            if samp_name in idata:
                plt.subplot(1, 3, idx + 1)

                lam_chains, lam_pooled = compute_running_llc(idata[samp_name])
                if lam_chains is not None:
                    n_evals = lam_chains.shape[1]
                    x = np.arange(1, n_evals + 1)

                    # Plot individual chains
                    for c in range(lam_chains.shape[0]):
                        plt.plot(x, lam_chains[c], alpha=0.4, lw=1)

                    # Plot pooled estimate
                    plt.plot(x, lam_pooled, "k-", lw=2, label="Pooled")

                    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                    plt.xlabel("L_n evaluations")
                    plt.ylabel("λ̂_t")
                    plt.title(f"{samp_name.upper()}: Running LLC")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(az, idata, plot_selector, plt, samplers):
    # L_n Trace plots
    if "L_n Traces" in plot_selector.value and idata:
        for samp_trace in samplers:
            if samp_trace in idata:
                fig, axes = plt.subplots(figsize=(10, 6))
                az.plot_trace(idata[samp_trace], var_names=["L"], axes=axes)
                plt.suptitle(f"{samp_trace.upper()}: L_n Traces", y=1.02)
                plt.tight_layout()
                plt.show()
    return


@app.cell
def _(az, idata, plot_selector, plt, samplers):
    # Autocorrelation plots
    if "Autocorrelation" in plot_selector.value and idata:
        plt.figure(figsize=(15, 4))

        for idx2, samp_acf in enumerate(samplers):
            if samp_acf in idata:
                plt.subplot(1, 3, idx2 + 1)
                try:
                    az.plot_autocorr(idata[samp_acf], var_names=["L"], ax=plt.gca())
                    plt.title(f"{samp_acf.upper()}: L_n ACF")
                except Exception as e:
                    plt.text(
                        0.5,
                        0.5,
                        f"Error: {e}",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title(f"{samp_acf.upper()}: L_n ACF (Error)")

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(az, idata, plot_selector, plt, samplers):
    # Effective Sample Size plots
    if "Effective Sample Size" in plot_selector.value and idata:
        plt.figure(figsize=(15, 4))

        for idx3, samp_ess in enumerate(samplers):
            if samp_ess in idata:
                plt.subplot(1, 3, idx3 + 1)
                try:
                    az.plot_ess(idata[samp_ess], var_names=["L"], ax=plt.gca())
                    plt.title(f"{samp_ess.upper()}: ESS(L_n)")
                except Exception as e:
                    plt.text(
                        0.5,
                        0.5,
                        f"Error: {e}",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title(f"{samp_ess.upper()}: ESS (Error)")

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(az, idata, plot_selector, plt, samplers):
    # R-hat diagnostics
    if "R-hat Diagnostics" in plot_selector.value and idata:
        plt.figure(figsize=(15, 4))

        for idx4, samp_rhat in enumerate(samplers):
            if samp_rhat in idata:
                plt.subplot(1, 3, idx4 + 1)
                try:
                    az.plot_forest(
                        idata[samp_rhat], var_names=["L"], r_hat=True, ax=plt.gca()
                    )
                    plt.title(f"{samp_rhat.upper()}: R̂(L_n)")
                except Exception as e:
                    plt.text(
                        0.5,
                        0.5,
                        f"Error: {e}",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title(f"{samp_rhat.upper()}: R̂ (Error)")

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(metrics, mo, pd, plot_selector, samplers):
    # Summary table
    if "Summary Table" in plot_selector.value and metrics:
        # Build summary dataframe
        summary_data = []

        for samp_summary in samplers:
            if samp_summary in metrics:
                m = metrics[samp_summary]
                row = {
                    "Sampler": samp_summary.upper(),
                    "LLC Mean": f"{m.get('llc_mean', 0):.4f}",
                    "LLC SE": f"{m.get('llc_se', 0):.4f}",
                    "ESS": f"{m.get('ess', 0):.1f}",
                    "Warmup (s)": f"{m.get('timing_warmup', 0):.2f}",
                    "Sampling (s)": f"{m.get('timing_sampling', 0):.2f}",
                    "Gradient Work": f"{m.get('n_leapfrog_grads', m.get('n_steps', 0))}",
                }

                # Add sampler-specific metrics
                if samp_summary == "hmc" and "mean_acceptance" in m:
                    row["Acceptance"] = f"{m['mean_acceptance']:.3f}"
                elif samp_summary in ["sgld", "mclmc"]:
                    row["Acceptance"] = "N/A"

                summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            mo.ui.table(summary_df, selection=None)
        else:
            mo.md("No metrics data available.")
    return (summary_df,)


@app.cell
def _(mo):
    # Export section
    mo.md("""
    ## Export Options

    Generate reports or export data for further analysis.
    """)
    return


@app.cell
def _(mo):
    # Export controls
    export_markdown = mo.ui.button(
        label="📄 Export Markdown Report",
        on_click=lambda: print("Markdown export triggered"),
    )

    mo.hstack([export_markdown], justify="start")
    return (export_markdown,)


@app.cell
def _(L0, beta, config, datetime, export_markdown, run_info, summary_df):
    # Export markdown report functionality
    def export_markdown_report():
        if not run_info:
            print("No run selected for export.")
            return

        report_dir = run_info["path"] / "viewer_report"
        report_dir.mkdir(exist_ok=True)

        # Generate markdown content
        md_lines = [
            "# LLC Diagnostics Report",
            "",
            f"**Run:** `{run_info['path'].name}`",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            f"- Model: {config.get('depth', 'N/A')}-layer MLP",
            f"- Data points: {config.get('n_data', 'N/A')}",
            f"- Beta: {beta:.6f}",
            f"- L0: {L0:.6f}",
            "",
            "## Results Summary",
        ]

        # Add metrics table
        if "summary_df" in globals():
            md_lines.append("```")
            md_lines.append(summary_df.to_string(index=False))
            md_lines.append("```")

        md_content = "\n".join(md_lines)

        # Save report
        report_path = report_dir / "report.md"
        with open(report_path, "w") as report_file:
            report_file.write(md_content)

        print(f"✅ Markdown report exported to: {report_path}")
        return str(report_path)

    # Trigger export when button is clicked
    if export_markdown.value:
        export_markdown_report()
    return


if __name__ == "__main__":
    app.run()
