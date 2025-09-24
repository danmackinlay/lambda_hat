# llc/artifacts.py
"""Artifact management and gallery generation utilities"""

import os
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np

# ArviZ is heavy; import lazily inside the functions that need it
from pathlib import Path


def save_config(run_dir: str, cfg) -> str:
    """Save configuration to JSON file"""
    config_path = f"{run_dir}/config.json"

    # Convert config to dict, handling special types
    config_dict = {}
    for key, value in cfg.__dict__.items():
        if isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()
        elif hasattr(value, "__dict__"):  # nested dataclass
            config_dict[key] = value.__dict__
        else:
            try:
                json.dumps(value)  # Test JSON serializability
                config_dict[key] = value
            except (TypeError, ValueError):
                config_dict[key] = str(value)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return config_path


def save_metrics(run_dir: str, metrics: Dict[str, Any]) -> str:
    """Save comprehensive metrics to JSON"""
    metrics_path = f"{run_dir}/metrics.json"

    # Convert numpy types to Python natives for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj

    clean_metrics = convert_numpy(metrics)

    with open(metrics_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)

    return metrics_path


# NOTE: create_manifest has been removed - use generate_gallery_html instead
# The HTML gallery provides a lightweight preview without manifest machinery


def generate_gallery_html(run_dir: str, cfg, metrics: Dict[str, Any]) -> str:
    """Generate HTML gallery index page"""

    # Find all PNG files in the directory
    png_files = [f for f in os.listdir(run_dir) if f.endswith(".png")]
    png_files.sort()

    # Group plots by sampler - determine from metrics instead of cfg
    samplers_run = [
        s for s in ("sgld", "sghmc", "hmc", "mclmc") if f"{s}_llc_mean" in metrics
    ]
    samplers = (
        samplers_run if samplers_run else ["sgld", "sghmc", "hmc", "mclmc"]
    )  # fallback
    plot_groups = {sampler: [] for sampler in samplers}

    for png_file in png_files:
        for sampler in samplers:
            if sampler.lower() in png_file.lower():
                plot_groups[sampler].append(png_file)
                break

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLC Analysis Results - {os.path.basename(run_dir)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-item {{
            background-color: #fff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .plot-item img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .plot-item h3 {{
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.1em;
        }}
        .sampler-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
        .note {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Local Learning Coefficient Analysis</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Run ID:</strong> {os.path.basename(run_dir)}</p>

        <h2>Summary Results</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Sampler</th>
                    <th>LLC Mean</th>
                    <th>LLC SE</th>
                    <th>ESS</th>
                    <th>WNV (Time)</th>
                    <th>WNV (FDE)</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add metrics rows
    for sampler in samplers:
        llc_mean = metrics.get(f"{sampler}_llc_mean", "N/A")
        llc_se = metrics.get(f"{sampler}_llc_se", "N/A")
        ess = metrics.get(f"{sampler}_ess", "N/A")
        wnv_time = metrics.get(f"{sampler}_wnv_time", "N/A")
        wnv_fde = metrics.get(f"{sampler}_wnv_fde", "N/A")

        # Format values properly for display
        llc_mean_str = (
            f"{llc_mean:.4f}" if isinstance(llc_mean, (int, float)) else str(llc_mean)
        )
        llc_se_str = (
            f"{llc_se:.4f}" if isinstance(llc_se, (int, float)) else str(llc_se)
        )
        ess_str = f"{ess:.1f}" if isinstance(ess, (int, float)) else str(ess)
        wnv_time_str = (
            f"{wnv_time:.6f}" if isinstance(wnv_time, (int, float)) else str(wnv_time)
        )
        wnv_fde_str = (
            f"{wnv_fde:.6f}" if isinstance(wnv_fde, (int, float)) else str(wnv_fde)
        )

        html_content += f"""
                <tr>
                    <td><strong>{sampler.upper()}</strong></td>
                    <td>{llc_mean_str}</td>
                    <td>{llc_se_str}</td>
                    <td>{ess_str}</td>
                    <td>{wnv_time_str}</td>
                    <td>{wnv_fde_str}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>

        <div class="note">
            <strong>Note:</strong> WNV = Work-Normalized Variance. ESS = Effective Sample Size using bulk method.
        </div>

        <h2>Diagnostic Plots</h2>
    """

    # Add plot sections for each sampler
    for sampler in samplers:
        if plot_groups[sampler]:
            html_content += f"""
        <div class="sampler-section">
            <h3>{sampler.upper()} Diagnostics</h3>
            <div class="plot-grid">
            """

            for plot_file in plot_groups[sampler]:
                plot_title = (
                    plot_file.replace(f"{sampler}_", "")
                    .replace(".png", "")
                    .replace("_", " ")
                    .title()
                )
                html_content += f"""
                <div class="plot-item">
                    <h3>{plot_title}</h3>
                    <img src="{plot_file}" alt="{plot_title}">
                </div>
                """

            html_content += """
            </div>
        </div>
            """

    html_content += f"""

        <h2>Configuration</h2>
        <ul>
            <li><strong>Data points:</strong> {cfg.n_data}</li>
            <li><strong>Chains:</strong> {cfg.chains}</li>
            <li><strong>Loss function:</strong> {cfg.loss}</li>
            <li><strong>Model depth:</strong> {cfg.depth}</li>
            <li><strong>Parameter count:</strong> {getattr(cfg, "target_params", "N/A")}</li>
        </ul>

        <div class="note">
            <strong>Files:</strong> All artifacts including NetCDF data files, configuration, and metrics are saved in this directory.
        </div>
    </div>
</body>
</html>
    """

    gallery_path = f"{run_dir}/index.html"
    with open(gallery_path, "w") as f:
        f.write(html_content)

    return gallery_path


def save_L0(run_dir: str, L0: float) -> str:
    """Save L0 value for running LLC reconstruction"""

    path = Path(run_dir) / "L0.txt"
    path.write_text(f"{L0:.10f}")
    return str(path)


def save_plot(fig, path: str, **kwargs) -> None:
    """Save matplotlib figure with consistent settings"""
    default_kwargs = {"dpi": 150, "bbox_inches": "tight", "facecolor": "white"}
    default_kwargs.update(kwargs)
    fig.savefig(path, **default_kwargs)


def _to_py(obj: Any):
    """Convert JAX/NumPy containers to plain Python (lists/floats/ints) for JSON."""
    # NumPy / JAX arrays (works for jax.Array, DeviceArray, np.ndarray)
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        try:
            return np.asarray(obj).tolist()
        except Exception:
            return repr(obj)
    # NumPy scalar
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return repr(obj)
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj

def save_tuned_params(run_dir: str, name: str, params: Dict[str, Any]) -> str:
    """Save tuned sampler parameters to JSON (JAX/NumPy safe)."""
    path = Path(run_dir) / f"tuned_{name}.json"
    params_py = _to_py(params)
    path.write_text(json.dumps(params_py, indent=2))
    return str(path)


def load_tuned_params(run_dir: str, name: str) -> Dict[str, Any] | None:
    """Load tuned sampler parameters if present; else return None."""
    path = Path(run_dir) / f"tuned_{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
