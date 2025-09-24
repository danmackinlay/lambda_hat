import json
import os
import numpy as np
from pathlib import Path

def save_config(run_dir: str, cfg) -> str:
    d = {k: (v.tolist() if hasattr(v, "shape") else v) for k,v in cfg.__dict__.items()}
    p = Path(run_dir, "config.json"); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, default=str)); return str(p)

def save_metrics(run_dir: str, metrics: dict) -> str:
    def to_py(o):
        if hasattr(o, "shape"): return np.asarray(o).tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
        if isinstance(o, dict): return {k: to_py(v) for k,v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        return o
    p = Path(run_dir, "metrics.json")
    p.write_text(json.dumps(to_py(metrics), indent=2)); return str(p)

def save_L0(run_dir: str, L0: float) -> str:
    p = Path(run_dir, "L0.txt"); p.write_text(f"{L0:.10f}"); return str(p)

def generate_gallery_html(run_dir: str, sampler: str) -> str:
    """Generate simple HTML gallery for run results."""
    png_files = [f for f in os.listdir(run_dir) if f.endswith(".png")]
    png_files.sort()

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLC Results - {sampler.upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot {{ margin: 10px; text-align: center; }}
        img {{ max-width: 800px; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <h1>LLC Analysis Results - {sampler.upper()}</h1>
    <p>Run directory: {os.path.basename(run_dir)}</p>
"""

    for png_file in png_files:
        html_content += f"""
    <div class="plot">
        <h3>{png_file}</h3>
        <img src="{png_file}" alt="{png_file}">
    </div>
"""

    html_content += """
</body>
</html>
"""

    html_path = os.path.join(run_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html_content)

    return html_path