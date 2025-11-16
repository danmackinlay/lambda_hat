#!/usr/bin/env python3
"""Generate configuration reference documentation from YAML files."""

import yaml
from pathlib import Path
from collections import defaultdict


def generate_config_doc():
    """Generate config reference from lambda_hat/conf/**/*.yaml."""
    root = Path(__file__).parent.parent / "lambda_hat" / "conf"

    output = [
        "# Configuration Reference",
        "",
        "**Auto-generated from `lambda_hat/conf/**/*.yaml`** — do not edit by hand.",
        "",
        "Use this page to look up exact YAML defaults and schemas.",
        "For a conceptual guide on composing experiments, see [Experiments Guide](./experiments.md).",
        "",
        "This page lists all configuration options with their default values.",
        "Configuration files use Hydra/OmegaConf for composition and interpolation.",
        "",
    ]

    # Group by directory
    by_group = defaultdict(list)
    for yaml_path in sorted(root.rglob("*.yaml")):
        rel_path = yaml_path.relative_to(root)
        group = str(rel_path.parent) if rel_path.parent != Path(".") else "root"
        by_group[group].append((str(rel_path), yaml_path))

    # Render each group
    for group in sorted(by_group.keys()):
        output.append(f"## {group.replace('/', ' / ')}")
        output.append("")

        for rel_path, yaml_path in sorted(by_group[group]):
            output.append(f"### `{rel_path}`")
            output.append("")
            output.append("```yaml")
            output.append(yaml_path.read_text().rstrip())
            output.append("```")
            output.append("")

    # Write output
    doc_path = Path(__file__).parent / "config.md"
    doc_path.write_text("\n".join(output))
    print(f"✓ Generated {doc_path}")


if __name__ == "__main__":
    generate_config_doc()
