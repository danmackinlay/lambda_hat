#!/usr/bin/env python3
"""Generate CLI reference documentation from Click app."""

from pathlib import Path

from click.testing import CliRunner

from lambda_hat.cli import cli


def generate_cli_doc():
    """Generate CLI reference from Click command tree."""
    runner = CliRunner()

    # Get main help
    result = runner.invoke(cli, ["--help"])
    output = [
        "# CLI Reference",
        "",
        "**Auto-generated from `lambda_hat/cli.py`** — do not edit by hand.",
        "",
        "## Main Command",
        "",
        "```text",
        result.output,
        "```",
        "",
    ]

    # Get help for each subcommand group
    for cmd_name in ["build", "sample", "artifacts", "promote", "workflow"]:
        result = runner.invoke(cli, [cmd_name, "--help"])
        if result.exit_code == 0:
            output.extend(
                [
                    f"## `lambda-hat {cmd_name}`",
                    "",
                    "```text",
                    result.output,
                    "```",
                    "",
                ]
            )

    # Write output
    doc_path = Path(__file__).parent / "cli.md"
    doc_path.write_text("\n".join(output))
    print(f"✓ Generated {doc_path}")


if __name__ == "__main__":
    generate_cli_doc()
