"""Command modules for lambda_hat CLI.

This package contains the business logic for all CLI commands, separated from
the CLI parsing/routing layer. Each command module provides entry functions
that can be called from:
- CLI wrappers (lambda_hat/cli.py)
- Parsl orchestration (@python_app)
- Unit tests

Design principles:
- Commands take data (dicts, strings, paths), not argparse.Namespace
- Commands return structured results (dicts) or write to stdout
- No CLI parsing logic in command modules
- Business logic only - routing belongs in cli.py
"""
