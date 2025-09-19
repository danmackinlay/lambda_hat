"""Pull artifacts command implementation."""

from llc.util.modal_utils import pull_and_extract_runs


def pull_runs_entry(run_id: str = None, target: str = "runs") -> None:
    """Entry point for pull-runs command. Pulls into runs/ by default."""
    pull_and_extract_runs(run_id, target)
