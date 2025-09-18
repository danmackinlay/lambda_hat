"""Pull artifacts command implementation."""

from llc.util.modal_utils import pull_and_extract_artifacts


def pull_artifacts_entry(run_id: str = None, target: str = "artifacts") -> None:
    """Entry point for pull-artifacts command."""
    pull_and_extract_artifacts(run_id, target)