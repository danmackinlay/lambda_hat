"""Pull artifacts command implementation."""

from llc.util.modal_utils import pull_and_extract_artifacts


def pull_artifacts_entry(run_id: str = None, target: str = "runs") -> None:
    """Entry point for pull-artifacts command. Pulls into runs/ by default."""
    pull_and_extract_artifacts(run_id, target)
