"""Modal artifact utilities for CLI."""

import io
import os
import tarfile
from pathlib import Path


def extract_modal_artifacts_locally(result_dict: dict) -> None:
    """Download and extract artifact tarball to ./artifacts/<run_id> if present."""
    if result_dict.get("artifact_tar") and result_dict.get("run_id"):
        rid = result_dict["run_id"]
        dest_root = "artifacts"
        os.makedirs(dest_root, exist_ok=True)
        dest = os.path.join(dest_root, rid)

        # Clean existing
        if os.path.exists(dest):
            import shutil
            shutil.rmtree(dest)

        with tarfile.open(
            fileobj=io.BytesIO(result_dict["artifact_tar"]), mode="r:gz"
        ) as tf:
            tf.extractall(dest_root)
        print(f"Artifacts downloaded and extracted to: {dest}")
        result_dict["run_dir"] = dest


def pull_and_extract_artifacts(run_id: str = None, target: str = "artifacts") -> str:
    """Pull artifacts from Modal using SDK and extract locally."""
    import modal

    APP = "llc-experiments"
    FN_LIST = "list_artifacts"
    FN_EXPORT = "export_artifacts"

    list_fn = modal.Function.from_name(APP, FN_LIST)
    export_fn = modal.Function.from_name(APP, FN_EXPORT)

    if run_id:
        print(f"[pull-sdk] Pulling specific run: {run_id}")
    else:
        print("[pull-sdk] Discovering latest run on server...")
        paths = list_fn.remote("/artifacts")
        if not paths:
            raise RuntimeError("No remote artifacts found.")
        run_id = Path(sorted(paths)[-1]).name
        print(f"[pull-sdk] Latest on server: {run_id}")

    print(f"[pull-sdk] Downloading and extracting {run_id}...")
    data = export_fn.remote(run_id)
    dest_root = Path(target)
    dest_root.mkdir(parents=True, exist_ok=True)

    target_dir = dest_root / run_id
    if target_dir.exists():
        import shutil
        shutil.rmtree(target_dir)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest_root)

    extracted_path = str(dest_root / run_id)
    print(f"[pull-sdk] Extracted into {extracted_path}")
    return extracted_path