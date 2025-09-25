# llc/commands/sweep_cmd.py
import csv
import os
import json

def _save_sweep_results(results):
    """Save sweep results to CSV and JSON files."""
    os.makedirs("runs", exist_ok=True)

    ok_rows = [r for r in results if r.get("status","ok")=="ok"]
    err_rows = [r for r in results if r.get("status")=="error"]

    if ok_rows:
        with open("llc_sweep_results.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in ok_rows for k in r}))
            w.writeheader()
            w.writerows(ok_rows)

    if err_rows:
        with open("llc_sweep_errors.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in err_rows for k in r}))
            w.writeheader()
            w.writerows(err_rows)

    # optional log file for inspection
    with open("llc_sweep_results.json","w") as f:
        json.dump(results, f, indent=2)