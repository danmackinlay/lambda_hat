Given where the repo is now, I’d seriously consider switching to a numeric container on disk and keep JSON only for manifest/metadata.

---

### Why JSON ended up here (in *your* code)

Look at `sample_entry`:

```python
traces_serializable = {}
for k, v in traces.items():
    if hasattr(v, "__array__") or hasattr(v, "tolist"):
        traces_serializable[k] = np.asarray(v).tolist()
    else:
        traces_serializable[k] = v

(run_dir / "traces_raw.json").write_text(json.dumps(traces_serializable, indent=2))
```

So you’re explicitly flattening JAX/NumPy arrays to list-of-lists and dumping to text.

Given the rest of the design, there are a few very plausible reasons this happened:

1. **Zero new dependencies in workers.**

   * Workers already need `jax`, `numpy`, `equinox`, etc.
   * Writing JSON uses only the stdlib.
   * Writing NetCDF/HDF5/Zarr requires `netCDF4`, `h5py`, `zarr` or similar, which are notoriously annoying on clusters (MPI/HDF version hell). You explicitly pushed ArviZ + NetCDF to the controller stage to avoid importing heavy libs on workers.

2. **Human-inspectable “flight recorder”.**

   * Someone wanted to `cat`/`jq` `traces_raw.json` on a failed run and see what the sampler did without loading Python.
   * In early iterations, trace lengths were small enough that file size wasn’t offensive, so JSON was “good enough”.

3. **Type safety / robustness under refactors.**

   * JSON degrades everything to primitive types; if you change internal dtypes, JAX vs NumPy, shapes, etc., the dump still works.
   * Deserialization in `diagnose_cmd` only assumes “list → np.array”; it doesn’t care if those arrays came from JAX or NumPy or something else in the past.

4. **Parsl and error surface area.**

   * Text files degrade nicely when something half-writes, and are easier to inspect when Parsl crashes mid-task.
   * Binary formats can produce corrupt blobs that are less obvious until you try to load them.

So the short honest answer to “why” is: ease and robustness. Not performance, not storage.

---

### Is JSON a good idea here?

It depends on your scale:

* With your current traces (LLC-only, a handful of samplers, a few hundred–thousand draws, maybe 2–4 chains), JSON is tolerable but wasteful.
* If you ever:

  * increase T by an order of magnitude,
  * run dozens of samplers × targets,
  * or start adding more scalar diagnostics,
    you’ll notice JSON I/O and size pain.

For a Bayesian/statistics person: you’re currently using a human-readable wire format for what is effectively a tiny ad-hoc column store. It’s fine at toy scale; it’s the wrong tool at serious scale.

You’ve also now made `analysis.py` the golden path that produces `trace.nc` and metrics. That means:

* Workers don’t need to be debuggable by non-Python tools anymore; you debug via `lambda-hat diagnose ...`.
* You already require NumPy everywhere.
* You’re comfortable with binary outputs (`trace.nc`) in Stage C.

Given that, JSON is now the weak point.

---

### What I would do instead

If you want to keep workers light and keep the architecture you just built, the least-painful improvement is:

#### 1. Switch worker output to `.npz` (NumPy’s zipped container)

* Worker writes `traces_raw.npz` instead of `traces_raw.json`.
* Manifest stays JSON (good for grepping, small).

In `sample_entry`:

```python
traces_arrays = {k: np.asarray(v) for k, v in traces.items()}
np.savez_compressed(run_dir / "traces_raw.npz", **traces_arrays)
```

In `diagnose_cmd`:

* Replace `_load_traces_raw_json` with `_load_traces_raw_npz`:

```python
def _load_traces_raw_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
```

* Update `diagnose_entry` to look for `traces_raw.npz` instead of JSON and then feed that into `analyze_traces`.

This buys you:

* Much smaller files.
* Faster load/dump.
* Still no heavy dependencies on workers.
* No change to the golden path (`analyze_traces` already works with `Dict[str, np.ndarray]`).


Now you can glance at shapes and ranges without blowing out disk.

#### 2. If you wanted to be more opinionated

If you’re willing to fully embrace ArviZ as the only consumer, you can skip `traces_raw.*` entirely and have workers write **ArviZ InferenceData** caches directly (`trace.nc`) – but you just went the other direction (push diagnostics out of workers), for good reasons:

* ArviZ + NetCDF4 import cost on workers.
* Potential library mismatches on HPC nodes.
* Bigger failure surface inside Parsl tasks.

Given that, `.npz` is a better compromise than trying to resurrect NetCDF in Stage B.

---

### So: why JSON, and what now?

* You used JSON because it’s easy, robust, and doesn’t require new dependencies. It’s not because it’s particularly sensible for numeric traces.
* With the new golden-path design, JSON is now the least coherent piece.
* The clean path forward is:

  * Keep `manifest.json` (metadata).
  * Replace `traces_raw.json` with `traces_raw.npz`.
  * Optionally emit a small human-readable summary JSON.
  * Leave `analyze_traces` and Stage C unchanged except for using `np.load` instead of `json.loads`.


---

## Pros & Cons of `.npz` vs `.nc`

### 1. `.npz` (NumPy zipped arrays)

**Pros**

* Lightweight dependency: only `numpy` needed (already present).
* Compact: binary storage of arrays, faster load and dump compared to JSON lists of lists.
* Simple to implement on the worker side (fits your goal of minimal worker burden).
* Works naturally with arbitrary arrays (chains × draws, sample_stats etc.).
* No semantic “sampling” baggage: you’re just storing arrays.

**Cons**

* Lacks rich metadata/grouping semantics: you’ll need to manage shapes, names, dims manually (e.g., you must document chain/draw axes).
* Doesn’t integrate as neatly with ArviZ/xarray diagnostic ecosystem (though you could convert later).
* No built-in “group” concept (posterior, sample_stats, etc) beyond your convention.
* Less language‐agnostic than NetCDF when you care about external tooling.

**Implication for your sampler types**

* Works for both MCMC and VI: you just store whatever arrays you have (e.g., VI ELBO trace, or LLC trace). No claim of “posterior samples” needed: you’re storing traces.
* Because you’re not “selling” them as independent draws from a posterior, the simple array format fits and avoids confusion.
* Later diagnosis (in your `analyze_traces`) only requires arrays anyway (you convert to InferenceData there). So using `.npz` is a clean fit.


**Plan:**

1. **Workers (Stage B: sample_entry)**

   * Replace `traces_raw.json` with `traces_raw.npz`.

     ```python
     # after sampling result
     traces_arrays = {k: np.asarray(v) for k, v in traces.items()}
     np.savez_compressed(run_dir / "traces_raw.npz", **traces_arrays)
     # optionally also write a small summary JSON about shapes/min/max
     ```
   * Keep `manifest.json` unchanged.

2. **Diagnose Command (`diagnose_cmd.py`)**

   * Detect `traces_raw.npz` instead of JSON.
   * Provide loader `_load_traces_raw_npz(path)` returning `Dict[str, np.ndarray]`.
   * Convert arrays into `analyze_traces(...)` exactly as before.
   * Remove JSON‐list parsing logic.
   * (Optional) If legacy `traces_raw.json` exists, fallback for compatibility.

3. **`analyze_traces` & `analysis.py`**

   * No change needed: method accepts `np.ndarray` already; only change is how input is loaded.

4. **Artefact storage/metadata**

   * Everything else (manifests, diagnostics, plots, trace.nc, metrics) remains unchanged.
   * Optionally write `trace.nc` in diagnostics for archival of InferenceData for all samplers (both MCMC/VI). For VI you’ll treat “chain=0, draw=steps”.

5. **Backward compatibility / migration**

   * delete all previous experiments and start fresh to avoid migration heaadaches
   * aggressively delete all previous JSON-based code paths.

6. **Benefit**

   * Workers remain lightweight, minimal new dependencies.
   * Storage efficient, loading fast.
   * Keeps semantics simple for both sampler types.
   * Diagnostics and plots unchanged.
