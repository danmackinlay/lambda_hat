Below is an expanded, opinionated blueprint for moving your project to a **Metaflow-first** workflow that keeps dynamic configuration as a first‑class citizen, makes large (N \times M) parameter sweeps easy, and collates results into tidy data frames—while still giving you parallelism and a clean DAG. I’ve also included a concise list of compatible backends at the end.

---

## Executive summary

* **Replace Snakemake** for orchestration with **Metaflow flows** that (a) generate targets/teachers & datasets once and (b) fan‑out to many posterior/variational configurations via `foreach`, then **join** to collate metrics into a single pandas **DataFrame artifact** and a shareable **Card** report. This addresses your two goals (dependency/DAG management + parallelism) in one place. ([Metaflow Docs][1])
* Keep **config-driven exploration** by combining Metaflow’s new **`Config`** object and classic **`Parameter`**s with **OmegaConf** for hierarchical YAML + CLI/dotlist merges. This gives the “fluid” overrides you missed after moving off Hydra. ([Metaflow Docs][2])
* Scale out with **Kubernetes or AWS Batch** at the step level using `@kubernetes` / `@batch` and control concurrency with `--max-workers`. No code changes needed to move from local to cluster/backends. ([Metaflow Docs][3])
* Use **Metaflow Cards** to render tables/plots from each run and a final “sweep” card summarizing (N \times M) results. Store both the raw DataFrame and the Card as artifacts for reproducibility and fast inspection. ([Metaflow Docs][4])
* For cross-run analytics or building a history of experiments, query past results with the **Metaflow Client API** (and tags/namespaces) or optionally add **MLflow / W&B / Comet** for a richer experiment UI—minimal glue code. ([Metaflow Docs][5])

---

## Proposed flow topology

### 1) **`BuildDataFlow`** (single producer of shared inputs)

Purpose: deterministically produce **N** teachers/targets and their datasets exactly once, tagged and versioned.

* **Steps**

  * `start` → parse configs, set seeds.
  * `make_teachers` → `foreach` over a list of `teacher_cfgs` to generate teachers + datasets (fan‑out).
  * `join_data` → collate metadata and publish an index (list of dataset “handles”) as an artifact (e.g., list of S3 keys / ids).

* **CLI**

  ```bash
  python flows/build_data_flow.py run \
    --teacher-configs @configs/teachers.yaml \
    --with kubernetes \
    --max-workers 64
  ```

  * `--with kubernetes` (or `--with batch`) flips steps to run remotely; `--max-workers` throttles parallelism globally. ([Metaflow Docs][6])

* **Artifacts & visibility**

  * Persist each teacher/dataset as artifacts.
  * Emit an **index DataFrame** and a **default Card** showing the DAG, parameters, and pointers to outputs. ([Metaflow Docs][7])

### 2) **`FitApproxFlow`** (matrix of experiments on shared inputs)

Purpose: apply **M** posterior/variational configurations to each of the **N** datasets from `BuildDataFlow` → (N \times M).

* **Steps**

  * `start` → read `teacher_index_tag` or a `Run` id to pull artifacts from `BuildDataFlow` via **Client API**. ([Metaflow Docs][5])
  * `sweep_axes` → prepare two axes: `teachers` × `approx_cfgs`.
  * `fit_one` → **nested `foreach`** over (teacher, approx_cfg). Each branch runs the estimation algorithm, logs metrics/artifacts, and can request GPUs/CPUs with `@resources`. ([Outerbounds Docs][8])
  * `aggregate` (join) → collate all per‑branch metrics into a tidy **pandas DataFrame** (one row per pair), persist as an artifact, and render a Card with sortable table + quick plots. ([Metaflow Docs][9])

* **CLI**

  ```bash
  python flows/fit_approx_flow.py run \
    --teacher-index-selector 'tag:teacherset=v1' \
    --approx-configs @configs/approx_grid.yaml \
    --overrides "approx.lr=1e-3,approx.beta=0.9" \
    --with kubernetes:namespace=ml,service_account=runner,image=ghcr.io/you/llc:latest \
    --max-workers 200
  ```

  * You can pass a config file, and still **override arbitrarily from the command line** (see the Config + OmegaConf design below).
  * `@kubernetes` parameters and env can also be driven by **Config** or CLI. ([Metaflow Docs][10])

---

## Configuration & overrides (keeping it fluid)

### Use **Metaflow `Config`** + **OmegaConf** for deep merges & dotlists

* Define one or more `Config(...)` entries on the flow that can come from JSON/YAML and **custom parsers**. The parser can load YAML, merge multiple files, and accept **OmegaConf dotlist** overrides, producing a plain dict for Metaflow. ([Metaflow Docs][11])
* OmegaConf supports **from dotlist / from CLI** and deep merges, which gives you the Hydra‑like ergonomics you want for exploratory runs. ([OmegaConf][12])

**Sketch (inside your flow):**

```python
from metaflow import Config
from omegaconf import OmegaConf

def omegaconf_parser(path_or_none, overrides):
    base = OmegaConf.create({})
    if path_or_none:
        base = OmegaConf.merge(base, OmegaConf.load(path_or_none))
    if overrides: # comma-separated dotlist string
        dotlist = [s.strip() for s in overrides.split(",")]
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))
    return OmegaConf.to_container(base, resolve=True)  # plain dict for Config

class FitApproxFlow(FlowSpec):
    cfg = Config(
        "cfg",
        default="configs/default.yaml",
        parser=lambda p: omegaconf_parser(p, os.environ.get("CFG_OVERRIDES"))
    )
```

* You can also expose targeted **`Parameter`**s (with `type=JSONType`) for structures like a grid of approximations and set them directly via CLI:
  `python flow.py run --approx-configs='[{"name":"vi","lr":3e-4},{"name":"laplace"}]'` ([Metaflow Docs][13])

* Metaflow will happily **use `Config` to configure decorators** (e.g., timeouts, resources) pre‑execution, which is exactly the gap Parameters can’t cover. ([Metaflow Docs][2])

---

## The (N \times M) sweep mechanics

* Use **nested `foreach`**: outer axis over teachers/datasets, inner axis over approximation configs. Metaflow clones the inner steps per item and can run them **in parallel** on your chosen backend. Control fan‑out with `--max-workers`. ([Metaflow Docs][14])
* Request resources per task with `@resources(gpu=1, cpu=8, memory=...)`; run remotely with `--with kubernetes` or `--with batch` to scale out. ([Metaflow Docs][15])
* If your grid becomes huge, you can also programmatically launch runs with the **Runner API** (spawning many small Metaflow runs) and still collate via the Client API. ([Metaflow Docs][16])

---

## Results collation & reporting

* In the **join** step, traverse `inputs` to build a single **DataFrame** with all metrics (e.g., ELBO, predictive log‑likelihood, timing, seeds, config hashes). Persist it as `self.results_df`. As an additional output, render a **Card** that shows the table and quick plots (histograms, scatter of metric vs. hyperparam). ([Metaflow Docs][9])
* Cards can be opened locally, in the CLI, or in **Metaflow UI**; they also display the flow DAG and parameter values by default (great for auditability). ([Metaflow Docs][7])

**Optional experiment tracking**

* If you want a richer cross‑run UI without writing glue code, integrate **MLflow** (call `mlflow.log_params/metrics` in each branch; autologging works for many libraries) or use the **W&B / Comet** integrations (decorators exist). Start simple with Cards + Client API; add one of these if you miss a leaderboard UI. ([MLflow][17])

---

## Coordinating multiple flows (if you keep build & fit separate)

* Use the **Client API inside flows** to fetch artifacts from other flows (e.g., `BuildDataFlow.latest_successful_run` by tag). This preserves the “produce once, consume many” teacher/dataset invariant across dependent workflows. ([Metaflow Docs][18])
* For production scheduling, Metaflow supports **Argo Workflows** (Kubernetes), **AWS Step Functions**, or **Apache Airflow**—including event‑triggering with Argo if you want runs to start when new data lands. ([Metaflow Docs][19])

---

## Parallelism & reliability knobs

* **Concurrency:** `--max-workers` caps the number of concurrent tasks per run—useful to avoid overwhelming the cluster or quotas. ([Metaflow Docs][20])
* **Resources & accelerators:** add `@resources(gpu=1)` and friends to steps; GPUs are supported on both **AWS Batch** and **Kubernetes**. ([Metaflow Docs][15])
* **Retries / timeouts:** add `--with retry` globally or `@retry`/`@timeout` per step to gracefully handle transient failures and long‑running tasks. ([Metaflow Docs][6])

---

## Minimal doc content to add to your repo

1. **Quickstart: explore parameters from the CLI**

```bash
# Run 2 teachers × 4 approx configs locally
python flows/fit_approx_flow.py run \
  --teacher-configs @configs/teachers.small.yaml \
  --approx-configs  @configs/approx.grid.yaml \
  --overrides "approx.lr=3e-4,approx.steps=2000"

# Burst to Kubernetes with 200-way parallelism
python flows/fit_approx_flow.py run \
  --teacher-configs @configs/teachers.medium.yaml \
  --approx-configs  @configs/approx.grid.yaml \
  --with kubernetes:namespace=ml,image=ghcr.io/you/llc:latest \
  --max-workers 200
```

`--with` lets you attach step decorators from the CLI (e.g., `--with kubernetes` or `--with batch:cpu=8,memory=32G`). ([Metaflow Docs][6])

2. **Config files & overrides**

* Defaults live in `configs/*.yaml`.
* Override anything from the CLI with `--overrides "a.b=... , c.d=..."` (OmegaConf dotlist) **or** pass JSON to a `JSONType` Parameter. ([OmegaConf][12])

3. **Where results go**

* Per-branch artifacts (metrics, fitted objects) are persisted automatically and can be retrieved later via **Client API**. The run emits a final `results_df.parquet` and a **Card** summary. ([Metaflow Docs][1])

4. **Running on the cloud**

* Configure once (`metaflow configure kubernetes` / set `METAFLOW_*` env vars) and use `--with kubernetes` or `--with batch` to scale out. ([outerbounds.com][21])

---

## Do you also need MLflow (or similar)?

Not strictly. Metaflow already snapshots code, persists artifacts, and provides UI/CLI/Client APIs and Cards. If you want a **centralized experiment UI** with fast comparison across many runs, add **MLflow** (or W&B/Comet). Integration is just a few lines per step and doesn’t complicate the DAG. ([MLflow][17])

---

## Backends you can use with this design

**Compute backends (per‑step remote execution)**

* **Local** (your laptop) — good for dev.
* **AWS Batch** via `@batch` / `--with batch`. ([Metaflow Docs][3])
* **Kubernetes** anywhere (EKS, GKE, AKS, on‑prem) via `@kubernetes` / `--with kubernetes`. ([Metaflow Docs][10])

**Production orchestrators (scheduling/deploying whole flows)**

* **Argo Workflows** (Kubernetes‑native; supports **event triggers**). ([Metaflow Docs][22])
* **AWS Step Functions** (managed AWS orchestration). ([Metaflow Docs][23])
* **Apache Airflow** (Metaflow integration available). ([Metaflow Docs][24])

**Datastore for artifacts / code snapshots**

* **S3** (first‑class). ([Metaflow Docs][25])
* **GCS / Azure Blob** with the Kubernetes stack and proper credentials (Outerbounds docs show this pattern). ([outerbounds.com][26])
* **On‑prem S3‑compatible** (e.g., **MinIO**), used by some deployments. ([Metaflow Docs][27])

**Metadata & UI**

* **Local metadata** (`.metaflow/`) for dev or single‑user mode. ([Metaflow Docs][28])
* **Metaflow Service** (central metadata DB + **Metaflow UI**). ([GitHub][29])
* **DevStack** for a one‑click local cluster with UI on Minikube. ([Metaflow Docs][30])

**Metrics/experiment systems (optional)**

* **MLflow** (Tracking & UI). ([MLflow][17])
* **Weights & Biases** (official integration). ([Weights & Biases Documentation][31])
* **Comet** (official integration). ([Comet][32])

---

## Migration plan (practical & low‑maintenance)

1. **Lay down the structure**

   ```
   flows/
     build_data_flow.py
     fit_approx_flow.py
   configs/
     teachers.yaml
     approx.grid.yaml
     default.yaml
   ```

   Each flow keeps pure Python steps; model code stays in `src/` package for reuse.

2. **Wire up Config + OmegaConf parser** as shown above. Document `--overrides` and JSON `Parameter` options.

3. **Implement nested `foreach`** and the **join** to emit `results_df` + Card. Use `@resources(...)` where needed.

4. **Turn on remote compute** incrementally (`--with kubernetes` on a few steps, then all). Set concurrency with `--max-workers`. ([Metaflow Docs][20])

5. **(Optional)** Add MLflow/W&B/Comet logging in `fit_one` if you want external tracking. ([MLflow][17])

6. **(Optional)** For prod, deploy an orchestrator: Argo on any K8s, or Step Functions on AWS. Add event triggers if helpful. ([Metaflow Docs][19])

---

## Why this resolves the Snakemake tension

* **DAG + shared inputs:** Flows encode dependencies explicitly (`self.next(...)`), and shared teachers/datasets are either produced once in the same flow before the fan‑out **or** fetched from a prior flow with the Client API—no need to invent external file targets. ([Metaflow Docs][18])
* **Parallelism:** `foreach` + `--max-workers` + `@resources` gives you coarse and fine control over parallel launches across any supported backend with minimal boilerplate. ([Metaflow Docs][20])
* **Exploration:** `Config` + OmegaConf restores the “fluid CLI override” feel you had with Hydra, including decorator‑level tweaks (timeouts, resource requests) that Parameters alone can’t provide. ([Metaflow Docs][2])

---

## Known trade‑offs / notes

* Metaflow’s first‑class datastore is **S3**; for on‑prem or other clouds, the recommended path is K8s + **GCS/Azure Blob** or **S3‑compatible** stores like MinIO (works in practice, but plan for light infra ops). ([outerbounds.com][26])
* Very large sweeps can hit practical scheduler limits; keep an eye on max concurrency and split extremely large grids into multiple runs or use the **Runner API** to launch in batches. ([Metaflow Docs][16])

---

### Bottom line

Adopting this Metaflow‑first plan will give you: a reproducible DAG, fast parallel sweeps, ergonomic CLI overrides, one‑shot collation to a DataFrame, and low maintenance. Start with Cards + Client API; add MLflow/W&B later only if the UI payoff is worth it.

If you’d like, I can draft the two flows (`BuildDataFlow`, `FitApproxFlow`) and the OmegaConf parser module next, plus the README section that documents the exact CLI patterns shown above.

[1]: https://docs.metaflow.org/metaflow/basics "Creating Flows"
[2]: https://docs.metaflow.org/metaflow/configuring-flows/introduction "Configuring Flows"
[3]: https://docs.metaflow.org/scaling/remote-tasks/introduction "Computing at Scale"
[4]: https://docs.metaflow.org/api/cards "Cards - Visualizing results"
[5]: https://docs.metaflow.org/metaflow/client "Inspecting Flows and Results"
[6]: https://docs.metaflow.org/api/step-decorators "Step Decorators"
[7]: https://docs.metaflow.org/metaflow/visualizing-results/effortless-task-inspection-with-default-cards "Effortless Task Inspection with Default Cards"
[8]: https://docs.outerbounds.com/nested-foreach/ "Nested Foreach Flows"
[9]: https://docs.metaflow.org/scaling/data "Loading and Storing Data"
[10]: https://docs.metaflow.org/scaling/remote-tasks/kubernetes "Using Kubernetes"
[11]: https://docs.metaflow.org/metaflow/configuring-flows/basic-configuration "Basic Configuration"
[12]: https://omegaconf.readthedocs.io/en/1.4_branch/usage.html "Installation — OmegaConf 1.0 documentation - Read the Docs"
[13]: https://docs.metaflow.org/api/flowspec "FlowSpec - Constructing flows"
[14]: https://docs.metaflow.org/v/r/metaflow/basics "Basics of Metaflow"
[15]: https://docs.metaflow.org/scaling/remote-tasks/gpu-compute "Using GPUs and Other Accelerators"
[16]: https://docs.metaflow.org/api/runner "Runner - Running flows programmatically"
[17]: https://mlflow.org/docs/latest/ml/tracking/ "MLflow Tracking"
[18]: https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode03 "Episode 3: Playlist Redux"
[19]: https://docs.metaflow.org/production/introduction "Production Deployments"
[20]: https://docs.metaflow.org/scaling/remote-tasks/controlling-parallelism "Controlling Parallelism"
[21]: https://outerbounds.com/engineering/operations/configure-metaflow/ "Configuring Metaflow"
[22]: https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-argo-workflows "Scheduling Metaflow Flows with Argo Workflows"
[23]: https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-aws-step-functions "Scheduling Metaflow Flows with AWS Step Functions"
[24]: https://docs.metaflow.org/production/scheduling-metaflow-flows/scheduling-with-airflow "Scheduling Metaflow Flows with Apache Airflow"
[25]: https://docs.metaflow.org/internals/technical-overview "Technical Overview"
[26]: https://outerbounds.com/engineering/operations/airflow/ "Using Airflow with Metaflow"
[27]: https://docs.metaflow.org/getting-started/infrastructure "Deploying Infrastructure for Metaflow"
[28]: https://docs.metaflow.org/v/r/metaflow/client "Inspecting Flows and Results"
[29]: https://github.com/Netflix/metaflow-service "rocket: Metadata tracking and UI service for Metaflow!"
[30]: https://docs.metaflow.org/getting-started/devstack "Setting Up the Dev Stack"
[31]: https://docs.wandb.ai/models/integrations/metaflow "Metaflow - Weights & Biases Documentation - Wandb"
[32]: https://www.comet.com/docs/v2/integrations/third-party-tools/metaflow/ "Metaflow - Comet Docs"
