
* **HTEX doesn’t require you to run a separate “ZMQ server.”** It spawns a tiny *interchange* process for you and uses ZeroMQ sockets between the driver, the interchange, and workers. You don’t need any daemon beyond what HTEX starts itself. The networking pain you saw (binding to `10.100.0.2`) is address auto‑detection going wrong, not a missing service. The fix is to **set the address explicitly** (and, optionally, pin port ranges). ([Parsl][1])

* **“Slurm executor” vs HTEX:** In Parsl terms, **Slurm is a *provider*** (how blocks are obtained), while **HTEX is an *executor*** (how tasks are run inside those blocks). HTEX uses a **pilot‑job model**: Slurm gives you *blocks* (batch jobs), and Parsl schedules many tasks *inside* those blocks. That is a different goal than “one Slurm job per small task.” ([GitLab][2])

* If your real goal is **“let Slurm schedule many small jobs itself”**, you have three viable paths:

  1. Configure HTEX to submit **many tiny one‑node blocks** (each block is one Slurm job) and let Parsl fill them—this preserves the Parsl workflow and still leverages Slurm queueing.
  2. Use **WorkQueueExecutor** (start many workers via Slurm job arrays; Parsl sends tasks to whichever workers check in).
  3. Skip Parsl for the compute step and **use Slurm job arrays** (each array element runs a `lambda-hat` CLI), then have your Python orchestrator collect outputs.
     (Details and examples below.)

---

## What’s antithetical and what isn’t

* **Antithetical:** With *vanilla* HTEX, **Slurm does not schedule each Parsl task**; Slurm gives HTEX long‑lived allocations (“blocks”), and *Parsl* schedules your tasks inside those. That’s the pilot‑job model by design. ([Parsl][3])

* **Compatible:** You can still exploit Slurm’s economics and backfill by **requesting lots of small, short blocks** (e.g., `nodes_per_block=1`, small `walltime`, `min_blocks=0`, high `max_blocks`). Then **Parsl adaptively scales blocks up/down** with the Simple strategy as the backlog changes. The scheduler sees many short jobs; Parsl still batches tasks efficiently inside each job. ([Parsl][4])

---

## Minimalist, robust local execution (no ZMQ flakes)

Your help‑desk report shows HTEX was binding to a VPN (`utun3`, `10.100.0.2`). On macOS, fix by *explicitly* telling HTEX to use loopback:

```python
# lambda_hat/parsl_cards.py  (pseudo-diff)
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.addresses import address_by_interface, address_by_hostname

htex = HighThroughputExecutor(
    label="htex_local",
    # For local-only runs:
    address="127.0.0.1",            # <- force loopback
    loopback_address="127.0.0.1",   # keep internal chatter local
    start_method="spawn",           # safer than fork for JAX/NumPy backends
    worker_port_range=(55000, 55100),
    interchange_port_range=(55101, 55200),
    cores_per_worker=1,
    max_workers_per_node=4,
    provider=LocalProvider(
        init_blocks=1, min_blocks=0, max_blocks=1,
        worker_init="export MPLBACKEND=Agg; export JAX_ENABLE_X64=1",
    ),
)
```

* `address` and `loopback_address` remove the auto‑detection failure.
* Pinning `worker_port_range`/`interchange_port_range` makes ports predictable.
* `start_method="spawn"` avoids fork‑related issues in scientific stacks (BLAS, XLA, etc.). ([Parsl][1])

> If you want this configurable from YAML, add an optional `address:` key; allow values like a literal IP (`127.0.0.1`), `hostname`, or `if:lo0` to go through `address_by_hostname()`/`address_by_interface()`. ([GitLab][5])

**Why this addresses your bug report:** HTEX’s interchange is already “the server”; the error came from it binding to the wrong NIC. For local runs, `127.0.0.1` is the most robust choice and needs no daemons. ([Parsl][1])

---

## Slurm: best‑practice configurations by goal

### A) Keep Parsl + HTEX, but let Slurm “see” many small jobs

Use one node per block and adaptive scaling:

```python
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname

htex = HighThroughputExecutor(
    label="htex_slurm_smallblocks",
    address=address_by_hostname(),        # must be reachable from compute nodes
    start_method="spawn",
    max_workers_per_node=1, cores_per_worker=1,
    worker_port_range=(55000,55100), interchange_port_range=(55101,55200),
    provider=SlurmProvider(
        partition="short",
        nodes_per_block=1,         # <- one job per block
        init_blocks=0, min_blocks=0, max_blocks=200,  # scale with backlog
        walltime="00:15:00",
        launcher=SrunLauncher(),   # standard Slurm launcher
        worker_init="source /path/to/env; export MPLBACKEND=Agg",
    ),
)
```

* This preserves the Parsl workflow and usually gives **excellent throughput for many small tasks** while benefiting from Slurm’s queueing/backfill for lots of short jobs.
* Ensure `address` is resolvable from compute nodes (use `address_by_hostname()` or `address_by_interface('ib0')` depending on site). ([Parsl][1])

### B) Run Parsl **inside** a Slurm allocation (no inbound connections)

If your site blocks inbound connections to the login node, submit a single Slurm job that runs your Parsl driver on node 0 and uses `SrunLauncher`. In this pattern the HTEX address can remain loopback and all traffic is intra‑allocation. Example patterns are documented by NERSC. ([NERSC Documentation][6])

### C) If you truly want “one scheduler job per unit of compute”

* **Job arrays (no Parsl):** For embarrassingly parallel tasks, create an array script that calls your CLI once per element (e.g., per target/sampler/seed), then run a tiny Python step to aggregate parquet outputs. This is the purest “let Slurm schedule it” approach and avoids ZMQ entirely. (Many centers recommend arrays for such workloads.) ([rc-docs.northeastern.edu][7])

* **WorkQueueExecutor:** Keep Parsl’s Python DAG, but start many **`work_queue_worker`** processes under Slurm (often via a Slurm job array). Workers come and go, and Parsl feeds tasks to whichever are alive. This is a good fit when you want scheduler control over worker lifetimes and counts but still want Parsl to manage task/data dependencies. ([Parsl][8])

> There is no built‑in “SlurmExecutor that submits one Slurm job per Parsl task.” The standard model is pilot jobs (HTEX) or external worker pools (WorkQueue/TaskVine/Flux). ([Parsl][3])

---

## Concrete changes I recommend for **lambda_hat**

1. **Local card (`type: local`)**

   * Add an `address` option to your YAML (default `127.0.0.1`) and pass it to HTEX.
   * Pin `worker_port_range` and `interchange_port_range`.
   * Use `start_method: spawn`.
     This eliminates the “ZMQ URL not viable” flakiness on macOS. ([Parsl][1])

2. **Slurm card (`type: slurm`)**

   * Decide between **A** (pilot jobs: many small blocks) and **B** (run inside allocation).
   * For **A**: `nodes_per_block=1`, `min_blocks=0`, `max_blocks` large, short `walltime`, `SrunLauncher()`, `address=address_by_hostname()` or `address_by_interface("ib0")`.
   * For **B**: run your Parsl driver under `sbatch`, use `SrunLauncher()` only, and keep address/loopback internal as shown by the NERSC example. ([NERSC Documentation][6])

3. **(Optional) WorkQueue mode**

   * Add a second executor profile using `WorkQueueExecutor`. Provide a small helper `sbatch` to launch `work_queue_worker` via an array (`min_workers=0, max_workers=...`). Use this profile when you want Slurm to elastically provide workers without holding big long‑lived allocations. ([Parsl][8])

4. **Docs + diagnostics**

   * In your README, add a short “Networking FAQ” that calls out `address=` and port‑pinning and shows how to select an interface. Mention the “compile pyzmq from source” fallback if the site’s prebuilt `pyzmq` is incompatible. ([Parsl][9])

---

## Will a multiprocess executor fix the plotting/x64 flakiness?

Very likely *yes* for the plotting: separate **processes** isolate Matplotlib/ArviZ state and avoid the thread‑safety issues you hit with the ThreadPool. You’ve already set `MPLBACKEND=Agg`; combine that with process isolation and you shouldn’t need locks. (Keep your `rcParams['text.usetex']=False` change.) For the **x64 dtype drift**, processes also help by giving each run its own `JAX_ENABLE_X64` environment and JAX config, rather than sharing process‑global state across threads.

HTEX on a single node with `LocalProvider` is essentially a managed **ProcessPool with better control over workers**—that’s why Parsl’s quickstart compares HTEX to `ProcessPoolExecutor`. ([Parsl][10])

---

## TL;DR decision guide

* **“I want minimal local parallelism and no network surprises.”**
  Use **HTEX + LocalProvider** with `address="127.0.0.1"`, pinned port ranges, and `start_method="spawn"`. No extra daemons; robust on macOS. ([Parsl][1])

* **“I’m on Slurm and want many cheap small jobs with the Parsl DAG.”**
  Keep **HTEX + SlurmProvider** but set `nodes_per_block=1`, short `walltime`, adaptive scaling (`min_blocks=0`, big `max_blocks`). Slurm still schedules many small jobs; Parsl fills them. ([Parsl][4])

* **“I want the scheduler to elastically provide workers, not long pilots.”**
  Consider **WorkQueueExecutor** and launch `work_queue_worker` via a Slurm job array. ([Parsl][8])

* **“I truly want one Slurm job per task and don’t need Parsl’s DAG.”**
  Use **Slurm job arrays** invoking `lambda-hat` CLI per element; aggregate afterward. ([rc-docs.northeastern.edu][7])


[1]: https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html?utm_source=chatgpt.com "parsl.executors.HighThroughputExecutor - Read the Docs"
[2]: https://gitlab.ebrains.eu/noelp/parsl/-/blob/3687404abf2eb00d3baa5076c70813f5d7bf8646/docs/userguide/configuring.rst?utm_source=chatgpt.com "parsl - docs - userguide - configuring.rst"
[3]: https://parsl.readthedocs.io/en/stable/userguide/overview.html?utm_source=chatgpt.com "Overview — Parsl 1.3.0-dev documentation - Read the Docs"
[4]: https://parsl-project.org/parslfest/2024/parslguts.pdf?utm_source=chatgpt.com "Parsl Guts"
[5]: https://gitlab.ebrains.eu/noelp/parsl/-/blob/master/parsl/addresses.py?utm_source=chatgpt.com "parsl/addresses.py · master - Klaus Noelp"
[6]: https://docs.nersc.gov/jobs/workflow/parsl/?utm_source=chatgpt.com "Parsl"
[7]: https://rc-docs.northeastern.edu/en/latest/runningjobs/slurmarray.html?utm_source=chatgpt.com "Slurm Jobs Array"
[8]: https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.WorkQueueExecutor.html?utm_source=chatgpt.com "parsl.executors.WorkQueueExecutor - Read the Docs"
[9]: https://parsl.readthedocs.io/en/stable/faq.html?utm_source=chatgpt.com "FAQ — Parsl 1.3.0-dev documentation"
[10]: https://parsl.readthedocs.io/en/stable/quickstart.html?utm_source=chatgpt.com "Quickstart — Parsl 1.3.0-dev documentation"
