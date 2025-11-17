Great observation—and yes: moving to *consistent, configurable logging* will make life easier across your CLI, tests, JAX code, Parsl/distributed runs, and containers. Below is a concrete, repo‑ready plan with code you can copy in, plus notes on how it plays with each execution context.

---

## 0) Goals & principles

* **One way to talk**: Library code *never* uses `print()`. Use `logging.getLogger(__name__)` everywhere. Configure logging only in entrypoints (CLI, notebooks, benchmarks) — not inside the library. ([Python documentation][1])
* **Human vs. machine outputs**: Human-facing CLI output is fine to *render* to the console (pretty/colored), but it should still go through logging (INFO/DEBUG) or a dedicated console writer; machine-facing output should be structured (JSON). JSON logs are first-class citizens for aggregation. ([PyPI][2])
* **Execution-context aware**: Prefer stdout/stderr streams (containers/K8s); Queue-based handlers for multiprocessing/Parsl; special tools for JAX inside `jit` (i.e., `jax.debug.print`). ([Kubernetes][3])

---

## 1) Repository changes (single PR you can open now)

### 1.1 Add a central logger config module

Create **`lambda_hat/logging_config.py`**:

```python
# lambda_hat/logging_config.py
from __future__ import annotations

import json
import logging
import logging.config
import os
import queue
from logging.handlers import QueueHandler, QueueListener

try:
    from pythonjsonlogger import jsonlogger  # pip install python-json-logger
except Exception:  # optional dependency
    jsonlogger = None

_QUEUE = None
_LISTENER: QueueListener | None = None

def configure_logging(
    *,
    level: str | None = None,
    format: str | None = None,
    json_mode: bool | None = None,
    capture_warnings: bool = True,
) -> None:
    """
    Call once from entrypoints (CLI, notebooks, servers, scripts).
    Library code MUST NOT call this.
    """
    global _QUEUE, _LISTENER

    # Defaults from env, then sensible fallbacks
    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    json_mode = json_mode if json_mode is not None else os.getenv("LOG_FORMAT") == "json"

    # Base formatter(s)
    if json_mode and jsonlogger:
        fmt_name = "json"
        fmt_def = {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    else:
        fmt_name = "plain"
        fmt_def = {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            "datefmt": "%H:%M:%S",
        }

    # Handlers: stream only; containers and CI should log to stdout/stderr.
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": fmt_name,
        }
    }

    # Optional: swap in JSON formatter
    formatters = {
        "plain": {"format": fmt_def["format"], "datefmt": fmt_def.get("datefmt")},
    }
    if json_mode and jsonlogger:
        formatters["json"] = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": fmt_def["format"],
        }
    else:
        formatters["json"] = formatters["plain"]  # fallback

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)

    if capture_warnings:
        logging.captureWarnings(True)  # route warnings -> logging under 'py.warnings'

def start_mp_logging_queue(base_handlers: list[logging.Handler] | None = None):
    """
    Multiprocessing-safe logging: use QueueHandler in workers, QueueListener in parent.
    Call *in the parent process* once, then call get_worker_queue_handler() in workers.
    """
    global _QUEUE, _LISTENER
    if _QUEUE is None:
        _QUEUE = queue.Queue()
        handlers = base_handlers or list(logging.getLogger().handlers)
        _LISTENER = QueueListener(_QUEUE, *handlers, respect_handler_level=True)
        _LISTENER.start()

def get_worker_queue_handler() -> logging.Handler:
    """Return a QueueHandler to install in child processes."""
    if _QUEUE is None:
        raise RuntimeError("Call start_mp_logging_queue() in the parent first.")
    return QueueHandler(_QUEUE)

def shutdown_logging():
    global _LISTENER
    if _LISTENER:
        _LISTENER.stop()
        _LISTENER = None
```

* Uses `dictConfig` (standard) and routes warnings via `logging.captureWarnings(True)`. ([Python documentation][4])
* Supports JSON logs via `python-json-logger` (optional dependency) for machine parsing. ([PyPI][2])
* Provides a **Queue-based** path for multiprocessing/Parsl workers (`QueueHandler`/`QueueListener`) to avoid interleaved logs. ([Python documentation][5])

### 1.2 Adopt module-level loggers in the library

At the top of each module:

```python
import logging
log = logging.getLogger(__name__)
```

Replace `print(...)` with `log.info(...)`/`log.debug(...)` etc. (see level map below). The official HOWTO recommends module-level loggers named `__name__`. ([Python documentation][1])

### 1.3 Entry points configure (and only entry points)

* **CLI** (`lambda_hat/commands/*.py`): call `configure_logging()` as the first thing in `main()`.
* **Parsl/distributed launcher**: call `configure_logging()` *once in the parent*, then `start_mp_logging_queue()`; in worker bootstrap code, attach `get_worker_queue_handler()`. Parsl itself emits under the `parsl` logger; you can disable Parsl’s own file logging with `Config(initialize_logging=False)` to keep ownership of formatting. ([parsl.readthedocs.io][6])
* **Notebooks**: prefer `configure_logging(level="INFO")` in the first cell. (It’s safer than ad‑hoc `basicConfig()` and works in Jupyter.) For pretty human output, you can later switch to `rich`’s `RichHandler`, but start simple. ([rich.readthedocs.io][7])
* **Containers/Kubernetes**: do *not* log to files by default. Stream to stdout/stderr and let the runtime collect. If you want structured ingestion, set `LOG_FORMAT=json` and let your log agent (e.g., Loki/Fluent Bit) parse JSON. ([Kubernetes][3])

### 1.4 Lint & CI guardrails

* Add Ruff’s `T201` rule to ban `print()` in non-test code. In `pyproject.toml`:

  ```toml
  [tool.ruff]
  select = ["E", "F", "I", "T201"]
  ignore = []
  ```

  (Ruff documents the rule and notes that auto-fixes can be unsafe; run it as **non‑fixing** in CI.) ([Astral Docs][8])

* In tests, prefer `caplog` for assertions on logs; avoid reconfiguring the *root* logger mid-test (it can break `caplog`). ([pytest][9])

### 1.5 Level policy (cut/paste into CONTRIBUTING.md)

* `DEBUG`: shapes/dtypes once per init; sampler hyperparams; seeds; per-step loss only in small tests.
* `INFO`: start/stop of major phases (compile, warmup, sample), progress summaries (every N steps), final metrics.
* `WARNING`: recoverable issues, deprecated flags.
* `ERROR`: failed steps/retries.
* `CRITICAL`: process is about to abort.

The official HOWTO lists when to use each level; mapping above keeps logs useful without spam. ([Python documentation][1])

### 1.6 JAX-specific guidance

* Inside `@jax.jit`/`vmap`/`grad`, **don’t** call Python logging or `print()` to inspect dynamic values — you’ll hit tracers or reorderings. Use `jax.debug.print()` (and `ordered=True` when you truly need order). For logging integration, you can wrap `jax.debug.callback` to emit via the logger. ([docs.jax.dev][10])

Example:

```python
import jax, jax.numpy as jnp, logging
log = logging.getLogger(__name__)

@jax.jit
def step(params, x):
    # Log dynamic values at runtime (JIT-safe):
    jax.debug.print("step: mean={m}", m=jnp.mean(x))

    # Or: send to logging via callback (still runs at runtime)
    jax.debug.callback(lambda v: log.debug("step: var=%s", v), jnp.var(x))

    return params
```

Equinox docs echo these debugging tools. ([docs.kidger.site][11])

---

## 2) Execution-context specifics (what to watch out for)

### CLI & local dev

* Use `configure_logging()`; `LOG_LEVEL=DEBUG` locally; `LOG_FORMAT=human|json`. Prefer human format for consoles; JSON for CI capture. `logging.config.dictConfig` supports both easily. ([Python documentation][4])

### PyTest

* Tests should *not* reconfigure the root logger; if you must, append handlers instead of replacing them. Assert on logs via `caplog`. ([pytest][9])

### Parsl / multiprocessing

* Parent process: `configure_logging()` → `start_mp_logging_queue()`.
* Child processes/workers: attach `get_worker_queue_handler()` to the module logger at start; don’t add stream handlers in workers. Python’s cookbook shows this pattern; it prevents garbled interleaved lines and is safer across processes. ([Python documentation][5])
* Parsl can avoid its own file logging with `initialize_logging=False`, emitting under `parsl` logger into your configuration. ([parsl.readthedocs.io][6])

### Notebooks

* Call `configure_logging(level="INFO")` once per kernel. If you later want pretty logs, switch the handler to `rich.logging.RichHandler` in your configuration, which renders cleanly in Jupyter (Kedro documents this pattern). ([rich.readthedocs.io][7])

### Containers / Kubernetes

* *Never* write rotating files inside containers unless you must. Stream to stdout/stderr; let the platform collect. Use JSON logs for structured ingestion. ([Kubernetes][3])

### Warnings

* Route `warnings.warn(...)` to logging via `logging.captureWarnings(True)` to keep a single stream and make warnings visible in CI/K8s. ([Python documentation][12])

### Optional: trace–log correlation (OpenTelemetry)

* If you add tracing later, OpenTelemetry’s logging bridge can inject trace/span IDs into your log records and export via OTLP — without changing your logging calls. Keep this in mind for future observability work. ([opentelemetry-python-contrib.readthedocs.io][13])

---

## 3) Migration steps (1–2 short PRs)

1. **Scaffold & CI**

   * Add `logging_config.py` and call `configure_logging()` from CLI entrypoints / runners.
   * Enable Ruff rule `T201` (ban prints) for `lambda_hat/*` except `tests/` and `notebooks/`. ([Astral Docs][8])

2. **Codemod**

   * Replace `print()` in library modules with `log.info/debug/warning/error`; adopt the level policy above.
   * Where you currently `print` from inside JAX-jitted code, replace with `jax.debug.print` (or `jax.debug.callback` → logger) as appropriate. ([docs.jax.dev][10])

3. **Parsl/distributed**

   * In your runners, start the queue listener, and in worker bootstraps attach a `QueueHandler`. (Short helper is already in the snippet.) ([Python documentation][5])

4. **Tests**

   * Use `caplog` to assert on logging behavior; avoid resetting the root logger mid-test. ([pytest][9])

5. **Containers**

   * In Docker/K8s, set `LOG_FORMAT=json` and keep only a stream handler. Your collector can parse JSON reliably. ([Kubernetes][3])

---

## 4) Policy: when *prints* are allowed

* **Never** in library code. (CI enforces this via Ruff T201.) ([Astral Docs][8])
* **JAX inside `jit`**: use `jax.debug.print` for dynamic values; Python `print()` is only for static, trace-time info and often misleading there. ([docs.jax.dev][10])
* **One-off user-facing CLI summaries**: Prefer logging at `INFO`, but if you want “bare” user output (e.g., a table), write it to stdout explicitly and keep logs separate.

---

## 5) Small helpers you’ll likely want

**A context injector** (request/run IDs, sampler name, seed) without switching to a new library:

```python
# lambda_hat/logging_context.py
import contextvars, logging

ctx = contextvars.ContextVar("lambda_hat_log_ctx", default={})

class ContextFilter(logging.Filter):
    def filter(self, record):
        for k, v in ctx.get().items():
            setattr(record, k, v)
        return True

def bind(**kv):
    d = dict(ctx.get())
    d.update(kv)
    ctx.set(d)

def install_context_filter():
    f = ContextFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(f)
```

Use `bind(run_id=..., sampler=..., seed=...)` at the start of a run so every line carries those fields; in JSON mode they’ll appear as additional keys. If you later adopt `structlog`, its `contextvars` helpers provide the same idea with batteries included. ([structlog][14])

---

## FAQ

**“Is logging better just for verbosity control?”**
Yes — and more. You get standardized levels, multiple outputs (console/JSON/file/OTLP), warning capture, and testability. The stdlib HOWTO explains the idioms and level guidance. ([Python documentation][1])

**“Do we need to think about execution contexts?”**
Absolutely. The plan above treats:

* **JAX** (use `jax.debug.print`/callbacks inside `jit`),
* **Parsl/multiprocessing** (QueueHandler/Listener),
* **PyTest** (`caplog`, don’t reconfigure root),
* **Kubernetes/containers** (stdout/stderr, JSON), and
* **Notebooks** (one-time config; optional Rich console). ([Python documentation][5])

---

## TL;DR checklist you can drop in the PR description

* [ ] Add `lambda_hat/logging_config.py`; call `configure_logging()` in entrypoints. ([Python documentation][4])
* [ ] Replace `print()` with module loggers and appropriate levels. ([Python documentation][1])
* [ ] Enforce Ruff `T201` (no prints) in CI. ([Astral Docs][8])
* [ ] Use `jax.debug.print` in jitted paths; avoid Python logging there. ([docs.jax.dev][10])
* [ ] Parent runner starts `start_mp_logging_queue()`; workers attach `QueueHandler`. ([Python documentation][5])
* [ ] Route warnings to logging (`logging.captureWarnings(True)`). ([Python documentation][12])
* [ ] Default to stdout/stderr handlers; `LOG_FORMAT=json` for containers. ([Kubernetes][3])
* [ ] Tests use `caplog`; don’t re‑init the root logger mid-test. ([pytest][9])

If you’d like, I can prep the codemod and the minimal PR diff based on your tree in a follow-up.

[1]: https://docs.python.org/3/howto/logging.html?utm_source=chatgpt.com "Logging HOWTO"
[2]: https://pypi.org/project/python-json-logger/?utm_source=chatgpt.com "python-json-logger"
[3]: https://kubernetes.io/docs/concepts/cluster-administration/logging/?utm_source=chatgpt.com "Logging Architecture"
[4]: https://docs.python.org/3/library/logging.config.html?utm_source=chatgpt.com "logging.config — Logging configuration"
[5]: https://docs.python.org/3/howto/logging-cookbook.html?utm_source=chatgpt.com "Logging Cookbook — Python 3.14.0 documentation"
[6]: https://parsl.readthedocs.io/en/stable/stubs/parsl.config.Config.html?utm_source=chatgpt.com "parsl.config.Config — Parsl 1.3.0-dev documentation"
[7]: https://rich.readthedocs.io/en/latest/logging.html?utm_source=chatgpt.com "Logging Handler — Rich 14.1.0 documentation"
[8]: https://docs.astral.sh/ruff/rules/print/?utm_source=chatgpt.com "print (T201) | Ruff"
[9]: https://docs.pytest.org/en/stable/how-to/logging.html?utm_source=chatgpt.com "How to manage logging"
[10]: https://docs.jax.dev/en/latest/debugging.html?utm_source=chatgpt.com "Introduction to debugging"
[11]: https://docs.kidger.site/equinox/api/debug/?utm_source=chatgpt.com "Debugging tools - Equinox"
[12]: https://docs.python.org/3/library/warnings.html?utm_source=chatgpt.com "warnings — Warning control"
[13]: https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/logging/logging.html?utm_source=chatgpt.com "OpenTelemetry Logging Instrumentation"
[14]: https://www.structlog.org/en/stable/api.html?utm_source=chatgpt.com "API Reference — structlog 25.5.0 documentation"
