# Snakefile
from pathlib import Path
from omegaconf import OmegaConf
import hashlib, json

# Register OmegaConf resolvers
from lambda_hat import omegaconf_support  # noqa: F401

configfile: "config/experiments.yaml"

CONF = Path("lambda_hat/conf")
STORE = config.get("store_root", "runs")
JAX64 = bool(config.get("jax_enable_x64", True))

def compose_build_cfg(t):
    cfg = OmegaConf.load(CONF / "workflow.yaml")
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.load(CONF / "model" / f"{t['model']}.yaml"),
        OmegaConf.load(CONF / "data" / f"{t['data']}.yaml"),
        OmegaConf.load(CONF / "teacher" / f"{t.get('teacher','_null')}.yaml"),
        {"target": {"seed": t["seed"]},
         "jax": {"enable_x64": JAX64},
         "store": {"root": STORE}}
    )
    if "overrides" in t: cfg = OmegaConf.merge(cfg, t["overrides"])
    return cfg

def compose_sample_cfg(tid, s):
    base = OmegaConf.load(CONF / "sample" / "base.yaml")
    smpl = OmegaConf.load(CONF / "sample" / "sampler" / f"{s['name']}.yaml")
    cfg = OmegaConf.merge(base, smpl, {
        "target_id": tid,
        "jax": {"enable_x64": JAX64},
        "store": {"root": STORE},
        "runtime": {"seed": s.get("seed", 12345)}
    })
    if "overrides" in s:
        cfg = OmegaConf.merge(cfg, {"sampler": {s["name"]: s["overrides"]}})
    return cfg

def target_id_for(cfg):
    blob = json.dumps(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True), sort_keys=True)
    return "tgt_" + hashlib.sha256(blob.encode()).hexdigest()[:12]

def run_id_for(cfg):
    # free to choose: sha1 of fully resolved sample cfg
    blob = json.dumps(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]

TARGETS = []
for t in config["targets"]:
    bcfg = compose_build_cfg(t)
    tid = target_id_for(bcfg)
    TARGETS.append((tid, bcfg))

RUNS = []
for tid, bcfg in TARGETS:
    for s in config["samplers"]:
        scfg = compose_sample_cfg(tid, s)
        rid = run_id_for(scfg)
        RUNS.append((tid, s["name"], rid, scfg))

rule all:
    input:
        [f"{STORE}/targets/{tid}/run_{sampler}_{rid}/analysis.json" for tid,sampler,rid,_ in RUNS]

rule cfg_build:
    output: cfg = temp("temp_config/build/{tid}.yaml")
    run:
        tid = wildcards.tid
        bcfg = next(c for (t,c) in TARGETS if t==tid)
        Path(output.cfg).parent.mkdir(parents=True, exist_ok=True)
        Path(output.cfg).write_text(OmegaConf.to_yaml(bcfg))

rule build_target:
    input: cfg = rules.cfg_build.output.cfg
    output: meta = f"{STORE}/targets/{{tid}}/meta.json"
    params:
        tid = lambda wc: wc.tid,
        tdir = lambda wc: f"{STORE}/targets/{wc.tid}"
    log: f"logs/build_target/{{tid}}.log"
    shell:
        r"""
        mkdir -p $(dirname {log})
        JAX_ENABLE_X64={JAX64} uv run python -m lambda_hat.entrypoints.build_target \
          --config-yaml {input.cfg} \
          --target-id {params.tid} \
          --target-dir {params.tdir} > {log} 2>&1
        """

rule cfg_sample:
    output: cfg = temp("temp_config/sample/{tid}_{sampler}_{rid}.yaml")
    run:
        tid, sampler, rid = wildcards.tid, wildcards.sampler, wildcards.rid
        scfg = next(c for (t,s,r,c) in RUNS if t==tid and s==sampler and r==rid)
        Path(output.cfg).parent.mkdir(parents=True, exist_ok=True)
        Path(output.cfg).write_text(OmegaConf.to_yaml(scfg))

rule run_sampler:
    input:
        meta = rules.build_target.output.meta,
        cfg  = rules.cfg_sample.output.cfg
    output:
        analysis = f"{STORE}/targets/{{tid}}/run_{{sampler}}_{{rid}}/analysis.json"
    params:
        tid = lambda wc: wc.tid,
        rdir = lambda wc: f"{STORE}/targets/{wc.tid}/run_{wc.sampler}_{wc.rid}"
    log: f"logs/run_sampler/{{tid}}_{{sampler}}_{{rid}}.log"
    shell:
        r"""
        mkdir -p $(dirname {log})
        JAX_ENABLE_X64={JAX64} uv run python -m lambda_hat.entrypoints.sample \
          --config-yaml {input.cfg} \
          --target-id {params.tid} \
          --run-dir {params.rdir} > {log} 2>&1
        """