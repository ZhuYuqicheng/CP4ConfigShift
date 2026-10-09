"""Command line: collect the runs of a configuration sweep.

    cp4cs collect sweep.yaml [--dataset D] [--output-dir O] [--limit N]

sweep.yaml names the data and any RunConfig fields; a list sweeps over its values::

    dataset: data/mmlu.jsonl
    templates: examples/configs/templates_mcqa.yaml
    output_dir: runs/mmlu_prompt_logits
    model: [meta-llama/Llama-3.2-1B-Instruct, Qwen/Qwen3.5-0.8B]
    template: [Baseline, formal, minimal]
    mode: logits

Each configuration is written to <output_dir>/<model name>/<template>, plus one
"<field>=<value>" path component for every other swept field. Configurations
that share a model reuse the loaded model.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from dataclasses import fields
from pathlib import Path

import yaml

from cp4configshift.data import load_examples
from cp4configshift.prompts import load_templates
from cp4configshift.runs import RunConfig, collect


def model_name(model: str) -> str:
    return Path(model).stem if model.endswith(".gguf") else model.rstrip("/").split("/")[-1]


def expand(sweep: dict, output_dir: str | Path) -> list[tuple[RunConfig, Path]]:
    names = {f.name for f in fields(RunConfig)}
    unknown = set(sweep) - names
    if unknown:
        raise ValueError(f"unknown sweep keys: {sorted(unknown)}")
    swept = [k for k in sweep if isinstance(sweep[k], list)]
    if sweep.get("mode", "logits") == "logits" and "temperature" in swept:
        raise ValueError("logits runs do not depend on temperature; apply it when scoring instead")
    runs = []
    for values in itertools.product(*(sweep[k] for k in swept)):
        cfg = RunConfig(**{**sweep, **dict(zip(swept, values))})
        parts = [model_name(cfg.model), cfg.template]
        parts += [f"{k}={getattr(cfg, k)}" for k in swept if k not in ("model", "template")]
        runs.append((cfg, Path(output_dir, *parts)))
    return runs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cp4cs")
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="collect the runs of a sweep file")
    c.add_argument("sweep")
    c.add_argument("--dataset", help="override the sweep's dataset")
    c.add_argument("--output-dir", help="override the sweep's output_dir")
    c.add_argument("--limit", type=int, help="only the first N examples (for a quick test)")
    args = ap.parse_args(argv)

    sweep = yaml.safe_load(open(args.sweep))
    examples = load_examples(args.dataset or sweep.pop("dataset"))[: args.limit]
    sweep.pop("dataset", None)
    templates = load_templates(sweep.pop("templates"))
    output_dir = args.output_dir or sweep.pop("output_dir")
    sweep.pop("output_dir", None)

    runs = expand(sweep, output_dir)
    key = lambda r: json.dumps([r[0].backend, r[0].model, r[0].backend_args, r[0].seed], sort_keys=True)
    for _, group in itertools.groupby(sorted(runs, key=key), key=key):
        group = list(group)
        backend = _LazyBackend(group[0][0])
        for cfg, run_dir in group:
            collect(examples, templates[cfg.template], cfg, run_dir, backend=backend)
        del backend
        gc.collect()


class _LazyBackend:
    """Loads the model on first use, so finished runs never load it."""

    def __init__(self, cfg: RunConfig):
        self.cfg, self.backend = cfg, None

    def __getattr__(self, name):
        if self.backend is None:
            self.backend = self.cfg.make_backend()
        return getattr(self.backend, name)


if __name__ == "__main__":
    main()
