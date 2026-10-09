"""Runs: the raw outputs of one LLM under one inference configuration.

A run directory holds

    run.json        the configuration and the prompt template
    outputs.jsonl   one line per example, either
                    {"id": ..., "label_logprobs": [...]}   (mode "logits")
                    {"id": ..., "samples": [...]}          (mode "samples")

``collect`` fills a run and can be interrupted and restarted at any time;
``score_run`` turns a run into a ScoreTable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from cp4configshift.backends import BACKENDS
from cp4configshift.data import Example
from cp4configshift.prompts import Template
from cp4configshift.scores import (
    contains_alias, lac, lofreecp, logit_margin, mc_pools, open_pools, parse_choice,
    self_consistency, sentence_embedder, strip_thinking,
)
from cp4configshift.table import ScoreTable


@dataclass
class RunConfig:
    """One inference configuration.

    mode "logits" records the log-probabilities of the option labels
    (multiple-choice only); the temperature is then applied when scoring, so
    one run serves every temperature. mode "samples" records ``n_samples``
    completions drawn at ``temperature``.
    """
    model: str                      # HF model id or path, .gguf file, or API model name
    template: str                   # template name, for the record
    mode: str = "logits"            # "logits" or "samples"
    backend: str = "hf"             # "hf", "llamacpp" or "openai"
    temperature: float = 1.0
    n_samples: int = 20
    max_new_tokens: int = 8
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    backend_args: dict = field(default_factory=dict)   # e.g. {"n_ctx": 4096} or {"base_url": ...}

    def make_backend(self):
        return BACKENDS[self.backend](self.model, seed=self.seed, **self.backend_args)


@dataclass
class Run:
    path: Path
    config: RunConfig
    template: Template
    outputs: dict[str, dict]        # example id -> output record


def _read_outputs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    lines = path.read_text().split("\n")
    # the last element is "" for a complete file, or a line cut off by a crash
    return {rec["id"]: rec for rec in map(json.loads, filter(None, lines[:-1]))}


def collect(examples: list[Example], template: Template, config: RunConfig,
            run_dir: str | Path, backend=None) -> Path:
    """Run ``config`` on every example not yet in ``run_dir``.

    Pass ``backend`` to reuse an already loaded model across runs; any object
    with ``label_logprobs`` / ``sample`` methods like those in
    ``cp4configshift.backends`` works.
    """
    from tqdm import tqdm

    if config.mode not in ("logits", "samples"):
        raise ValueError(f"unknown mode {config.mode!r}")
    if config.mode == "logits" and not all(ex.is_multiple_choice for ex in examples):
        raise ValueError("mode 'logits' needs multiple-choice examples")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": asdict(config), "template": asdict(template)}
    meta_path = run_dir / "run.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError(f"{run_dir} already holds a run with a different configuration or template")
    meta_path.write_text(json.dumps(meta, indent=2))

    out_path = run_dir / "outputs.jsonl"
    done = _read_outputs(out_path)
    if out_path.exists() and not out_path.read_bytes().endswith(b"\n"):   # drop a line cut off by a crash
        out_path.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    todo = [ex for ex in examples if ex.id not in done]
    if not todo:
        return run_dir

    backend = backend or config.make_backend()
    with open(out_path, "a") as f:
        for ex in tqdm(todo, desc=str(run_dir)):
            prompt = template.render(ex)
            if config.mode == "logits":
                rec = {"id": ex.id,
                       "label_logprobs": np.asarray(backend.label_logprobs(prompt, template.labels)).tolist()}
            else:
                rec = {"id": ex.id,
                       "samples": backend.sample(prompt, config.n_samples, config.temperature,
                                                 config.top_p, config.top_k, config.max_new_tokens)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
    return run_dir


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    return Run(run_dir, RunConfig(**meta["config"]), Template(**meta["template"]),
               _read_outputs(run_dir / "outputs.jsonl"))


def score_run(
    run: str | Path | Run,
    examples: list[Example],
    score: str,
    temperature: float | None = None,
    embed: Callable[[list[str]], np.ndarray] | None = None,
    is_correct: Callable[[str, Example], bool] = contains_alias,
    strip_think: bool = True,
) -> ScoreTable:
    """Nonconformity scores of ``examples`` under ``run``.

    score: "lac" or "logit_margin" for logits runs (optionally at
    ``temperature``); "self_consistency" or "lofreecp" for sample runs.
    ``embed`` is the LoFreeCP sentence embedder (default all-MiniLM-L6-v2),
    ``is_correct`` the open-ended answer matcher, and ``strip_think`` removes
    ``<think>`` blocks from samples before parsing (set False to reproduce the
    paper exactly).
    """
    run = run if isinstance(run, Run) else load_run(run)
    missing = [ex.id for ex in examples if ex.id not in run.outputs]
    if missing:
        raise ValueError(f"{run.path} has no output for {len(missing)} examples (e.g. {missing[0]!r})")
    recs = [run.outputs[ex.id] for ex in examples]

    if run.config.mode == "logits":
        fn = {"lac": lac, "logit_margin": logit_margin}.get(score)
        if fn is None:
            raise ValueError(f"score {score!r} needs a samples run; logits runs support lac, logit_margin")
        table = fn(np.array([r["label_logprobs"] for r in recs]), examples,
                   1.0 if temperature is None else temperature)
    else:
        if temperature is not None:
            raise ValueError("the temperature of a samples run is fixed at collection time")
        texts = [[strip_thinking(s) if strip_think else s for s in r["samples"]] for r in recs]
        if examples[0].is_multiple_choice:
            labels = run.template.labels
            pools = mc_pools([[parse_choice(s, labels) for s in row] for row in texts], examples)
        else:
            pools = open_pools(texts, examples, is_correct)
        if score == "self_consistency":
            table = self_consistency(pools)
        elif score == "lofreecp":
            table = lofreecp(pools, embed or sentence_embedder())
        else:
            raise ValueError(f"score {score!r} needs a logits run; samples runs support self_consistency, lofreecp")

    table.info.update(run=str(run.path), model=run.config.model, template=run.config.template)
    return table
