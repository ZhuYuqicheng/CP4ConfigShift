"""Example: the paper's results for the white-box scores (LAC, logit margin) on
MMLU and MedMCQA under prompt-template, temperature and quantization shift.

    python examples/paper/reproduce_whitebox.py                  # from the shipped log-probs
    python examples/paper/reproduce_whitebox.py --runs runs      # from runs you collected

The shipped files ``{mmlu,medmcqa}_logprobs.npz`` hold the label log-probabilities
we collected, plus example ids, answers and MMLU subjects. Keys
``prompt_logits/...`` and ``quantization_logits/...`` mirror the run directories
written by examples/configs/mcqa_{prompt,quantization}_logits.yaml; with
``--runs`` those are read from <runs>/<dataset>/<key> instead.

Protocol (Sec. 4.1): 10 calibration/test splits with seed 42; MMLU draws 30 + 30
questions per subject, MedMCQA 800 + 800 at random. Temperature shift uses the
Baseline prompt at T in {0.5, 0.7, 1.0, 1.3, 1.5}. For the paper we ran one
forward pass per temperature (shipped as ``temperature_logits/<model>/T<T>``,
holding log-softmax(logits / T)); with ``--runs`` the temperature is instead
applied to the Baseline run when scoring. 

"""

import argparse
import csv
from pathlib import Path

import numpy as np

import cp4configshift as cs

HERE = Path(__file__).resolve().parent
SCORES = {"lac": cs.lac, "logit_margin": cs.logit_margin}
TARGETS = [0.7, 0.8, 0.9, 0.95]
RHOS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
TEMPERATURES = [0.5, 0.7, 1.0, 1.3, 1.5]
N_CAL = {"mmlu": 30, "medmcqa": 800}      # per subject for MMLU
N_TRIALS, SEED = 10, 42


def load(dataset: str, runs: str | None):
    """Examples and {run key: (n, 4) label log-probs}."""
    z = np.load(HERE / f"{dataset}_logprobs.npz")
    subjects = z["subjects"] if "subjects" in z.files else [""] * len(z["ids"])
    examples = [cs.Example(str(i), "", int(a), list("ABCD"), {"subject": str(s)})
                for i, a, s in zip(z["ids"], z["answers"], subjects)]
    keys = [k for k in z.files if "/" in k]
    if runs is None:
        return examples, {k: z[k] for k in keys}
    logprobs = {}
    for k in keys:
        if k.startswith("temperature_logits/"):
            continue
        out = cs.load_run(Path(runs, dataset, k)).outputs
        logprobs[k] = np.array([out[ex.id]["label_logprobs"] for ex in examples])
    return examples, logprobs


def shift_axes(logprobs: dict) -> dict:
    """{axis: {model: {configuration: (run key, temperature)}}}."""
    prompt = {}
    for k in logprobs:
        kind, model, template = k.split("/")
        if kind == "prompt_logits":
            prompt.setdefault(model, {})[template] = (k, 1.0)
    quant = {k.split("/")[1].rsplit("-", 1)[1]: (k, 1.0) for k in logprobs if k.startswith("quantization_logits/")}
    return {
        "Prompt template": prompt,
        "Temperature": {m: {f"T{T}": (f"temperature_logits/{m}/T{T}", 1.0) if f"temperature_logits/{m}/T{T}" in logprobs
                                else (f"prompt_logits/{m}/Baseline", T) for T in TEMPERATURES} for m in prompt},
        "Quantization": {"Qwen3.5-9B": quant},
    }


def nested_splits(examples, dataset):
    """For each rho: 10 (calibration, test) pairs whose calibration sets are the
    first round(rho * n_cal) examples of one maximal calibration draw, with the
    test set held fixed across rho. At rho = 1 these equal cs.make_splits."""
    rng = np.random.default_rng(SEED)
    n_cal, n = N_CAL[dataset], len(examples)
    out = {rho: [] for rho in RHOS}
    if dataset == "medmcqa":
        for _ in range(N_TRIALS):
            perm = rng.permutation(n)
            test = np.sort(perm[n_cal:2 * n_cal])
            for rho in RHOS:
                out[rho].append((np.sort(perm[:max(1, round(rho * n_cal))]), test))
        return out
    subjects = np.array([ex.meta["subject"] for ex in examples])
    members = [np.flatnonzero(subjects == s) for s in sorted(set(subjects))]
    for _ in range(N_TRIALS):
        draws = [m[rng.permutation(len(m))] for m in members]
        test = np.concatenate([d[n_cal:2 * n_cal] for d in draws])
        for rho in RHOS:
            k = max(1, round(rho * n_cal))
            out[rho].append((np.concatenate([d[:k] for d in draws]), test))
    return out


def paper_metrics(result: cs.ShiftResult, target: float) -> dict:
    cov, size = result.coverage, result.set_size
    C = len(result.configs)
    base = np.empty(C)            # i.i.d. set size of each test configuration
    for j in range(C):
        ok = cov[j, j] >= target
        base[j] = size[j, j][ok].mean() if ok.any() else size[j, j].mean()
    infl = size / base[None, :, None]
    diag = np.eye(C, dtype=bool)
    row = {}
    for name, mask in (("iid", diag), ("shift", ~diag)):
        c, f = cov[mask].ravel(), infl[mask].ravel()
        row[f"U_{name}"] = float((c < target - 0.02).mean())
        row[f"Infl_{name}"] = float(f[c >= target].mean()) if (c >= target).any() else float("nan")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="read log-probs from collected runs in <runs>/<dataset>/...")
    ap.add_argument("--out", default=".", help="directory for the CSV files")
    args = ap.parse_args()

    target_rows, budget_rows = [], []
    for dataset in ("mmlu", "medmcqa"):
        examples, logprobs = load(dataset, args.runs)
        groups = [ex.meta["subject"] for ex in examples] if dataset == "mmlu" else None
        n_cal = N_CAL[dataset]
        splits = cs.make_splits(len(examples), n_cal, n_cal, N_TRIALS, seed=SEED, groups=groups)
        budgets = nested_splits(examples, dataset)
        for axis, models in shift_axes(logprobs).items():
            for model, configs in models.items():
                for score, fn in SCORES.items():
                    print(f"{dataset} | {axis} | {model} | {score}", flush=True)
                    tables = {c: fn(logprobs[k], examples, temperature=T) for c, (k, T) in configs.items()}
                    key = {"shift": axis, "dataset": dataset, "model": model, "score": score}
                    for target in TARGETS:
                        res = cs.evaluate_shift(tables, round(1 - target, 4), splits)
                        target_rows.append({**key, "target": target, **paper_metrics(res, target)})
                    for rho in RHOS:
                        res = cs.evaluate_shift(tables, 0.1, budgets[rho])
                        budget_rows.append({**key, "rho": rho, **paper_metrics(res, 0.9)})

    for name, rows in (("target_sweep.csv", target_rows), ("budget_sweep.csv", budget_rows)):
        path = Path(args.out) / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")

    print(f"\nmean over (dataset, model, score) at target 0.9:\n{'shift':<17}{'U_iid':>7}{'U_shift':>9}{'Infl_iid':>10}{'Infl_shift':>12}")
    for axis in ("Prompt template", "Temperature", "Quantization"):
        r = [x for x in target_rows if x["shift"] == axis and x["target"] == 0.9]
        m = {c: np.nanmean([x[c] for x in r]) for c in ("U_iid", "U_shift", "Infl_iid", "Infl_shift")}
        print(f"{axis:<17}{m['U_iid']:7.3f}{m['U_shift']:9.3f}{m['Infl_iid']:10.3f}{m['Infl_shift']:12.3f}")


if __name__ == "__main__":
    main()
