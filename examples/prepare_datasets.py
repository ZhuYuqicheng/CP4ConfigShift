"""Download MMLU and MedMCQA and write them as CP4ConfigShift JSONL files.

    python examples/prepare_datasets.py mmlu     data/mmlu.jsonl
    python examples/prepare_datasets.py medmcqa  data/medmcqa.jsonl

Needs ``pip install datasets``. The selection follows the paper:

* MMLU: test split, all 57 subjects (14,042 questions); one-shot demos from
  the dev split of the same subject.
* MedMCQA: validation questions with a single correct option (2,816); demos
  are the first two training questions of the same subject.

Example ids match those in examples/paper/*_logprobs.npz.

Use these files as a template for your own data: any JSONL with the same
fields works with the rest of the library.
"""

import argparse
from collections import defaultdict

from cp4configshift import Example, save_examples

N_DEMOS = 2   # the templates use demo_index 0 or 1


def mmlu() -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all")
    demos = defaultdict(list)
    for ex in ds["dev"]:
        demos[ex["subject"]].append({"question": ex["question"], "choices": ex["choices"],
                                     "answer": ex["answer"]})
    return [
        Example(f"mmlu-{i}", ex["question"], ex["answer"], ex["choices"],
                {"subject": ex["subject"].replace("_", " "), "demos": demos[ex["subject"]][:N_DEMOS]})
        for i, ex in enumerate(ds["test"])
    ]


def medmcqa() -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa")
    single = lambda ex: ex["choice_type"] == "single" and 0 <= ex["cop"] <= 3
    options = lambda ex: [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
    questions = [ex for ex in ds["validation"] if single(ex)]

    needed = {ex["subject_name"] for ex in questions}
    demos = defaultdict(list)
    for ex in ds["train"]:
        s = ex["subject_name"]
        if single(ex) and s in needed and len(demos[s]) < N_DEMOS:
            demos[s].append({"question": ex["question"], "choices": options(ex), "answer": ex["cop"]})
            if all(len(demos[k]) >= N_DEMOS for k in needed):
                break
    return [
        Example(ex["id"], ex["question"], ex["cop"], options(ex),
                {"subject": ex["subject_name"].replace("_", " "), "demos": demos[ex["subject_name"]]})
        for ex in questions
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["mmlu", "medmcqa"])
    ap.add_argument("out")
    args = ap.parse_args()
    examples = {"mmlu": mmlu, "medmcqa": medmcqa}[args.dataset]()
    save_examples(examples, args.out)
    print(f"wrote {len(examples)} examples to {args.out}")
