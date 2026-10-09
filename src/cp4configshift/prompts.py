"""Prompt templates.

Templates live in a YAML file::

    choice_styles:                          # multiple-choice only
      paren_upper: {format: "({label}) {text}", labels: [A, B, C, D]}
    templates:
      Baseline:
        choice_style: paren_upper           # multiple-choice only
        demo_index: 0                       # which of example.meta["demos"] to show
        prompt: |-
          This is a question from {subject}.
          ...
    demos:                                  # optional fixed few-shot demos (open-ended);
      - {question: "...", answer: "..."}    # a template may also define its own

Placeholders available in ``prompt``:

* ``{question}``, and every string field of ``example.meta`` (e.g. ``{subject}``);
* multiple-choice: ``{choices}`` (options formatted with the choice style), and
  ``{demo_question}``, ``{demo_choices}``, ``{demo_answer}`` taken from
  ``example.meta["demos"][demo_index]`` (each demo: question, choices, answer index);
* open-ended: ``{demos}`` (the fixed demos, one "<q_prefix>question\\n<a_prefix>answer"
  block each), ``{q_prefix}`` and ``{a_prefix}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cp4configshift.data import Example


@dataclass
class Template:
    prompt: str
    labels: list[str] | None = None          # option labels (multiple-choice)
    choice_format: str = "({label}) {text}"
    demo_index: int = 0
    demos: list[dict] = field(default_factory=list)
    q_prefix: str = "Q: "
    a_prefix: str = "A: "

    def format_choices(self, choices: list[str]) -> str:
        if self.labels is None or len(self.labels) != len(choices):
            raise ValueError(f"template has labels {self.labels} but the question has {len(choices)} options")
        return "\n".join(self.choice_format.format(label=lab, text=c) for lab, c in zip(self.labels, choices))

    def render(self, ex: Example) -> str:
        fields = {k: v for k, v in ex.meta.items() if isinstance(v, str)}
        fields["question"] = ex.question
        if ex.is_multiple_choice:
            fields["choices"] = self.format_choices(ex.choices)
            demos = ex.meta.get("demos", [])
            if self.demo_index < len(demos):
                d = demos[self.demo_index]
                fields.update(demo_question=d["question"], demo_choices=self.format_choices(d["choices"]),
                              demo_answer=self.labels[d["answer"]])
        else:
            fields.update(
                demos="\n\n".join(f"{self.q_prefix}{d['question']}\n{self.a_prefix}{d['answer']}"
                                  for d in self.demos),
                q_prefix=self.q_prefix, a_prefix=self.a_prefix,
            )
        return self.prompt.format(**fields)


def load_templates(path: str | Path) -> dict[str, Template]:
    import yaml

    cfg = yaml.safe_load(open(path))
    styles = cfg.get("choice_styles", {})
    out = {}
    for name, t in cfg["templates"].items():
        style = styles[t["choice_style"]] if "choice_style" in t else {}
        out[name] = Template(
            prompt=t["prompt"],
            labels=style.get("labels"),
            choice_format=style.get("format", "({label}) {text}"),
            demo_index=t.get("demo_index", 0),
            demos=t.get("demos") or cfg.get("demos") or [],
            q_prefix=t.get("q_prefix", "Q: "),
            a_prefix=t.get("a_prefix", "A: "),
        )
    return out
