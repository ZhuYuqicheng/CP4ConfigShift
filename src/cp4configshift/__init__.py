"""CP4ConfigShift: conformal prediction for LLMs under configuration shift."""

from cp4configshift.conformal import calibrate, conformal_threshold, evaluate
from cp4configshift.data import Example, load_examples, save_examples
from cp4configshift.prompts import Template, load_templates
from cp4configshift.runs import Run, RunConfig, collect, load_run, score_run
from cp4configshift.scores import (
    contains_alias,
    lac,
    lofreecp,
    logit_margin,
    mc_pools,
    normalize_answer,
    open_pools,
    parse_choice,
    self_consistency,
    sentence_embedder,
    strip_thinking,
)
from cp4configshift.shift import ShiftResult, evaluate_shift, make_splits
from cp4configshift.table import ScoreTable

__version__ = "0.1.0"
