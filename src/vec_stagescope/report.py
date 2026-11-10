"""Plain-text and JSON rendering of a Result."""
from __future__ import annotations

from dataclasses import asdict

from .proximity import Result

LABEL_TEXT = {
    "copy_like": "COPY-LIKE: after depth matching the prediction is as close to the input stage as "
                 "two samples of that stage are to each other.",
    "very_close": "VERY CLOSE to the input stage.",
    "info": "Not close to the input stage.",
    "no_separation": "No label: on this board the stages are not farther apart than a few noise "
                     "floors at matched depth.",
}


def to_dict(res: Result, board: str, prediction: str, base_assumed: bool, notes=()) -> dict:
    doc = asdict(res)
    doc["ed_by_stage"] = {k: {"min": v[0], "mean": v[1], "max": v[2]}
                          for k, v in res.ed_by_stage.items()}
    doc.update(board=board, prediction=prediction, base_assumed=base_assumed,
               warnings=list(notes) + list(res.warnings))
    return doc


def _f(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def to_text(doc: dict) -> str:
    lines = [f"vec-stagescope  board={doc['board']}  input stage={doc['base']}"
             + ("  (ASSUMED: not published for this board)" if doc["base_assumed"] else ""),
             f"prediction: {doc['prediction']}",
             f"samples: n={doc['n']} per side, {doc['repeats']} repeats "
             "(ranges cover sampling and dropout randomness only, not embryo-to-embryo variation)",
             "", "Median detected genes per cell:"]
    lines += [f"  {k:<12} {v:g}" for k, v in doc["depth"].items()]
    if doc["gate"]:
        lines += ["", f"GATE: {doc['gate']} - no placement."]
    else:
        lines += ["", "Depth-matched energy distance to each training stage:",
                  f"  {'stage':<10}{'ED mean':>10}{'range':>22}{'r':>9}"]
        for k, v in doc["ed_by_stage"].items():
            rng = f"[{_f(v['min'])}, {_f(v['max'])}]"
            mark = "  <- input stage" if k == doc["base"] else ""
            lines.append(f"  {k:<10}{_f(v['mean']):>10}{rng:>22}{_f(doc['r_by_stage'][k]):>9}{mark}")
        lines += ["",
                  f"noise floor (two disjoint samples of {doc['base']}): {_f(doc['ed_floor'])}",
                  f"r denominator (ED of {doc['base']} to nearest other stage): {_f(doc['denominator'])}",
                  f"r_floor = {_f(doc['r_floor'])}",
                  "", f"Result: {LABEL_TEXT[doc['label']]}"]
    if doc["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in doc["warnings"]]
    lines += ["", "This measures similarity to published stage files, not developmental time, and is "
              "not a score."]
    return "\n".join(lines)
