"""Plain-text and JSON rendering of a Result."""
from __future__ import annotations

from dataclasses import asdict

from .proximity import Result

def to_dict(res: Result, board: str, prediction: str, base_assumed: bool, notes=()) -> dict:
    doc = asdict(res)
    for key in ("ed_by_stage", "raw_ed_by_stage"):
        doc[key] = {k: {"min": v[0], "mean": v[1], "max": v[2]} for k, v in doc[key].items()}
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
        lines += ["", "Energy distance (ED) and r to each training stage, raw and depth-matched:",
                  f"  {'stage':<10}{'raw ED':>9}{'raw r':>8}{'matched ED':>12}{'range':>20}"
                  f"{'matched r':>11}"]
        for k, v in doc["ed_by_stage"].items():
            raw = doc["raw_ed_by_stage"][k]["mean"]
            rng = f"[{_f(v['min'])}, {_f(v['max'])}]"
            mark = "  <- input stage" if k == doc["base"] else ""
            lines.append(f"  {k:<10}{_f(raw):>9}{_f(doc['raw_r_by_stage'][k]):>8}"
                         f"{_f(v['mean']):>12}{rng:>20}{_f(doc['r_by_stage'][k]):>11}{mark}")
        lines += ["",
                  f"noise floor (two disjoint samples of {doc['base']}, matched): {_f(doc['ed_floor'])}",
                  f"r denominator (matched ED of {doc['base']} to nearest other stage): "
                  f"{_f(doc['denominator'])}",
                  "", "No copy verdict is issued (see README: 'Copy label: not shipped'). A small r "
                  "means close to the input stage; read raw and matched together."]
    if doc["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in doc["warnings"]]
    lines += ["", "This measures similarity to published stage files, not developmental time, and is "
              "not a score."]
    return "\n".join(lines)
