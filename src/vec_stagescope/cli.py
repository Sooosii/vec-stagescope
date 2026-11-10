"""Command line: vec-stagescope check pred.h5ad --board B --data-dir DIR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, boards
from .io import InputError, align, load, subset_genes
from .proximity import diagnose
from .report import to_dict, to_text

EXIT_OK, EXIT_WARN, EXIT_INPUT = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vec-stagescope", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="diagnose one prediction file")
    c.add_argument("prediction", type=Path)
    c.add_argument("--board", required=True)
    c.add_argument("--data-dir", required=True, type=Path,
                   help="folder with the training .h5ad files you downloaded")
    c.add_argument("--json", type=Path, help="also write the report as JSON")
    c.add_argument("--strict", action="store_true", help="exit 1 if there is any warning")
    c.add_argument("--n", type=int, default=1500)
    c.add_argument("--repeats", type=int, default=5)
    c.add_argument("--seed", type=int, default=0)
    return p


def _check(args) -> int:
    board = boards.get_board(args.board)
    pred, pred_genes = load(args.prediction)
    loaded = {label: load(args.data_dir / fname) for label, fname in board.stages}
    keep, notes = align(pred_genes, {k: g for k, (_, g) in loaded.items()})
    stages = {k: subset_genes(x, g, keep) for k, (x, g) in loaded.items()}
    res = diagnose(subset_genes(pred, pred_genes, keep), stages, base=board.base, n=args.n,
                   repeats=args.repeats, seed=args.seed)
    if board.base_assumed:
        notes.append(f"Input stage {board.base} is an assumption for {board.name}; "
                     "check r against every stage in the table.")
    doc = to_dict(res, board.name, str(args.prediction), board.base_assumed, notes)
    print(to_text(doc))
    if args.json:
        args.json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return EXIT_WARN if args.strict and doc["warnings"] else EXIT_OK


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _check(args)
    except (boards.BoardError, InputError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
