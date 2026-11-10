"""Which published training stages each leaderboard is compared against, and its baseline (input)
stage. ``base_assumed`` marks boards whose input stage is not published (test boards)."""
from __future__ import annotations

from dataclasses import dataclass


class BoardError(ValueError):
    pass


@dataclass(frozen=True)
class Board:
    name: str
    stages: tuple[tuple[str, str], ...]   # (label, file name in --data-dir)
    base: str
    base_assumed: bool


_EMBRYO = (("E6.75", "E6.75.h5ad"), ("E7.25", "E7.25.h5ad"), ("E8.0", "E8.0.h5ad"))
_HEART = (("E8.25", "E8.25_late.h5ad"), ("E8.75", "E8.75.h5ad"), ("E9.5", "E9.5.h5ad"))
_T1 = (("E8.5", "E8.5_RNA.h5ad"), ("E9.5", "E9.5_RNA.h5ad"))

BOARDS: dict[str, Board] = {
    b.name: b
    for b in (
        Board("T2:embryo:val_interp", _EMBRYO, "E7.25", False),
        Board("T2:embryo:test_interp", _EMBRYO, "E7.25", True),
        Board("T2:heart:val_interp", _HEART, "E8.25", False),
        Board("T2:heart:val_extrap", _HEART, "E9.5", False),
        Board("T2:heart:test_extrap", _HEART, "E9.5", True),
        Board("T1:val", _T1, "E9.5", False),
        Board("T1:test", _T1, "E9.5", True),
    )
}


def get_board(name: str) -> Board:
    if name.upper().startswith("T3"):
        raise BoardError("Task 3 (perturbation) boards are out of scope for vec-stagescope.")
    if name not in BOARDS:
        raise BoardError(f"Unknown board {name!r}. Known: {', '.join(sorted(BOARDS))}")
    return BOARDS[name]
