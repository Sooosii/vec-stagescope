import json

import numpy as np
import pytest

from conftest import write_h5ad
from synth import GENES, uniform_dropout
from vec_stagescope import boards
from vec_stagescope.cli import main


@pytest.fixture
def data_dir(tmp_path, stages, monkeypatch):
    board = boards.Board(name="TEST:synth", stages=(("S1", "S1.h5ad"), ("S2", "S2.h5ad"),
                                                     ("S3", "S3.h5ad")), base="S3", base_assumed=False)
    monkeypatch.setitem(boards.BOARDS, "TEST:synth", board)
    for name, x in stages.items():
        write_h5ad(tmp_path / f"{name}.h5ad", x)
    return tmp_path


def test_cli_reports_copy_and_writes_json(data_dir, stages, tmp_path, capsys):
    pred = write_h5ad(tmp_path / "pred.h5ad", uniform_dropout(stages["S3"], 0.5, 7))
    out = tmp_path / "out.json"
    code = main(["check", str(pred), "--board", "TEST:synth", "--data-dir", str(data_dir),
                 "--json", str(out), "--n", "400", "--repeats", "2"])
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["label"] == "copy_like"
    assert "depth" in doc and "r_by_stage" in doc
    assert "copy" in capsys.readouterr().out.lower()


def test_cli_strict_returns_1_on_warning(data_dir, stages, tmp_path):
    pred = write_h5ad(tmp_path / "pred.h5ad", stages["S3"][:800])
    code = main(["check", str(pred), "--board", "TEST:synth", "--data-dir", str(data_dir),
                 "--n", "300", "--repeats", "1", "--strict"])
    assert code == 1


def test_gene_mismatch_warns_and_intersects(data_dir, stages, tmp_path, capsys):
    genes = list(GENES[:-3]) + ["zz1", "zz2", "zz3"]
    pred = write_h5ad(tmp_path / "pred.h5ad", stages["S3"][:800], genes=genes)
    code = main(["check", str(pred), "--board", "TEST:synth", "--data-dir", str(data_dir),
                 "--n", "300", "--repeats", "1"])
    assert code == 0
    assert "3 prediction genes not in" in capsys.readouterr().out


def test_t3_board_is_rejected(tmp_path, capsys):
    pred = write_h5ad(tmp_path / "pred.h5ad", np.ones((5, len(GENES)), np.float32))
    assert main(["check", str(pred), "--board", "T3:val", "--data-dir", str(tmp_path)]) == 2
    assert "Task 3" in capsys.readouterr().err


def test_missing_stage_file_is_input_error(tmp_path, stages):
    pred = write_h5ad(tmp_path / "pred.h5ad", stages["S3"][:100])
    assert main(["check", str(pred), "--board", "T2:heart:val_interp",
                 "--data-dir", str(tmp_path / "nope")]) == 2
