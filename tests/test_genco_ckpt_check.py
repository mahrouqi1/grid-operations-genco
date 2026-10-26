"""ckpt_check fills missing fields on pre-existing rows (audit genco-adapt R2-F4) and handles lost checkpoints."""
import json

import pytest

torch = pytest.importorskip("torch")
fs = pytest.importorskip("genco_challenge.fewshot")


def _setup(tmp_path, monkeypatch, existing):
    pre = {"w": torch.zeros(2)}
    torch.save(pre, tmp_path / "pre.pt")
    runs = tmp_path / "runs"
    runs.mkdir()
    for name, same in (("a", True), ("b", False), ("lost", True)):
        d = runs / name
        (d / "metrics").mkdir(parents=True)
        (d / "metrics" / "Validation layer_11_residual").write_text("0 0.5\n1 0.2\n2 0.3\n")
        model = tmp_path / f"{name}.pt"
        if name != "lost":
            torch.save(pre if same else {"w": torch.ones(2)}, model)
        (runs / f"{name}.json").write_text(json.dumps({"key": name, "mode": "finetune", "model": str(model)}))
    out = tmp_path / "out"
    out.mkdir()
    if existing:
        (out / "ckpt_check.json").write_text(json.dumps(existing))
    monkeypatch.setattr(fs, "FS", tmp_path)
    monkeypatch.setattr(fs, "PRETRAINED", tmp_path / "pre.pt")
    return out


def test_fills_missing_fields_on_old_rows(tmp_path, monkeypatch):
    old = {"a": {"selected_is_pretrained": True, "val_l11_epoch1": 0.5, "val_l11_min": 0.2, "val_l11_argmin": 1},
           "lost": {"selected_is_pretrained": True}}
    out = _setup(tmp_path, monkeypatch, old)
    res = fs.ckpt_check(out_dir=out, only_existing=True)
    assert res["a"]["pretrained_val_l11_bound"] == "<= 0.2"
    assert res["a"]["val_l11_min"] == 0.2
    assert "b" not in res                       # only_existing adds no key
    assert res["lost"] == {"selected_is_pretrained": True}   # lost checkpoint: row left as it was
    # a row that already has the bound is not recomputed
    res["a"]["pretrained_val_l11_bound"] = "kept"
    (out / "ckpt_check.json").write_text(json.dumps(res))
    assert fs.ckpt_check(out_dir=out, only_existing=True)["a"]["pretrained_val_l11_bound"] == "kept"


def test_new_rows_get_bound(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch, {})
    res = fs.ckpt_check(out_dir=out)
    assert res["b"]["pretrained_val_l11_bound"] == "> 0.2"
    assert "lost" not in res
