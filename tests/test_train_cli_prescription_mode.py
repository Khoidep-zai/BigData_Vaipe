from __future__ import annotations

import json
import sys
from pathlib import Path

import train_cli


def test_train_cli_prescription_match_writes_json(monkeypatch, tmp_path: Path):
    output_json = tmp_path / "result.json"

    monkeypatch.setattr(
        train_cli,
        "discover_or_prepare_data_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(
        train_cli,
        "prepare_metadata_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    monkeypatch.setattr(train_cli, "match_pills_to_prescription", lambda **kwargs: object())
    monkeypatch.setattr(
        train_cli,
        "result_to_dict",
        lambda _result: {
            "prescription_image": "pres.png",
            "prescription_json": "pres.json",
            "classes_in_prescription": [1, 2],
            "pill_results": [],
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cli.py",
            "--mode",
            "prescription_match",
            "--prescription-image",
            "pres.png",
            "--pill-images",
            "pill_a.jpg",
            "pill_b.jpg",
            "--output-json",
            str(output_json),
        ],
    )

    train_cli.main()

    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["prescription_json"] == "pres.json"
    assert payload["classes_in_prescription"] == [1, 2]


def test_train_cli_prescription_match_writes_csv_and_pretty(monkeypatch, tmp_path: Path, capsys):
    output_csv = tmp_path / "result.csv"

    monkeypatch.setattr(train_cli, "match_pills_to_prescription", lambda **kwargs: object())
    monkeypatch.setattr(
        train_cli,
        "result_to_dict",
        lambda _result: {
            "prescription_image": "pres.png",
            "prescription_json": "pres.json",
            "classes_in_prescription": [1],
            "pill_results": [],
        },
    )

    def _write_result_csv(_result, output_csv_path):
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pill_image,group\n", encoding="utf-8")
        return path

    monkeypatch.setattr(train_cli, "write_result_csv", _write_result_csv)
    monkeypatch.setattr(train_cli, "format_pretty_summary", lambda _result: "pretty summary")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cli.py",
            "--mode",
            "prescription_match",
            "--prescription-image",
            "pres.png",
            "--pill-images",
            "pill_a.jpg",
            "--output-csv",
            str(output_csv),
            "--pretty",
        ],
    )

    train_cli.main()

    assert output_csv.exists()
    captured = capsys.readouterr()
    assert "Da ghi CSV" in captured.out
    assert "pretty summary" in captured.out


