"""Submission JSON shown in public logs never carries the upload link."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "redact_submission", ROOT / "scripts" / "redact_submission.py",
)
redact_submission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(redact_submission)

SAS = ("https://ingestionpackagesprod1.blob.core.windows.net/ingestion/"
       "abc?sv=2024-05-04&se=2026-11-20T20%3A15Z&sp=rwl&sig=SECRET%3D")


def test_the_upload_link_and_any_other_blob_link_are_removed(tmp_path):
    submission = {
        "Id": "1",
        "FileUploadUrl": SAS,
        "Listings": {"en-us": {"BaseListing": {
            "Description": f"see {SAS} here",
            "Images": [{"FileName": "listing/a.png"}],
        }}},
    }
    source = tmp_path / "submission.json"
    source.write_text(json.dumps(submission), encoding="utf-8")
    out = tmp_path / "redacted.json"
    assert redact_submission.main([str(source), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "SECRET" not in text
    assert "blob.core.windows.net" not in text
    data = json.loads(text)
    assert data["FileUploadUrl"] == redact_submission.REDACTED
    assert data["Listings"]["en-us"]["BaseListing"]["Images"] == [
        {"FileName": "listing/a.png"},
    ]


def test_msstore_output_with_log_lines_is_parsed(tmp_path, capsys):
    source = tmp_path / "out.txt"
    source.write_text(
        "08:02:53 info: retrieving submission\n"
        + json.dumps({"FileUploadUrl": SAS, "Status": "PendingCommit"}),
        encoding="utf-8",
    )
    assert redact_submission.main([str(source)]) == 0
    printed = capsys.readouterr().out
    assert "SECRET" not in printed
    assert '"Status": "PendingCommit"' in printed
