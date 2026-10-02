import json

from tools.run_random_validation import main


def test_small_pipeline_records_case_and_summary(tmp_path):
    assert main(["--count", "1", "--seed", "1", "--layer-height", "2",
                 "--max-edge", "4", "--output-dir", str(tmp_path)]) == 0
    case = json.loads((tmp_path / "cases.jsonl").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert case["prediction"]["angle_deg"] == 0
    assert case["selected_angle_result"]["gcode_validation"]["internal"]["valid"]
    assert case["external_validation_complete"] is False
    assert sum(summary["statuses"].values()) == 1
