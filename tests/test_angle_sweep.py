from tools.run_random_validation import summarize_sweep


def test_sweep_keeps_separate_metric_optima():
    angles = [
        {"angle_deg": 0, "status": "OK", "geometry": {"volco": {
            "sampled_bidirectional_chamfer_mm": 2}}, "support": {"internal": {
            "chained": {"unsupported_pct": 5}}}},
        {"angle_deg": 4, "status": "OK", "geometry": {"volco": {
            "sampled_bidirectional_chamfer_mm": 1}}, "support": {"internal": {
            "chained": {"unsupported_pct": 7}}}},
        {"angle_deg": 8, "status": "COLLISION", "geometry": {"volco": {
            "sampled_bidirectional_chamfer_mm": 0}}, "support": {"internal": {
            "chained": {"unsupported_pct": 1}}}},
    ]
    result = summarize_sweep(4, angles)
    assert result["theta_best_geometry"] == 4
    assert result["theta_best_support_internal"] == 0
    assert result["internally_eligible_angles"] == [0, 4]
    assert result["pareto_geometry_vs_internal_support_angles"] == [0, 4]
