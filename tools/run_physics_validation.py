"""Prepare one local, G-code-driven OpenFOAM-14 case.

Example:
  python tools/run_physics_validation.py --gcode output.gcode --stl input.stl \
      --material materials/pla.example.yaml --window layer:0 --output-dir physics
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.external.physics_cfd import execute_openfoam14_case, validate_physics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcode", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--material", type=Path,
                        default=ROOT / "materials" / "pla.example.yaml")
    parser.add_argument("--window", default="auto")
    parser.add_argument("--max-path-mm", type=float, default=2.0)
    parser.add_argument("--machine-name", default="prusa_mini")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="run blockMesh, foamRun, foamToVTK with 120s timeouts")
    args = parser.parse_args(argv)
    result = validate_physics(args.gcode, args.stl, args.output_dir,
                              args.material, args.window, args.max_path_mm,
                              args.machine_name)
    if args.execute and result["status"] == "CASE_READY":
        result = execute_openfoam14_case(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in ("CASE_READY", "CFD_COMPLETED",
                                      "CFD_COMPLETED_UNCALIBRATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
