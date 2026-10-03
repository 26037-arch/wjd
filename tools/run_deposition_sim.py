"""Run a bounded REP5X-aware voxel deposition reconstruction.

Example:
  python tools/run_deposition_sim.py sample.gcode --backend volco-oriented \
      --output-dir results/deposition/sample
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external.validators import (  # noqa: E402
    validate_hybrid_deposition,
    validate_volco_oriented,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gcode", type=Path)
    parser.add_argument("--backend", choices=("volco-oriented", "hybrid"),
                        default="volco-oriented")
    parser.add_argument("--volco-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.025,
                        help="mm; use 0.0125 for a convergence check")
    parser.add_argument("--step-size", type=float, help="mm; default voxel/4")
    parser.add_argument("--filament-diameter", type=float, default=1.75)
    parser.add_argument("--nozzle-diameter", type=float, default=0.4)
    parser.add_argument("--sphere-offset", type=float, default=0.2)
    parser.add_argument("--max-voxels", type=int, default=2_000_000)
    args = parser.parse_args(argv)
    kwargs = {"voxel_size": args.voxel_size, "step_size": args.step_size,
              "filament_diameter": args.filament_diameter,
              "nozzle_diameter": args.nozzle_diameter,
              "sphere_offset": args.sphere_offset,
              "volco_dir": args.volco_dir, "max_voxels": args.max_voxels}
    run = (validate_volco_oriented if args.backend == "volco-oriented"
           else validate_hybrid_deposition)
    result = run(args.gcode, args.output_dir, **kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
