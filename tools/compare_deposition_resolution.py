"""Compare two oriented voxel runs on their common fine grid.

This measures numerical self consistency for the same G-code and sample step;
it does not compare to a physical printed object.
"""

import argparse
import json
import math
from pathlib import Path


def _occupied(folder, manifest):
    path = Path(folder) / manifest["files"]["frames"]
    occupied = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        occupied.update(tuple(index) for index in json.loads(line)["added"])
    return occupied


def compare(coarse_dir, fine_dir):
    coarse_dir, fine_dir = Path(coarse_dir), Path(fine_dir)
    a = json.loads((coarse_dir / "manifest.json").read_text(encoding="utf-8"))
    b = json.loads((fine_dir / "manifest.json").read_text(encoding="utf-8"))
    if a["backend"] != "volco_oriented_extension" or a["backend"] != b["backend"]:
        raise ValueError("Both runs must use the oriented voxel backend")
    if a["source_gcode"] != b["source_gcode"]:
        raise ValueError("Runs must use the same source G-code")
    if not math.isclose(a["step_size_mm"], b["step_size_mm"], abs_tol=1e-12):
        raise ValueError("Runs must use the same path sample step")
    if a["coordinate_convention"] != b["coordinate_convention"]:
        raise ValueError("Runs use different coordinate conventions")
    ratio = a["voxel_size_mm"] / b["voxel_size_mm"]
    scale = round(ratio)
    if scale < 1 or not math.isclose(ratio, scale, abs_tol=1e-9):
        raise ValueError("Fine voxel size must divide coarse voxel size")
    coarse = _occupied(coarse_dir, a)
    fine = _occupied(fine_dir, b)
    expanded = {(i*scale+di, j*scale+dj, k*scale+dk)
                for i, j, k in coarse
                for di in range(scale) for dj in range(scale)
                for dk in range(scale)}
    intersection = len(expanded & fine)
    excess = len(expanded - fine)
    missing = len(fine - expanded)
    volume = b["voxel_size_mm"]**3
    return {"coarse_voxel_mm": a["voxel_size_mm"],
            "fine_voxel_mm": b["voxel_size_mm"],
            "step_size_mm": b["step_size_mm"],
            "iou": intersection / (intersection + excess + missing)
            if intersection + excess + missing else 1.0,
            "coarse_excess_mm3": excess * volume,
            "coarse_missing_mm3": missing * volume,
            "reference": "finer numerical run; not physical ground truth"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse_dir", type=Path)
    parser.add_argument("fine_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.coarse_dir, args.fine_dir)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
