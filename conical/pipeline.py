"""Reusable fixed-angle branch of conical_slice.py (no selector changes)."""

import json

import trimesh

from . import gcode as gc
from .backtransform import backtransform
from .planar_slicer import slice_mesh
from .rep5x import REP5X, add_c_rewinds, to_rep5x
from .transform import transform_cone


def generate_conical_gcode(mesh, angle_deg, direction, output_path,
                           layer_height=0.3, perimeters=2, infill_spacing=2.5,
                           chord_tol=0.05, max_edge=1.5, rewind=True):
    """Use the same refine/transform/slice/backtransform/REP5X stages as CLI."""
    from conical_slice import refine
    fine = refine(mesh, max_edge)
    warped = (trimesh.Trimesh(vertices=transform_cone(fine.vertices, angle_deg,
                                                       direction),
                               faces=fine.faces, process=False)
              if angle_deg > 0 else fine)
    items = slice_mesh(warped, layer_height=layer_height, perimeters=perimeters,
                       infill_spacing=infill_spacing)
    real, back_stats = backtransform(items, angle_deg, direction,
                                     chord_tol=chord_tol)
    physical = real
    rep, rep_stats = to_rep5x(real, angle_deg, direction, REP5X)
    rewind_stats = None
    if rewind:
        rep, rewind_stats = add_c_rewinds(rep, REP5X)
    meta = {"version": 1, "direction": direction,
            "profile": [[0.0, float(angle_deg)]], "layer_height": layer_height,
            "extrusion_width": 0.45, "chord_tol": chord_tol,
            "mode": "rep5x"}
    gc.write([("raw", ";CONICAL_META " + json.dumps(meta, separators=(",", ":")))]
             + rep, output_path)
    return {"physical_items": physical, "rep5x_items": rep,
            "backtransform": back_stats, "rep5x": rep_stats,
            "rewind": rewind_stats}
