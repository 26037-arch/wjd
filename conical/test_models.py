"""Controlled synthetic solids for repeatable validation experiments."""

import numpy as np
import trimesh

from .meshio import center_on_axis


def sphere(radius=8.0):
    return center_on_axis(trimesh.creation.icosphere(subdivisions=2, radius=radius))


def cylinder(radius=7.0, height=12.0):
    return center_on_axis(trimesh.creation.cylinder(radius=radius, height=height, sections=48))


def box(width=12.0, depth=12.0, height=12.0):
    return center_on_axis(trimesh.creation.box(extents=(width, depth, height)))


def flaring_cone(bottom_radius=5.0, top_radius=9.0, height=15.0):
    profile = np.array([[0, 0], [bottom_radius, 0],
                        [top_radius, height], [0, height]])
    mesh = trimesh.creation.revolve(profile, sections=48)
    mesh.merge_vertices()
    mesh.fix_normals()
    return center_on_axis(mesh)


def waisted_model(r_neck=2.0):
    # Keep the existing research geometry as the single source of truth.
    from compare_waist import waisted_model as existing
    return existing(r_neck)


def widen(mesh, factor):
    from analyze_blend_ratio import widen as existing
    return existing(mesh, factor)


def lobe(mesh, amplitude):
    from analyze_blend_ratio import lobe as existing
    return existing(mesh, amplitude)


def generate_random_model(global_seed, model_id, attempt=0):
    """Same (seed, index, retry) produces the same mesh and metadata."""
    seq = np.random.SeedSequence([int(global_seed), int(model_id), int(attempt)])
    model_seed = int(seq.generate_state(1, dtype=np.uint64)[0])
    rng = np.random.default_rng(seq)
    family = ("sphere", "cylinder", "box", "flaring_cone", "waisted_lobe")[
        int(rng.integers(0, 5))]
    if family == "sphere":
        params = {"radius": float(rng.uniform(4.0, 9.0))}
        mesh = sphere(**params)
    elif family == "cylinder":
        params = {"radius": float(rng.uniform(4.0, 9.0)),
                  "height": float(rng.uniform(8.0, 18.0))}
        mesh = cylinder(**params)
    elif family == "box":
        params = {"width": float(rng.uniform(7.0, 17.0)),
                  "depth": float(rng.uniform(7.0, 17.0)),
                  "height": float(rng.uniform(8.0, 18.0))}
        mesh = box(**params)
    elif family == "flaring_cone":
        params = {"bottom_radius": float(rng.uniform(3.5, 7.0)),
                  "top_radius": float(rng.uniform(7.0, 11.0)),
                  "height": float(rng.uniform(10.0, 20.0))}
        mesh = flaring_cone(**params)
    else:
        params = {"r_neck": float(rng.uniform(1.5, 5.5)),
                  "xy_scale": float(rng.uniform(0.7, 1.3)),
                  "lobe_amplitude": float(rng.uniform(0.0, 0.18))}
        mesh = lobe(widen(waisted_model(params["r_neck"]), params["xy_scale"]),
                    params["lobe_amplitude"])
    meta = {"model_id": model_id, "global_seed": global_seed,
            "model_seed": model_seed, "attempt": attempt,
            "generator": family, "parameters": params,
            "vertices": len(mesh.vertices), "faces": len(mesh.faces)}
    return mesh, meta
