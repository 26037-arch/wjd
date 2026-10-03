"""Boundary for a future validated local, oriented CFD deposition kernel.

The existing OpenFOAM experiment is Cartesian and its bead surface has not
converged. It cannot supply spatial corrections to the REP5X voxel model.
"""

from __future__ import annotations

import math
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass

from conical.deposition_events import MotionEvent


@dataclass(frozen=True)
class LocalDepositionInput:
    tip_xyz_mm: tuple[float, float, float]
    tool_direction: tuple[float, float, float]
    b_deg: float
    c_deg: float
    volumetric_flow_mm3_s: float
    nozzle_velocity_mm_s: tuple[float, float, float]
    nozzle_diameter_mm: float
    filament_diameter_mm: float
    nozzle_gap_mm: float | None
    material_model: str | None
    temperature_c: float | None

    @classmethod
    def from_event(cls, event: MotionEvent, fraction: float, *,
                   nozzle_diameter_mm: float, filament_diameter_mm: float,
                   nozzle_gap_mm: float | None = None,
                   material_model: str | None = None,
                   temperature_c: float | None = None):
        pose = event.pose(fraction)
        duration = event.estimated_duration
        velocity = tuple((end - start) / duration if duration > 0 else 0.0
                         for start, end in zip(event.start_position,
                                               event.end_position, strict=True))
        rate = max(event.delta_e, 0.0) / duration if duration > 0 else 0.0
        flow = math.pi * (filament_diameter_mm / 2)**2 * rate
        return cls(tuple(pose["position"]), tuple(pose["direction"]),
                   pose["b"], pose["c"], flow, velocity,
                   nozzle_diameter_mm, filament_diameter_mm, nozzle_gap_mm,
                   material_model, temperature_c)


class CFDBackend(ABC):
    @abstractmethod
    def available(self) -> bool:
        """Whether a validated oriented local deposition solver is ready."""

    @abstractmethod
    def simulate_local_deposition(self, request: LocalDepositionInput) -> dict:
        """Return a spatial kernel or an explicit unavailable status."""


class OpenFOAMLocalBackend(CFDBackend):
    """Availability gate for the unvalidated Cartesian OpenFOAM experiment."""

    def available(self) -> bool:
        return False

    def status(self) -> dict:
        return {"status": "NOT_AVAILABLE", "cfd_applied": False,
                "solver_found_on_path": shutil.which("foamRun") is not None,
                "reason": "Oriented local OpenFOAM kernel has not been validated"}

    def simulate_local_deposition(self, request: LocalDepositionInput) -> dict:
        return self.status()
