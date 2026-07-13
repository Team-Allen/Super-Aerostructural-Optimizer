"""
Constraint definitions for the MDO problem.

Aero constraints: CL target, CM bounds, maximum Mach divergence
Structural constraints: failure index, tip deflection, minimum gauge
Geometric constraints: thickness bounds, LE radius, trailing edge angle
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConstraintResult:
    """Evaluation of all constraints at a design point."""
    cl_error: float           # |CL - CL_target|
    failure_index: float      # Max Tsai-Wu failure index
    tip_deflection: float     # Absolute tip deflection [m]
    max_thickness: float      # Maximum t/c
    min_thickness: float      # Minimum t/c (local)
    cm: float                 # Pitching moment coefficient
    feasible: bool            # All constraints satisfied?

    @property
    def constraint_violations(self) -> dict[str, float]:
        """Return dict of violated constraints and their violation magnitude."""
        violations = {}
        if self.cl_error > 0.01:
            violations['cl_error'] = self.cl_error
        if self.failure_index > 1.0:
            violations['failure_index'] = self.failure_index - 1.0
        return violations


class ConstraintManager:
    """Manages all optimization constraints."""

    def __init__(
        self,
        cl_target: float = 0.5,
        cl_tolerance: float = 0.01,
        max_failure_index: float = 0.8,
        max_tip_deflection: float = 2.0,
        min_thickness: float = 0.08,
        max_thickness: float = 0.20,
        cm_lower: float = -0.15,
        cm_upper: float = 0.05,
    ):
        self.cl_target = cl_target
        self.cl_tolerance = cl_tolerance
        self.max_failure_index = max_failure_index
        self.max_tip_deflection = max_tip_deflection
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        self.cm_lower = cm_lower
        self.cm_upper = cm_upper

    def evaluate(
        self,
        cl: float,
        cd: float,
        cm: float,
        failure_index: float,
        tip_deflection: float,
        max_thickness: float,
    ) -> ConstraintResult:
        """Evaluate all constraints."""
        cl_error = abs(cl - self.cl_target)

        feasible = (
            cl_error <= self.cl_tolerance and
            failure_index <= self.max_failure_index and
            abs(tip_deflection) <= self.max_tip_deflection and
            max_thickness >= self.min_thickness and
            max_thickness <= self.max_thickness and
            cm >= self.cm_lower and
            cm <= self.cm_upper
        )

        return ConstraintResult(
            cl_error=cl_error,
            failure_index=failure_index,
            tip_deflection=abs(tip_deflection),
            max_thickness=max_thickness,
            min_thickness=max_thickness,  # simplified
            cm=cm,
            feasible=feasible,
        )

    def penalty(self, result: ConstraintResult, weight: float = 100.0) -> float:
        """Compute exterior penalty for constraint violations."""
        p = 0.0

        if result.cl_error > self.cl_tolerance:
            p += weight * (result.cl_error - self.cl_tolerance)**2

        if result.failure_index > self.max_failure_index:
            p += weight * (result.failure_index - self.max_failure_index)**2

        if result.tip_deflection > self.max_tip_deflection:
            p += weight * (result.tip_deflection - self.max_tip_deflection)**2

        if result.max_thickness < self.min_thickness:
            p += weight * (self.min_thickness - result.max_thickness)**2

        if result.max_thickness > self.max_thickness:
            p += weight * (result.max_thickness - self.max_thickness)**2

        return p
