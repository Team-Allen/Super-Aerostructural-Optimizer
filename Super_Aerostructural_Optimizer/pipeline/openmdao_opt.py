"""OpenMDAO + pyOptSparse optimization wrapper for the coupled workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import openmdao.api as om
except ImportError as exc:  # pragma: no cover - environment-specific import
    raise ImportError(
        "openmdao is required to run optimization. Use the Linux/WSL mdo-best environment."
    ) from exc

from .workflow import AerostructuralMDOPipeline, DesignVariables


def _scalar(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0])


def _available_pyoptsparse_optimizer(preferred_order: List[str]) -> Optional[str]:
    try:
        from pyoptsparse import OPT
    except Exception:
        return None
    for name in preferred_order:
        try:
            _ = OPT(name)
            return name
        except Exception:
            continue
    return None


class AerostructuralCoupledComp(om.ExplicitComponent):
    """OpenMDAO component that calls the coupled TACS+FUNtoFEM workflow."""

    def initialize(self) -> None:
        self.options.declare("pipeline", types=AerostructuralMDOPipeline, recordable=False)

    def setup(self) -> None:
        self.add_input("twist_root_delta_deg", val=0.0)
        self.add_input("twist_tip_delta_deg", val=0.0)
        self.add_input("thickness_root_scale", val=1.0)
        self.add_input("thickness_tip_scale", val=1.0)

        self.add_output("objective_cd", val=1.0)
        self.add_output("objective_cd_raw", val=1.0)
        self.add_output("ks_failure_minus_1", val=0.0)
        self.add_output("tip_deflection_minus_limit_m", val=0.0)
        self.add_output("cl_minus_target", val=0.0)

        self.add_output("cl", val=0.0)
        self.add_output("cd", val=1.0)
        self.add_output("l_over_d", val=1.0)
        self.add_output("mass_kg", val=0.0)
        self.add_output("alpha_deg", val=0.0)

        # Use finite differencing for robust optimizer interoperability.
        self.declare_partials("*", "*", method="fd")

        self.last_result: Dict[str, object] | None = None

    def compute(self, inputs, outputs) -> None:
        design = DesignVariables(
            twist_root_delta_deg=_scalar(inputs["twist_root_delta_deg"]),
            twist_tip_delta_deg=_scalar(inputs["twist_tip_delta_deg"]),
            thickness_root_scale=_scalar(inputs["thickness_root_scale"]),
            thickness_tip_scale=_scalar(inputs["thickness_tip_scale"]),
        )
        result = self.options["pipeline"].evaluate(design)
        self.last_result = result

        outputs["objective_cd"] = float(result["objective_cd"])
        outputs["objective_cd_raw"] = float(result["objective_cd_raw"])

        constraints = result["constraints"]
        outputs["ks_failure_minus_1"] = float(constraints["ks_failure_minus_1"])
        outputs["tip_deflection_minus_limit_m"] = float(
            constraints["tip_deflection_minus_limit_m"]
        )
        outputs["cl_minus_target"] = float(constraints["cl_minus_target"])

        aero = result["aero"]
        structure = result["structure"]
        outputs["cl"] = float(aero["cl"])
        outputs["cd"] = float(aero["cd"])
        outputs["l_over_d"] = float(aero["l_over_d"])
        outputs["alpha_deg"] = float(aero["alpha_deg"])
        outputs["mass_kg"] = float(structure["mass_kg"])


def run_openmdao_optimization(
    pipeline: AerostructuralMDOPipeline,
    record_path: str | Path | None = None,
) -> Dict[str, object]:
    """Run pyOptSparse/OpenMDAO optimization and return final solution summary."""
    cfg = pipeline.config
    opt_cfg = cfg.optimizer

    prob = om.Problem()
    indep = om.IndepVarComp()
    indep.add_output("twist_root_delta_deg", val=0.0)
    indep.add_output("twist_tip_delta_deg", val=0.0)
    indep.add_output("thickness_root_scale", val=1.0)
    indep.add_output("thickness_tip_scale", val=1.0)
    prob.model.add_subsystem("design", indep, promotes=["*"])

    coupled = AerostructuralCoupledComp(pipeline=pipeline)
    prob.model.add_subsystem("coupled", coupled, promotes=["*"])

    chosen = _available_pyoptsparse_optimizer(opt_cfg.fallback_sequence)
    if chosen is not None:
        prob.driver = om.pyOptSparseDriver(optimizer=chosen)
        if chosen == "IPOPT":
            prob.driver.opt_settings["tol"] = float(opt_cfg.tolerance)
            prob.driver.opt_settings["max_iter"] = int(opt_cfg.max_iterations)
            prob.driver.opt_settings["print_level"] = 5
        elif chosen in ("SLSQP", "PSQP"):
            prob.driver.opt_settings["ACC"] = float(opt_cfg.tolerance)
            prob.driver.opt_settings["MAXIT"] = int(opt_cfg.max_iterations)
    else:
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"
        prob.driver.options["tol"] = float(opt_cfg.tolerance)
        prob.driver.options["maxiter"] = int(opt_cfg.max_iterations)

    if record_path is not None:
        recorder = om.SqliteRecorder(str(record_path))
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options["record_derivatives"] = False
        prob.driver.recording_options["record_constraints"] = True
        prob.driver.recording_options["record_objectives"] = True
        prob.driver.recording_options["record_desvars"] = True

    prob.model.add_design_var(
        "twist_root_delta_deg",
        lower=opt_cfg.twist_root_delta_min_deg,
        upper=opt_cfg.twist_root_delta_max_deg,
    )
    prob.model.add_design_var(
        "twist_tip_delta_deg",
        lower=opt_cfg.twist_tip_delta_min_deg,
        upper=opt_cfg.twist_tip_delta_max_deg,
    )
    prob.model.add_design_var(
        "thickness_root_scale",
        lower=opt_cfg.thickness_root_scale_min,
        upper=opt_cfg.thickness_root_scale_max,
    )
    prob.model.add_design_var(
        "thickness_tip_scale",
        lower=opt_cfg.thickness_tip_scale_min,
        upper=opt_cfg.thickness_tip_scale_max,
    )

    prob.model.add_objective("objective_cd", scaler=1.0)
    prob.model.add_constraint("ks_failure_minus_1", upper=0.0)
    prob.model.add_constraint("tip_deflection_minus_limit_m", upper=0.0)
    prob.model.add_constraint("cl_minus_target", lower=-5.0e-3, upper=5.0e-3)

    prob.setup()
    prob.run_driver()

    solution = {
        "optimizer": chosen if chosen is not None else "SLSQP(SciPy)",
        "design": asdict(
            DesignVariables(
                twist_root_delta_deg=_scalar(prob.get_val("twist_root_delta_deg")),
                twist_tip_delta_deg=_scalar(prob.get_val("twist_tip_delta_deg")),
                thickness_root_scale=_scalar(prob.get_val("thickness_root_scale")),
                thickness_tip_scale=_scalar(prob.get_val("thickness_tip_scale")),
            )
        ),
        "outputs": {
            "objective_cd": _scalar(prob.get_val("objective_cd")),
            "objective_cd_raw": _scalar(prob.get_val("objective_cd_raw")),
            "cl": _scalar(prob.get_val("cl")),
            "cd": _scalar(prob.get_val("cd")),
            "l_over_d": _scalar(prob.get_val("l_over_d")),
            "mass_kg": _scalar(prob.get_val("mass_kg")),
            "alpha_deg": _scalar(prob.get_val("alpha_deg")),
            "ks_failure_minus_1": _scalar(prob.get_val("ks_failure_minus_1")),
            "tip_deflection_minus_limit_m": _scalar(prob.get_val("tip_deflection_minus_limit_m")),
            "cl_minus_target": _scalar(prob.get_val("cl_minus_target")),
        },
        "last_evaluation": coupled.last_result,
        "evaluation_count": int(pipeline.eval_count),
    }
    return solution
