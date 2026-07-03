"""Desktop GUI viewer for live aerostructural pipeline monitoring."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


class DesktopPipelineViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Super Aerostructural Optimizer - Live Pipeline Viewer")
        self.geometry("1400x900")
        self.minsize(1200, 760)

        self.repo_root = Path(__file__).resolve().parents[1]
        self.run_script = self.repo_root / "run_scripts" / "run_real_physics_mdo_wsl.ps1"
        self.default_config = self.repo_root / "configs" / "real_physics_pipeline.json"
        self.example_config = self.repo_root / "configs" / "real_physics_pipeline.json.example"
        self.live_dir = self.repo_root / "results" / "real_physics" / "live"
        self.live_dir.mkdir(parents=True, exist_ok=True)

        self.process: Optional[subprocess.Popen[str]] = None
        self.stop_event = threading.Event()
        self.queue: queue.Queue = queue.Queue()
        self.progress_file: Optional[Path] = None
        self.progress_offset = 0
        self.global_step = 0

        self._build_ui()
        self._reset_runtime_state()
        self.after(100, self._drain_queue)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(5, weight=1)

        ttk.Label(top, text="Config").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.config_var = tk.StringVar(value=str(self.default_config))
        ttk.Entry(top, textvariable=self.config_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(top, text="Browse", command=self._browse_config).grid(
            row=0, column=2, padx=(8, 12)
        )

        ttk.Label(top, text="Mode").grid(row=0, column=3, sticky="e", padx=(0, 8))
        self.mode_var = tk.StringVar(value="analyze")
        ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["analyze", "optimize"],
            state="readonly",
            width=12,
        ).grid(row=0, column=4, sticky="w")

        self.require_cuda_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top,
            text="Require CUDA (hardware)",
            variable=self.require_cuda_var,
        ).grid(row=0, column=5, sticky="w", padx=(12, 0))

        ttk.Label(top, text="CUDA_VISIBLE_DEVICES").grid(
            row=0, column=6, sticky="e", padx=(12, 6)
        )
        self.cuda_visible_var = tk.StringVar(value="0")
        ttk.Entry(top, textvariable=self.cuda_visible_var, width=8).grid(
            row=0, column=7, sticky="w"
        )

        ttk.Button(top, text="Start Run", command=self._start_run).grid(
            row=0, column=8, padx=(12, 6)
        )
        ttk.Button(top, text="Stop", command=self._stop_run).grid(row=0, column=9, padx=(0, 6))
        ttk.Button(top, text="Run E2E Test", command=self._run_e2e_test).grid(row=0, column=10)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=2)
        main.add(right, weight=3)

        # Left: stage + key metrics + run metadata
        left.columnconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(left, text="Run Status", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(left, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.stage_vars: Dict[str, tk.StringVar] = {}
        for idx, stage in enumerate(["Aero", "Transfer", "TACS", "Coupling", "Optimizer"]):
            ttk.Label(left, text=f"{stage}:").grid(row=2 + idx, column=0, sticky="w")
            var = tk.StringVar(value="idle")
            ttk.Label(left, textvariable=var).grid(row=2 + idx, column=1, sticky="w")
            self.stage_vars[stage] = var

        ttk.Separator(left, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=10
        )

        ttk.Label(left, text="Current Metrics", font=("Segoe UI", 11, "bold")).grid(
            row=8, column=0, columnspan=2, sticky="w"
        )
        metric_names = [
            ("Eval ID", "eval_id"),
            ("Iteration", "iteration"),
            ("CL", "cl"),
            ("CD", "cd"),
            ("L/D", "lod"),
            ("Mass (kg)", "mass"),
            ("KS Failure", "ks"),
            ("Tip Defl. (m)", "tip"),
            ("Load Rel. Change", "load_rel"),
            ("Disp Rel. Change", "disp_rel"),
            ("Handoff Finite", "handoff"),
        ]
        self.metric_vars: Dict[str, tk.StringVar] = {}
        for i, (label, key) in enumerate(metric_names):
            ttk.Label(left, text=f"{label}:").grid(row=9 + i, column=0, sticky="w")
            var = tk.StringVar(value="-")
            ttk.Label(left, textvariable=var).grid(row=9 + i, column=1, sticky="w")
            self.metric_vars[key] = var

        self.progress_file_var = tk.StringVar(value="-")
        ttk.Separator(left, orient="horizontal").grid(
            row=9 + len(metric_names), column=0, columnspan=2, sticky="ew", pady=10
        )
        ttk.Label(left, text="Progress File:").grid(
            row=10 + len(metric_names), column=0, sticky="nw"
        )
        ttk.Label(left, textvariable=self.progress_file_var, wraplength=420).grid(
            row=10 + len(metric_names), column=1, sticky="w"
        )

        # Right: chart + logs
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        self.chart_frame = ttk.LabelFrame(right, text="Live Charts", padding=8)
        self.chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.chart_frame.rowconfigure(0, weight=1)
        self.chart_frame.columnconfigure(0, weight=1)

        if HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(7.5, 4.5), dpi=100)
            self.ax_aero = self.figure.add_subplot(121)
            self.ax_conv = self.figure.add_subplot(122)
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        else:
            ttk.Label(
                self.chart_frame,
                text="matplotlib not available in this Python environment.",
            ).grid(row=0, column=0, sticky="w")
            self.figure = None
            self.ax_aero = None
            self.ax_conv = None
            self.canvas = None

        logs_frame = ttk.LabelFrame(right, text="Live Logs", padding=8)
        logs_frame.grid(row=1, column=0, sticky="nsew")
        logs_frame.rowconfigure(0, weight=1)
        logs_frame.columnconfigure(0, weight=1)

        self.logs = tk.Text(logs_frame, wrap="word", height=16)
        self.logs.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.logs.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        self.logs.configure(yscrollcommand=ybar.set)

    def _reset_runtime_state(self) -> None:
        self.global_step = 0
        self.data_x = []
        self.data_cl = []
        self.data_cd = []
        self.data_mass = []
        self.data_ks = []
        self.data_tip = []
        self.data_load_rel = []
        self.data_disp_rel = []
        self.data_handoff = []
        for stage in self.stage_vars.values():
            stage.set("idle")

    def _browse_config(self) -> None:
        p = filedialog.askopenfilename(
            title="Select pipeline config",
            initialdir=str(self.repo_root / "configs"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if p:
            self.config_var.set(p)

    def _ensure_config_exists(self, config_path: Path) -> Path:
        if config_path.exists():
            return config_path
        if self.example_config.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(self.example_config.read_text(encoding="utf-8"), encoding="utf-8")
            self._append_log(f"Config missing, created from example: {config_path}")
            return config_path
        raise FileNotFoundError(f"Config not found: {config_path}")

    def _build_progress_file(self, tag: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.live_dir / f"progress_{tag}_{stamp}.ndjson"

    def _start_run(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Run in progress", "A pipeline run is already active.")
            return

        self._reset_runtime_state()
        self.logs.delete("1.0", tk.END)
        self.status_var.set("Starting...")

        mode = self.mode_var.get().strip()
        config_path = Path(self.config_var.get()).expanduser()
        if not config_path.is_absolute():
            config_path = (self.repo_root / config_path).resolve()
        try:
            config_path = self._ensure_config_exists(config_path)
        except Exception as exc:
            messagebox.showerror("Config error", str(exc))
            return

        self.progress_file = self._build_progress_file(mode)
        self.progress_offset = 0
        self.progress_file_var.set(str(self.progress_file))
        self.stop_event.clear()

        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.run_script),
            "-Mode",
            mode,
            "-Config",
            str(config_path),
            "-ProgressFile",
            str(self.progress_file),
        ]
        cuda_visible = self.cuda_visible_var.get().strip()
        if cuda_visible:
            cmd.extend(["-CudaVisibleDevices", cuda_visible])

        self._append_log("Launching command:")
        self._append_log(" ".join(cmd))

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.status_var.set("Failed to start")
            messagebox.showerror("Launch error", str(exc))
            return

        threading.Thread(target=self._read_stdout_thread, daemon=True).start()
        threading.Thread(target=self._tail_progress_thread, daemon=True).start()
        self.after(300, self._monitor_process)

    def _run_e2e_test(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Run in progress", "Stop current run before launching E2E test.")
            return

        self._reset_runtime_state()
        self.logs.delete("1.0", tk.END)
        self.status_var.set("Running E2E test...")

        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.repo_root / "run_scripts" / "test_real_physics_e2e_wsl.ps1"),
            "-CouplingIters",
            "8",
            "-OptimizerIters",
            "3",
        ]
        cuda_visible = self.cuda_visible_var.get().strip()
        if cuda_visible:
            cmd.extend(["-CudaVisibleDevices", cuda_visible])
        if self.require_cuda_var.get():
            cmd.append("-RequireCuda")
        self._append_log("Launching command:")
        self._append_log(" ".join(cmd))
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.status_var.set("Failed to start")
            messagebox.showerror("Launch error", str(exc))
            return

        threading.Thread(target=self._read_stdout_thread, daemon=True).start()
        self.after(300, self._monitor_process)

    def _stop_run(self) -> None:
        self.stop_event.set()
        if self.process is not None and self.process.poll() is None:
            self._append_log("Stopping process...")
            self.process.terminate()
            time.sleep(0.3)
            if self.process.poll() is None:
                self.process.kill()
        self.status_var.set("Stopped")

    def _read_stdout_thread(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self.stop_event.is_set():
                break
            self.queue.put(("log", line.rstrip("\n")))

    def _tail_progress_thread(self) -> None:
        while not self.stop_event.is_set():
            path = self.progress_file
            if path is None:
                break
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        handle.seek(self.progress_offset)
                        chunk = handle.read()
                        self.progress_offset = handle.tell()
                    if chunk:
                        for line in chunk.splitlines():
                            if not line.strip():
                                continue
                            try:
                                payload = json.loads(line)
                            except Exception:
                                continue
                            self.queue.put(("event", payload))
                except Exception:
                    pass
            time.sleep(0.15)

    def _monitor_process(self) -> None:
        proc = self.process
        if proc is None:
            return
        code = proc.poll()
        if code is None:
            self.after(300, self._monitor_process)
            return
        self.stop_event.set()
        if code == 0:
            self.status_var.set("Completed")
        else:
            self.status_var.set(f"Failed (exit={code})")
        self._append_log(f"Process exited with code {code}")

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "event":
                    self._handle_event(payload)
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _handle_event(self, event: Dict[str, object]) -> None:
        evt = str(event.get("event", ""))
        if evt == "run_start":
            mode = event.get("mode", "?")
            self.status_var.set(f"Running ({mode})")
            self.stage_vars["Aero"].set("pending")
            self.stage_vars["Transfer"].set("pending")
            self.stage_vars["TACS"].set("pending")
            self.stage_vars["Coupling"].set("pending")
            self.stage_vars["Optimizer"].set("pending" if mode == "optimize" else "n/a")
            self._append_log(f"[event] run_start mode={mode}")
            return

        if evt == "coupling_iteration":
            it = event.get("iteration", {})
            eval_id = event.get("evaluation_id", "-")

            self.global_step += 1
            self.metric_vars["eval_id"].set(str(eval_id))
            self.metric_vars["iteration"].set(str(it.get("iteration", "-")))
            cl = float(it.get("cl", 0.0))
            cd = float(it.get("cd", 0.0))
            mass = float(it.get("mass_kg", 0.0))
            ks = float(it.get("ks_failure", 0.0))
            tip = float(it.get("tip_deflection_m", 0.0))
            load_rel = float(it.get("load_rel_change", 0.0))
            disp_rel = float(it.get("disp_rel_change", 0.0))
            handoff_ok = bool(it.get("handoff_finite", False))
            lod = cl / max(cd, 1.0e-12)

            self.metric_vars["cl"].set(f"{cl:.6f}")
            self.metric_vars["cd"].set(f"{cd:.6f}")
            self.metric_vars["lod"].set(f"{lod:.3f}")
            self.metric_vars["mass"].set(f"{mass:.3f}")
            self.metric_vars["ks"].set(f"{ks:.6f}")
            self.metric_vars["tip"].set(f"{tip:.6f}")
            self.metric_vars["load_rel"].set(f"{load_rel:.6f}")
            self.metric_vars["disp_rel"].set(f"{disp_rel:.6f}")
            self.metric_vars["handoff"].set("YES" if handoff_ok else "NO")

            self.stage_vars["Aero"].set("running")
            self.stage_vars["Transfer"].set("running")
            self.stage_vars["TACS"].set("running")
            self.stage_vars["Coupling"].set("running")
            if handoff_ok:
                self.stage_vars["Aero"].set("ok")
                self.stage_vars["Transfer"].set("ok")
                self.stage_vars["TACS"].set("ok")
                self.stage_vars["Coupling"].set("ok")

            self.data_x.append(self.global_step)
            self.data_cl.append(cl)
            self.data_cd.append(cd)
            self.data_mass.append(mass)
            self.data_ks.append(ks)
            self.data_tip.append(tip)
            self.data_load_rel.append(load_rel)
            self.data_disp_rel.append(disp_rel)
            self.data_handoff.append(1.0 if handoff_ok else 0.0)
            self._refresh_plots()

            self._append_log(
                "[event] eval=%s iter=%s cl=%.5f cd=%.5f mass=%.2f ks=%.4f tip=%.4f handoff=%s"
                % (
                    eval_id,
                    it.get("iteration", "-"),
                    cl,
                    cd,
                    mass,
                    ks,
                    tip,
                    "ok" if handoff_ok else "bad",
                )
            )
            return

        if evt == "analysis_complete":
            self.status_var.set("Analysis complete")
            self.stage_vars["Optimizer"].set("n/a")
            self._append_log(f"[event] analysis_complete -> {event.get('output_file', '')}")
            return

        if evt == "optimization_start":
            self.stage_vars["Optimizer"].set("running")
            self._append_log("[event] optimization_start")
            return

        if evt == "optimization_complete":
            self.stage_vars["Optimizer"].set("ok")
            self._append_log(f"[event] optimization_complete -> {event.get('output_file', '')}")
            return

        if evt == "run_error":
            self.status_var.set("Error")
            self._append_log(
                "[event] run_error type=%s message=%s"
                % (event.get("error_type", "?"), event.get("error_message", ""))
            )
            return

        self._append_log(f"[event] {evt}: {event}")

    def _refresh_plots(self) -> None:
        if not HAS_MATPLOTLIB or self.canvas is None:
            return
        if len(self.data_x) == 0:
            return

        assert self.ax_aero is not None
        assert self.ax_conv is not None

        self.ax_aero.clear()
        self.ax_aero.plot(self.data_x, self.data_cl, label="CL")
        self.ax_aero.plot(self.data_x, self.data_cd, label="CD")
        self.ax_aero.set_title("Aerodynamic Coefficients")
        self.ax_aero.set_xlabel("Global Coupling Step")
        self.ax_aero.grid(True, alpha=0.3)
        self.ax_aero.legend(loc="best")

        self.ax_conv.clear()
        self.ax_conv.plot(self.data_x, self.data_load_rel, label="Load rel change")
        self.ax_conv.plot(self.data_x, self.data_disp_rel, label="Disp rel change")
        self.ax_conv.set_title("Coupling Convergence")
        self.ax_conv.set_xlabel("Global Coupling Step")
        self.ax_conv.grid(True, alpha=0.3)
        self.ax_conv.legend(loc="best")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _append_log(self, text: str) -> None:
        self.logs.insert(tk.END, text + "\n")
        self.logs.see(tk.END)


def main() -> int:
    app = DesktopPipelineViewer()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
