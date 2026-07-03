import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

repo = Path(r'f:\\MDO LAB\\Super-Aerostructural-Optimizer')
progress_path = repo / 'results' / 'real_physics' / 'live' / 'proof_progress.ndjson'
opt_path = repo / 'results' / 'real_physics' / 'real_physics_proof_optimum.json'
report_path = repo / 'results' / 'real_physics' / 'e2e_test' / 'real_physics_e2e_e2e_report.json'
out_dir = repo / 'results' / 'real_physics' / 'proof_graphs'
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for line in progress_path.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    ev = json.loads(line)
    if ev.get('event') == 'coupling_iteration':
        it = ev['iteration']
        rows.append({
            'eval': int(ev['evaluation_id']),
            'iter': int(it['iteration']),
            'cl': float(it['cl']),
            'cd': float(it['cd']),
            'mass': float(it['mass_kg']),
            'ks': float(it['ks_failure']),
            'tip': float(it['tip_deflection_m']),
            'load_rel': float(it['load_rel_change']),
            'disp_rel': float(it['disp_rel_change']),
            'handoff': 1.0 if bool(it['handoff_finite']) else 0.0,
            'aero_load_norm': float(it['aero_load_norm']),
            'struct_load_norm': float(it['struct_load_norm']),
            'aero_disp_norm': float(it['aero_disp_norm']),
            'struct_disp_norm': float(it['struct_disp_norm']),
        })

if not rows:
    raise RuntimeError('No coupling_iteration events found in progress file')

for i, r in enumerate(rows, start=1):
    r['step'] = i

x = np.array([r['step'] for r in rows])
cl = np.array([r['cl'] for r in rows])
cd = np.array([r['cd'] for r in rows])
load_rel = np.array([r['load_rel'] for r in rows])
disp_rel = np.array([r['disp_rel'] for r in rows])
handoff = np.array([r['handoff'] for r in rows])
aero_load_norm = np.array([r['aero_load_norm'] for r in rows])
struct_load_norm = np.array([r['struct_load_norm'] for r in rows])
aero_disp_norm = np.array([r['aero_disp_norm'] for r in rows])
struct_disp_norm = np.array([r['struct_disp_norm'] for r in rows])

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs[0,0].plot(x, cl, marker='o', label='CL')
axs[0,0].plot(x, cd, marker='s', label='CD')
axs[0,0].set_title('Coupling Iterations: CL/CD')
axs[0,0].set_xlabel('Global Coupling Step')
axs[0,0].grid(True, alpha=0.3)
axs[0,0].legend()

axs[0,1].plot(x, load_rel, marker='o', label='load_rel_change')
axs[0,1].plot(x, disp_rel, marker='s', label='disp_rel_change')
axs[0,1].set_title('Coupling Convergence Metrics')
axs[0,1].set_xlabel('Global Coupling Step')
axs[0,1].set_yscale('log')
axs[0,1].grid(True, which='both', alpha=0.3)
axs[0,1].legend()

axs[1,0].plot(x, aero_load_norm, marker='o', label='Aero load norm')
axs[1,0].plot(x, struct_load_norm, marker='s', label='Struct load norm')
axs[1,0].set_title('Load Norms Through Transfer')
axs[1,0].set_xlabel('Global Coupling Step')
axs[1,0].grid(True, alpha=0.3)
axs[1,0].legend()

axs[1,1].plot(x, aero_disp_norm, marker='o', label='Aero disp norm')
axs[1,1].plot(x, struct_disp_norm, marker='s', label='Struct disp norm')
axs[1,1].set_title('Displacement Norms Through Transfer')
axs[1,1].set_xlabel('Global Coupling Step')
axs[1,1].grid(True, alpha=0.3)
axs[1,1].legend()

fig.tight_layout()
fig.savefig(out_dir / 'proof_coupling_handoff_metrics.png', dpi=180)
plt.close(fig)

by_eval = {}
for r in rows:
    by_eval[r['eval']] = r

evals = sorted(by_eval)
final_cd = np.array([by_eval[e]['cd'] for e in evals])
final_ks = np.array([by_eval[e]['ks'] for e in evals])
final_tip = np.array([by_eval[e]['tip'] for e in evals])
final_handoff = np.array([by_eval[e]['handoff'] for e in evals])

fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axs[0].plot(evals, final_cd, marker='o', label='Final CD / eval')
axs[0].plot(evals, final_ks, marker='s', label='Final KS / eval')
axs[0].set_title('Optimization Evaluations: Final Coupling Outputs')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(evals, final_tip, marker='o', label='Final tip deflection (m)')
axs[1].step(evals, final_handoff, where='mid', label='handoff_finite (1=OK)')
axs[1].set_xlabel('Evaluation ID')
axs[1].set_ylabel('Tip Defl / Handoff')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

fig.tight_layout()
fig.savefig(out_dir / 'proof_eval_summary.png', dpi=180)
plt.close(fig)

rep = json.loads(report_path.read_text(encoding='utf-8'))
backend = rep.get('cuda_report', {})
labels = ['torch', 'jax', 'cupy']
avail = [1 if backend.get(k, {}).get('installed', False) else 0 for k in labels]
gpu_ok = [
    1 if backend.get('torch', {}).get('cuda_available', False) else 0,
    1 if backend.get('jax', {}).get('gpu_devices', 0) > 0 else 0,
    1 if backend.get('cupy', {}).get('cuda_available', False) else 0,
]

xpos = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(xpos - width/2, avail, width, label='Installed')
ax.bar(xpos + width/2, gpu_ok, width, label='GPU Active')
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(['No', 'Yes'])
ax.set_title('CUDA Python Toolchain Readiness (mdo-best)')
ax.grid(True, axis='y', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / 'proof_cuda_backend_status.png', dpi=180)
plt.close(fig)

opt = json.loads(opt_path.read_text(encoding='utf-8'))
outs = opt['outputs']
labels = ['objective_cd', 'cl', 'cd', 'l_over_d', 'mass_kg', 'ks_failure_minus_1', 'tip_deflection_minus_limit_m']
vals = [float(outs[k]) for k in labels]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(np.arange(len(labels)), vals)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_title('Proof Run: Final Optimization Outputs')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / 'proof_opt_outputs.png', dpi=180)
plt.close(fig)

summary = {
    'progress_file': str(progress_path),
    'proof_optimum_file': str(opt_path),
    'e2e_report_file': str(report_path),
    'num_coupling_events': len(rows),
    'num_evaluations': len(evals),
    'all_handoff_finite': bool(np.all(handoff > 0.5)),
    'generated_graphs': [
        'proof_coupling_handoff_metrics.png',
        'proof_eval_summary.png',
        'proof_cuda_backend_status.png',
        'proof_opt_outputs.png',
    ]
}
(summary_path := out_dir / 'proof_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Wrote', summary_path)
for name in summary['generated_graphs']:
    print('Wrote', out_dir / name)
