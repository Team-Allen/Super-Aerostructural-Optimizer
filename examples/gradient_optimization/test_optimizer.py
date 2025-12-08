"""
Test the new gradient-based optimizer vs old random perturbation
"""

import sys
sys.path.append(r"f:\MDO LAB\_RootFiles")

from aerodynamic_ai_assistant import AerodynamicAIAssistant

print("=" * 60)
print("TESTING GRADIENT-BASED OPTIMIZATION")
print("=" * 60)

# Initialize assistant
assistant = AerodynamicAIAssistant()

# Set requirements
assistant.requirements.aircraft_type = "glider"
assistant.requirements.reynolds_number = 1e6
assistant.requirements.angle_of_attack = 5.0
assistant.requirements.target_lift_coefficient = 1.2
assistant.requirements.min_lift_to_drag_ratio = 40.0

print("\n🎯 User Requirements:")
print(f"   Aircraft: {assistant.requirements.aircraft_type}")
print(f"   Reynolds: {assistant.requirements.reynolds_number:.1e}")
print(f"   Angle of Attack: {assistant.requirements.angle_of_attack}°")
print(f"   Target CL: {assistant.requirements.target_lift_coefficient}")
print(f"   Minimum L/D: {assistant.requirements.min_lift_to_drag_ratio}")

# Run the full analysis workflow
assistant._find_optimal_airfoil()
print(f"\n✅ Selected: {assistant.selected_airfoil.name}")
print(f"   Score: {assistant.selected_airfoil.performance_score:.1f}/100")

# Run optimization
assistant._run_optimization()

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Selected Airfoil: {assistant.selected_airfoil.name}")
print(f"Baseline L/D: {assistant.final_performance['baseline_ld']:.1f}")
print(f"Optimized L/D: {assistant.final_performance['ld_ratio']:.1f}")
print(f"Improvement: {assistant.final_performance['improvement_percent']:+.2f}%")
print(f"Total Evaluations: {assistant.final_performance['total_evaluations']}")
print(f"Converged: {assistant.final_performance['converged']}")

# Show optimization history
if len(assistant.optimization_history) > 0:
    print("\n📊 Optimization History (first 10 evaluations):")
    for entry in assistant.optimization_history[:10]:
        print(f"   Eval {entry['iteration']:3d}: L/D = {entry['ld_ratio']:6.1f}  "
              f"(CL={entry['cl']:.3f}, CD={entry['cd']:.5f})")
