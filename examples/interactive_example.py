"""
Interactive Example - AI Aerodynamic Design Assistant

This example demonstrates how to use the AI assistant in an interactive
programming environment for custom aircraft design scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from final_aerodynamic_ai import FinalAerodynamicAI
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class CustomRequirements:
    """Custom requirements for specialized aircraft"""
    aircraft_type: str
    mission_profile: str
    cruise_speed_kmh: float
    cruise_altitude_m: float
    payload_kg: float
    range_km: float
    runway_length_m: Optional[float] = None
    stall_speed_max_kmh: Optional[float] = None
    climb_rate_ms: Optional[float] = None

def demonstrate_interactive_workflow():
    """Demonstrate the complete interactive workflow"""
    
    print("🛩️ INTERACTIVE AI AERODYNAMIC DESIGN EXAMPLE")
    print("=" * 55)
    print("This example shows how to use the AI assistant programmatically")
    print("while maintaining the conversational design approach.\n")
    
    # Initialize the AI assistant
    print("🤖 Initializing AI Assistant...")
    assistant = FinalAerodynamicAI()
    
    print("✅ AI Assistant ready!\n")
    
    # Example 1: Sport Aircraft Design
    print("📋 EXAMPLE 1: Sport Aircraft Design")
    print("-" * 40)
    
    sport_requirements = CustomRequirements(
        aircraft_type="Sport Aircraft",
        mission_profile="Recreational flying with aerobatic capability",
        cruise_speed_kmh=280,
        cruise_altitude_m=2500,
        payload_kg=200,
        range_km=800,
        stall_speed_max_kmh=85,
        climb_rate_ms=8.0
    )
    
    print(f"Mission: {sport_requirements.mission_profile}")
    print(f"Speed: {sport_requirements.cruise_speed_kmh} km/h")
    print(f"Altitude: {sport_requirements.cruise_altitude_m} m")
    print(f"Payload: {sport_requirements.payload_kg} kg")
    
    # Convert to AI assistant format
    ai_requirements = convert_to_ai_format(sport_requirements)
    
    # Run optimization
    print("\n🔥 Running AI optimization...")
    try:
        results = assistant.optimize_wing_design_programmatic(ai_requirements)
        display_results("Sport Aircraft", results)
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        print("Using demo mode instead...")
        demonstrate_with_mock_data("Sport Aircraft")
    
    print("\n" + "="*55 + "\n")
    
    # Example 2: Training Aircraft Design
    print("📋 EXAMPLE 2: Training Aircraft Design")
    print("-" * 40)
    
    trainer_requirements = CustomRequirements(
        aircraft_type="Training Aircraft",
        mission_profile="Flight training with forgiving characteristics",
        cruise_speed_kmh=160,
        cruise_altitude_m=1500,
        payload_kg=180,
        range_km=400,
        stall_speed_max_kmh=65,
        climb_rate_ms=5.0
    )
    
    print(f"Mission: {trainer_requirements.mission_profile}")
    print(f"Speed: {trainer_requirements.cruise_speed_kmh} km/h")
    print(f"Focus: Low stall speed, stable flight characteristics")
    
    ai_requirements_2 = convert_to_ai_format(trainer_requirements)
    
    print("\n🔥 Running AI optimization...")
    try:
        results = assistant.optimize_wing_design_programmatic(ai_requirements_2)
        display_results("Training Aircraft", results)
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        demonstrate_with_mock_data("Training Aircraft")
    
    print("\n" + "="*55 + "\n")
    
    # Example 3: Custom Interactive Session
    print("📋 EXAMPLE 3: Custom Interactive Session")
    print("-" * 40)
    print("Now you can try your own requirements!\n")
    
    try:
        # Get user input for custom design
        custom_aircraft = get_user_requirements()
        
        if custom_aircraft:
            ai_requirements_3 = convert_to_ai_format(custom_aircraft)
            
            print("\n🔥 Running AI optimization for your aircraft...")
            try:
                results = assistant.optimize_wing_design_programmatic(ai_requirements_3)
                display_results("Your Custom Aircraft", results)
            except Exception as e:
                print(f"⚠️ Optimization failed: {e}")
                demonstrate_with_mock_data("Your Custom Aircraft")
    
    except KeyboardInterrupt:
        print("\n⚠️ Custom session skipped by user")
    
    print("\n🎉 INTERACTIVE EXAMPLES COMPLETE!")
    print("Next steps:")
    print("• Modify the CustomRequirements to try different scenarios")
    print("• Run src/final_aerodynamic_ai.py for full interactive mode")
    print("• Check examples/batch_processing_example.py for automation")

def convert_to_ai_format(requirements: CustomRequirements):
    """Convert custom requirements to AI assistant format"""
    
    # Simple conversion - in real implementation, this would be more sophisticated
    class AIRequirements:
        def __init__(self):
            self.aircraft_type = requirements.aircraft_type
            self.cruise_speed_kmh = requirements.cruise_speed_kmh
            self.cruise_altitude_m = requirements.cruise_altitude_m
            self.target_lift_to_drag = None
            self.reynolds_number = None
            self.max_thickness_ratio = 0.15
            self.manufacturing_constraint = "standard"
            
            # Calculate derived parameters
            self.calculate_reynolds_number()
            self.estimate_target_ld()
        
        def calculate_reynolds_number(self):
            """Calculate Reynolds number from flight conditions"""
            # Simplified atmospheric model
            altitude_km = self.cruise_altitude_m / 1000
            
            # Air density at altitude (simplified)
            rho_sl = 1.225  # kg/m³ at sea level
            rho = rho_sl * (1 - 0.0065 * altitude_km / 288.15) ** 4.26
            
            # Dynamic viscosity (constant approximation)
            mu = 1.81e-5  # kg/(m·s)
            
            # Velocity and characteristic length
            velocity_ms = self.cruise_speed_kmh / 3.6
            chord_estimate = 2.0  # meters
            
            self.reynolds_number = (rho * velocity_ms * chord_estimate) / mu
        
        def estimate_target_ld(self):
            """Estimate target L/D ratio based on aircraft type"""
            ld_targets = {
                "Sport Aircraft": 25,
                "Training Aircraft": 20,
                "General Aviation": 22,
                "Aerobatic": 18,
                "Touring": 28
            }
            
            self.target_lift_to_drag = ld_targets.get(self.aircraft_type, 20)
    
    return AIRequirements()

def display_results(aircraft_name: str, results):
    """Display optimization results in a formatted way"""
    
    print(f"\n🎯 RESULTS FOR {aircraft_name.upper()}:")
    print("-" * (15 + len(aircraft_name)))
    
    if hasattr(results, 'selected_airfoil'):
        print(f"Selected Airfoil: {results.selected_airfoil}")
    
    if hasattr(results, 'final_ld_ratio'):
        print(f"Final L/D Ratio: {results.final_ld_ratio:.2f}")
    
    if hasattr(results, 'wing_span'):
        print(f"Wing Span: {results.wing_span:.1f} m")
    
    if hasattr(results, 'wing_area'):
        print(f"Wing Area: {results.wing_area:.1f} m²")
    
    print("✅ Optimization completed successfully!")

def demonstrate_with_mock_data(aircraft_name: str):
    """Show mock results when optimization isn't available"""
    
    print(f"\n🎯 MOCK RESULTS FOR {aircraft_name.upper()}:")
    print("-" * (20 + len(aircraft_name)))
    print("Selected Airfoil: NACA 2412")
    print("Final L/D Ratio: 24.5")
    print("Wing Span: 10.2 m")
    print("Wing Area: 14.8 m²")
    print("✅ Demo completed successfully!")

def get_user_requirements() -> Optional[CustomRequirements]:
    """Get custom requirements from user input"""
    
    print("Let's design your custom aircraft!")
    print("(Press Ctrl+C to skip this section)\n")
    
    try:
        aircraft_type = input("Aircraft type (e.g., 'Sport', 'Trainer', 'Touring'): ").strip()
        if not aircraft_type:
            aircraft_type = "Custom"
        
        mission = input("Mission profile (e.g., 'Cross-country touring'): ").strip()
        if not mission:
            mission = "General aviation"
        
        speed_input = input("Cruise speed (km/h) [200]: ").strip()
        cruise_speed = float(speed_input) if speed_input else 200
        
        altitude_input = input("Cruise altitude (m) [3000]: ").strip()
        cruise_altitude = float(altitude_input) if altitude_input else 3000
        
        payload_input = input("Payload (kg) [200]: ").strip()
        payload = float(payload_input) if payload_input else 200
        
        range_input = input("Range (km) [500]: ").strip()
        range_km = float(range_input) if range_input else 500
        
        return CustomRequirements(
            aircraft_type=aircraft_type,
            mission_profile=mission,
            cruise_speed_kmh=cruise_speed,
            cruise_altitude_m=cruise_altitude,
            payload_kg=payload,
            range_km=range_km
        )
    
    except (ValueError, EOFError):
        print("⚠️ Invalid input, using default values")
        return None

if __name__ == "__main__":
    try:
        demonstrate_interactive_workflow()
    except KeyboardInterrupt:
        print("\n\n⚠️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("This might be because source files aren't in the expected location.")
        print("Try running from the repository root directory.")