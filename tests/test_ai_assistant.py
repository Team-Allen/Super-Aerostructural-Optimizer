#!/usr/bin/env python3
"""
Test script for the AI-Powered Design Assistant
"""

import sys
sys.path.append('.')

from super_aerostructural_optimizer import InteractiveDesignAssistant, UserRequirements

def test_ai_assistant():
    """Test the AI assistant with predefined requirements"""
    print("🧪 Testing AI-Powered Design Assistant")
    print("="*50)
    
    # Create assistant
    assistant = InteractiveDesignAssistant()
    
    # Set predefined requirements (simulate user input)
    req = assistant.requirements
    req.aircraft_type = "general"
    req.design_purpose = "efficiency"
    req.design_speed = 250 / 3.6  # 250 km/h to m/s
    req.cruise_altitude = 3500
    req.angle_of_attack = 2.0
    req.target_lift_coefficient = 1.3
    req.max_drag_coefficient = 0.018
    req.min_lift_to_drag_ratio = 25
    req.max_chord_length = 2.5
    req.max_wingspan = 25
    req.material_type = "aluminum"
    req.safety_factor = 2.5
    
    # Calculate Reynolds number
    req.reynolds_number = assistant._calculate_reynolds_number()
    
    print("✅ Requirements set successfully!")
    assistant._display_requirements_summary()
    
    # Test airfoil database search
    print("\n🔍 Testing airfoil database search...")
    assistant._find_best_airfoil()
    
    print(f"\n🎯 Selected airfoil: {assistant.selected_airfoil.name}")
    print(f"   Performance score: {assistant.selected_airfoil.performance_score:.1f}/100")
    
    # Test performance evaluation
    print("\n🧮 Testing performance evaluation...")
    coords = assistant.selected_airfoil.coordinates
    performance = assistant._evaluate_performance(coords)
    
    print(f"   CL: {performance['cl']:.3f}")
    print(f"   CD: {performance['cd']:.4f}")
    print(f"   L/D: {performance['ld_ratio']:.1f}")
    
    print("\n✅ AI Assistant test completed successfully!")
    print("   Ready for interactive use!")

if __name__ == "__main__":
    test_ai_assistant()