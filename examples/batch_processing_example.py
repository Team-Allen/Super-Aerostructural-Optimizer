"""
Batch Processing Example - AI Aerodynamic Design Assistant

This example demonstrates how to process multiple aircraft designs
in batch mode for systematic analysis and comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Try to import the AI assistant
try:
    from final_aerodynamic_ai import FinalAerodynamicAI
    AI_AVAILABLE = True
except ImportError:
    print("⚠️ AI Assistant not available - running in demo mode")
    AI_AVAILABLE = False

@dataclass
class BatchDesignConfig:
    """Configuration for a single aircraft design in batch processing"""
    name: str
    aircraft_type: str
    cruise_speed_kmh: float
    cruise_altitude_m: float
    target_ld_ratio: float
    priority: str  # "performance", "efficiency", "stability"

@dataclass
class BatchResult:
    """Results from batch processing a single design"""
    config_name: str
    selected_airfoil: str
    final_ld_ratio: float
    wing_span: float
    wing_area: float
    processing_time: float
    status: str

class BatchDesignProcessor:
    """Handles batch processing of multiple aircraft designs"""
    
    def __init__(self):
        """Initialize the batch processor"""
        self.ai_assistant = None
        if AI_AVAILABLE:
            try:
                self.ai_assistant = FinalAerodynamicAI()
                print("✅ AI Assistant initialized for batch processing")
            except Exception as e:
                print(f"⚠️ AI Assistant initialization failed: {e}")
                self.ai_assistant = None
        else:
            print("⚠️ Running in demo mode - no actual optimization")
    
    def process_batch(self, configurations: List[BatchDesignConfig]) -> List[BatchResult]:
        """Process multiple designs in batch"""
        
        print(f"\n🚀 BATCH PROCESSING: {len(configurations)} designs")
        print("=" * 60)
        
        results = []
        
        for i, config in enumerate(configurations, 1):
            print(f"\n📋 Processing {i}/{len(configurations)}: {config.name}")
            print(f"   Type: {config.aircraft_type}")
            print(f"   Speed: {config.cruise_speed_kmh} km/h")
            print(f"   Altitude: {config.cruise_altitude_m} m")
            
            start_time = time.time()
            
            try:
                if self.ai_assistant:
                    # Actual AI processing
                    result = self.process_single_design(config)
                else:
                    # Demo mode
                    result = self.generate_demo_result(config)
                
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                result.status = "success"
                
                print(f"   ✅ Completed in {processing_time:.2f}s")
                print(f"   Airfoil: {result.selected_airfoil}")
                print(f"   L/D: {result.final_ld_ratio:.2f}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                result = BatchResult(
                    config_name=config.name,
                    selected_airfoil="ERROR",
                    final_ld_ratio=0.0,
                    wing_span=0.0,
                    wing_area=0.0,
                    processing_time=processing_time,
                    status=f"error: {str(e)}"
                )
                print(f"   ❌ Failed in {processing_time:.2f}s: {e}")
            
            results.append(result)
        
        return results
    
    def process_single_design(self, config: BatchDesignConfig) -> BatchResult:
        """Process a single aircraft design using AI"""
        
        # Convert config to AI assistant format
        ai_requirements = self.config_to_ai_format(config)
        
        # Run optimization
        ai_results = self.ai_assistant.optimize_wing_design_programmatic(ai_requirements)
        
        # Convert results back
        return BatchResult(
            config_name=config.name,
            selected_airfoil=getattr(ai_results, 'selected_airfoil', 'NACA 0012'),
            final_ld_ratio=getattr(ai_results, 'final_ld_ratio', 20.0),
            wing_span=getattr(ai_results, 'wing_span', 12.0),
            wing_area=getattr(ai_results, 'wing_area', 16.0),
            processing_time=0.0,  # Will be set by caller
            status="success"
        )
    
    def generate_demo_result(self, config: BatchDesignConfig) -> BatchResult:
        """Generate demo results when AI isn't available"""
        
        # Simulate some processing time
        time.sleep(0.5)
        
        # Generate realistic demo data based on config
        airfoil_map = {
            "Sport Aircraft": "NACA 2412",
            "Training Aircraft": "NACA 0012",
            "General Aviation": "NACA 4412",
            "Touring Aircraft": "NACA 23012",
            "Aerobatic": "NACA 0009"
        }
        
        selected_airfoil = airfoil_map.get(config.aircraft_type, "NACA 0012")
        
        # Estimate L/D based on target and aircraft type
        ld_efficiency = {
            "Sport Aircraft": 0.95,
            "Training Aircraft": 0.85,
            "General Aviation": 0.90,
            "Touring Aircraft": 1.05,
            "Aerobatic": 0.80
        }
        
        efficiency = ld_efficiency.get(config.aircraft_type, 0.90)
        final_ld = config.target_ld_ratio * efficiency
        
        # Estimate wing dimensions
        wing_span = 8.0 + (config.cruise_speed_kmh - 150) / 50 * 2.0
        wing_area = 12.0 + (config.cruise_speed_kmh - 150) / 100 * 4.0
        
        return BatchResult(
            config_name=config.name,
            selected_airfoil=selected_airfoil,
            final_ld_ratio=final_ld,
            wing_span=wing_span,
            wing_area=wing_area,
            processing_time=0.0,
            status="demo"
        )
    
    def config_to_ai_format(self, config: BatchDesignConfig):
        """Convert batch config to AI assistant format"""
        
        class AIRequirements:
            def __init__(self):
                self.aircraft_type = config.aircraft_type
                self.cruise_speed_kmh = config.cruise_speed_kmh
                self.cruise_altitude_m = config.cruise_altitude_m
                self.target_lift_to_drag = config.target_ld_ratio
                self.reynolds_number = self.calculate_reynolds()
                self.max_thickness_ratio = 0.15
                self.manufacturing_constraint = "standard"
                self.design_priority = [config.priority]
            
            def calculate_reynolds(self):
                """Calculate Reynolds number"""
                altitude_km = config.cruise_altitude_m / 1000
                rho_sl = 1.225
                rho = rho_sl * (1 - 0.0065 * altitude_km / 288.15) ** 4.26
                mu = 1.81e-5
                velocity_ms = config.cruise_speed_kmh / 3.6
                chord = 2.0
                return (rho * velocity_ms * chord) / mu
        
        return AIRequirements()
    
    def generate_comparison_report(self, results: List[BatchResult], output_file: str = None):
        """Generate a comparison report from batch results"""
        
        print("\n📊 BATCH PROCESSING RESULTS SUMMARY")
        print("=" * 60)
        
        successful_results = [r for r in results if r.status == "success" or r.status == "demo"]
        
        if not successful_results:
            print("❌ No successful results to analyze")
            return
        
        # Sort by L/D ratio
        sorted_results = sorted(successful_results, key=lambda x: x.final_ld_ratio, reverse=True)
        
        print(f"\n🏆 TOP PERFORMERS (by L/D ratio):")
        print("-" * 40)
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. {result.config_name}")
            print(f"   Airfoil: {result.selected_airfoil}")
            print(f"   L/D: {result.final_ld_ratio:.2f}")
            print(f"   Wing: {result.wing_span:.1f}m span, {result.wing_area:.1f}m² area")
            print(f"   Processing: {result.processing_time:.2f}s")
            print()
        
        # Statistics
        ld_ratios = [r.final_ld_ratio for r in successful_results]
        processing_times = [r.processing_time for r in successful_results]
        
        print("📈 STATISTICS:")
        print("-" * 20)
        print(f"Configurations processed: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(results) - len(successful_results)}")
        print(f"Average L/D ratio: {sum(ld_ratios)/len(ld_ratios):.2f}")
        print(f"Best L/D ratio: {max(ld_ratios):.2f}")
        print(f"Average processing time: {sum(processing_times)/len(processing_times):.2f}s")
        print(f"Total processing time: {sum(processing_times):.2f}s")
        
        # Airfoil usage
        airfoil_counts = {}
        for result in successful_results:
            airfoil = result.selected_airfoil
            airfoil_counts[airfoil] = airfoil_counts.get(airfoil, 0) + 1
        
        print("\n🛩️ AIRFOIL SELECTION FREQUENCY:")
        print("-" * 30)
        for airfoil, count in sorted(airfoil_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful_results)) * 100
            print(f"{airfoil}: {count} times ({percentage:.1f}%)")
        
        # Save to file if requested
        if output_file:
            self.save_results_to_file(results, successful_results, output_file)
            print(f"\n💾 Results saved to: {output_file}")
    
    def save_results_to_file(self, all_results: List[BatchResult], successful_results: List[BatchResult], filename: str):
        """Save results to JSON file"""
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_configurations": len(all_results),
                "successful": len(successful_results),
                "failed": len(all_results) - len(successful_results),
                "average_ld_ratio": sum(r.final_ld_ratio for r in successful_results) / len(successful_results) if successful_results else 0,
                "total_processing_time": sum(r.processing_time for r in all_results)
            },
            "detailed_results": [asdict(result) for result in all_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)

def create_sample_configurations() -> List[BatchDesignConfig]:
    """Create sample aircraft configurations for batch processing"""
    
    return [
        BatchDesignConfig(
            name="Sport Trainer",
            aircraft_type="Training Aircraft",
            cruise_speed_kmh=160,
            cruise_altitude_m=1500,
            target_ld_ratio=18,
            priority="stability"
        ),
        BatchDesignConfig(
            name="Cross-Country Tourer",
            aircraft_type="Touring Aircraft",
            cruise_speed_kmh=300,
            cruise_altitude_m=4000,
            target_ld_ratio=28,
            priority="efficiency"
        ),
        BatchDesignConfig(
            name="Weekend Sport",
            aircraft_type="Sport Aircraft",
            cruise_speed_kmh=250,
            cruise_altitude_m=2500,
            target_ld_ratio=24,
            priority="performance"
        ),
        BatchDesignConfig(
            name="Aerobatic Special",
            aircraft_type="Aerobatic",
            cruise_speed_kmh=200,
            cruise_altitude_m=2000,
            target_ld_ratio=16,
            priority="performance"
        ),
        BatchDesignConfig(
            name="General Aviation",
            aircraft_type="General Aviation",
            cruise_speed_kmh=220,
            cruise_altitude_m=3000,
            target_ld_ratio=22,
            priority="efficiency"
        ),
        BatchDesignConfig(
            name="High-Speed Cruiser",
            aircraft_type="Sport Aircraft",
            cruise_speed_kmh=350,
            cruise_altitude_m=5000,
            target_ld_ratio=26,
            priority="performance"
        )
    ]

def demonstrate_batch_processing():
    """Main demonstration of batch processing"""
    
    print("🔄 AI AERODYNAMIC DESIGN ASSISTANT - BATCH PROCESSING DEMO")
    print("=" * 65)
    print("This example shows how to process multiple aircraft designs")
    print("automatically and compare the results.\n")
    
    # Create sample configurations
    configurations = create_sample_configurations()
    
    print(f"📋 Created {len(configurations)} sample aircraft configurations:")
    for i, config in enumerate(configurations, 1):
        print(f"{i}. {config.name} ({config.aircraft_type})")
    
    # Initialize processor
    processor = BatchDesignProcessor()
    
    # Process all configurations
    print("\n🚀 Starting batch processing...")
    start_time = time.time()
    
    results = processor.process_batch(configurations)
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Batch processing completed in {total_time:.2f} seconds")
    
    # Generate comparison report
    output_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    processor.generate_comparison_report(results, output_filename)
    
    print("\n✅ BATCH PROCESSING DEMONSTRATION COMPLETE!")
    print("\nNext steps:")
    print("• Modify the configurations to test your own aircraft designs")
    print("• Add more configurations to the list")
    print("• Use the JSON output for further analysis")
    print("• Integrate with your own design workflows")

if __name__ == "__main__":
    try:
        demonstrate_batch_processing()
    except KeyboardInterrupt:
        print("\n\n⚠️ Batch processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Batch processing failed: {e}")
        print("This might be because source files aren't in the expected location.")
        print("Try running from the repository root directory.")