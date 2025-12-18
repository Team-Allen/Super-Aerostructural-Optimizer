"""
3D Wing Geometry Generator

Creates 3D wing geometries from design parameters (span, taper, sweep, twist, etc.)
and generates spanwise sections for analysis with GNN or NeuralFoil.

This bridges 2D airfoil optimization (Stage 1) with 3D wing analysis (Stage 2).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class WingParameters:
    """
    Wing design parameters for 3D geometry.
    
    These are the variables that will be optimized in Stage 2.
    """
    # Planform geometry
    span: float = 10.0              # Wing span (m)
    root_chord: float = 2.0          # Root chord length (m)
    taper_ratio: float = 0.6         # tip_chord / root_chord
    sweep_angle: float = 25.0        # Quarter-chord sweep (degrees)
    dihedral: float = 3.0            # Dihedral angle (degrees)
    
    # Twist distribution
    twist_root: float = 2.0          # Root twist (degrees)
    twist_tip: float = -3.0          # Tip twist (degrees)
    
    # Airfoil sections
    airfoil_root: str = "NACA4412"   # Root airfoil (from Stage 1)
    airfoil_tip: str = "NACA2412"    # Tip airfoil (from Stage 1)
    
    # Analysis parameters
    n_sections: int = 30             # Number of spanwise sections
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for optimization."""
        return {
            'span': self.span,
            'taper_ratio': self.taper_ratio,
            'sweep_angle': self.sweep_angle,
            'dihedral': self.dihedral,
            'twist_root': self.twist_root,
            'twist_tip': self.twist_tip,
        }
    
    @classmethod
    def from_array(cls, x: np.ndarray, airfoil_root: str = "NACA4412", airfoil_tip: str = "NACA2412"):
        """Create from optimization array."""
        return cls(
            span=x[0],
            taper_ratio=x[1],
            sweep_angle=x[2],
            dihedral=x[3],
            twist_root=x[4],
            twist_tip=x[5],
            airfoil_root=airfoil_root,
            airfoil_tip=airfoil_tip
        )


class WingSection:
    """
    Represents a single spanwise section of the wing.
    
    Contains local geometry, flow conditions, and aerodynamic properties.
    """
    
    def __init__(
        self,
        y_position: float,
        chord: float,
        twist: float,
        airfoil_name: str,
        airfoil_coords: np.ndarray,
        leading_edge_position: np.ndarray
    ):
        self.y_position = y_position
        self.chord = chord
        self.twist = twist
        self.airfoil_name = airfoil_name
        self.airfoil_coords = airfoil_coords  # [N, 2] array
        self.leading_edge_position = leading_edge_position  # [x, y, z]
        
        # Aerodynamic properties (filled by analysis)
        self.Re = None
        self.alpha_local = None
        self.CL = None
        self.CD = None
        self.CM = None
    
    def get_3d_coordinates(self) -> np.ndarray:
        """
        Get 3D coordinates of this section in wing reference frame.
        
        Returns:
            coords_3d: [N, 3] array of (x, y, z) coordinates
        """
        # Scale airfoil by chord
        coords_2d = self.airfoil_coords * self.chord
        
        # Apply twist rotation (in x-z plane)
        twist_rad = np.radians(self.twist)
        cos_twist = np.cos(twist_rad)
        sin_twist = np.sin(twist_rad)
        
        x_rotated = coords_2d[:, 0] * cos_twist - coords_2d[:, 1] * sin_twist
        z_rotated = coords_2d[:, 0] * sin_twist + coords_2d[:, 1] * cos_twist
        
        # Create 3D coordinates
        coords_3d = np.zeros((len(coords_2d), 3))
        coords_3d[:, 0] = x_rotated + self.leading_edge_position[0]
        coords_3d[:, 1] = self.y_position
        coords_3d[:, 2] = z_rotated + self.leading_edge_position[2]
        
        return coords_3d


class Wing3D:
    """
    3D wing geometry generator.
    
    Creates wing geometry from parameters and provides sections for analysis.
    """
    
    def __init__(self, params: WingParameters):
        self.params = params
        self.sections: List[WingSection] = []
        
        # Generate geometry
        self._generate_sections()
    
    def _generate_sections(self):
        """Generate spanwise sections."""
        n = self.params.n_sections
        
        # Spanwise positions (half wing, symmetry assumed)
        y_positions = np.linspace(0, self.params.span / 2, n)
        
        # Compute tip chord
        tip_chord = self.params.root_chord * self.params.taper_ratio
        
        for y in y_positions:
            # Interpolate chord (linear taper)
            eta = y / (self.params.span / 2)  # Normalized spanwise position
            chord = self.params.root_chord * (1 - eta) + tip_chord * eta
            
            # Interpolate twist
            twist = self.params.twist_root * (1 - eta) + self.params.twist_tip * eta
            
            # Interpolate airfoil (for now, use root airfoil everywhere)
            # TODO: Implement airfoil morphing between root and tip
            airfoil_name = self.params.airfoil_root
            airfoil_coords = self._load_airfoil(airfoil_name)
            
            # Compute leading edge position
            # Quarter-chord sweep
            sweep_rad = np.radians(self.params.sweep_angle)
            x_le = y * np.tan(sweep_rad) - 0.25 * chord
            
            # Dihedral
            dihedral_rad = np.radians(self.params.dihedral)
            z_le = y * np.tan(dihedral_rad)
            
            leading_edge_position = np.array([x_le, y, z_le])
            
            # Create section
            section = WingSection(
                y_position=y,
                chord=chord,
                twist=twist,
                airfoil_name=airfoil_name,
                airfoil_coords=airfoil_coords,
                leading_edge_position=leading_edge_position
            )
            
            self.sections.append(section)
    
    def _load_airfoil(self, airfoil_name: str) -> np.ndarray:
        """
        Load airfoil coordinates.
        
        For now, returns NACA 4-digit airfoil. 
        TODO: Load from optimized Stage 1 results or database.
        """
        # Simple NACA 4-digit generator (placeholder)
        # In production, load from your airfoil database
        
        # Generate points along chord
        n_points = 100
        x = np.linspace(0, 1, n_points)
        
        # Simple symmetric airfoil (placeholder)
        thickness = 0.12  # 12% thickness
        y_upper = thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        y_lower = -y_upper
        
        # Combine upper and lower surfaces
        coords = np.zeros((2 * n_points - 1, 2))
        coords[:n_points, 0] = x[::-1]  # Upper surface (LE to TE)
        coords[:n_points, 1] = y_upper[::-1]
        coords[n_points:, 0] = x[1:]  # Lower surface (TE to LE)
        coords[n_points:, 1] = y_lower[1:]
        
        return coords
    
    def get_area(self) -> float:
        """Compute wing planform area."""
        # Trapezoidal wing area
        root_chord = self.params.root_chord
        tip_chord = root_chord * self.params.taper_ratio
        span = self.params.span
        
        area = 0.5 * (root_chord + tip_chord) * span
        return area
    
    def get_aspect_ratio(self) -> float:
        """Compute wing aspect ratio."""
        return self.params.span**2 / self.get_area()
    
    def visualize(self, show_sections: bool = True, save_path: Optional[str] = None):
        """
        Visualize 3D wing geometry.
        
        Args:
            show_sections: Whether to show individual sections
            save_path: Path to save figure (if None, displays interactively)
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each section
        for section in self.sections:
            coords = section.get_3d_coordinates()
            
            if show_sections:
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', linewidth=1, alpha=0.6)
        
        # Plot leading and trailing edges
        le_points = np.array([s.leading_edge_position for s in self.sections])
        ax.plot(le_points[:, 0], le_points[:, 1], le_points[:, 2], 'r-', linewidth=2, label='Leading Edge')
        
        # Trailing edge (approx)
        te_points = le_points + np.array([[s.chord, 0, 0] for s in self.sections])
        ax.plot(te_points[:, 0], te_points[:, 1], te_points[:, 2], 'g-', linewidth=2, label='Trailing Edge')
        
        # Labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (Span, m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Wing Geometry\nSpan: {self.params.span:.1f}m, AR: {self.get_aspect_ratio():.1f}')
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.array([
            te_points[:, 0].max() - le_points[:, 0].min(),
            le_points[:, 1].max(),
            le_points[:, 2].max()
        ]).max() / 2.0
        
        mid_x = (te_points[:, 0].max() + le_points[:, 0].min()) * 0.5
        mid_y = le_points[:, 1].max() * 0.5
        mid_z = le_points[:, 2].max() * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved wing visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def summary(self):
        """Print wing geometry summary."""
        print(f"\n{'='*60}")
        print(f"3D Wing Geometry Summary")
        print(f"{'='*60}")
        print(f"Span:              {self.params.span:.2f} m")
        print(f"Root Chord:        {self.params.root_chord:.2f} m")
        print(f"Tip Chord:         {self.params.root_chord * self.params.taper_ratio:.2f} m")
        print(f"Taper Ratio:       {self.params.taper_ratio:.3f}")
        print(f"Sweep (c/4):       {self.params.sweep_angle:.1f}°")
        print(f"Dihedral:          {self.params.dihedral:.1f}°")
        print(f"Twist (root):      {self.params.twist_root:.1f}°")
        print(f"Twist (tip):       {self.params.twist_tip:.1f}°")
        print(f"Planform Area:     {self.get_area():.2f} m²")
        print(f"Aspect Ratio:      {self.get_aspect_ratio():.2f}")
        print(f"Number of Sections: {len(self.sections)}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    """Test the wing geometry generator."""
    
    # Create default wing parameters
    params = WingParameters(
        span=10.0,
        root_chord=2.0,
        taper_ratio=0.6,
        sweep_angle=25.0,
        dihedral=3.0,
        twist_root=2.0,
        twist_tip=-3.0,
        n_sections=30
    )
    
    # Generate wing
    print("Generating 3D wing geometry...")
    wing = Wing3D(params)
    
    # Print summary
    wing.summary()
    
    # Visualize
    print("Visualizing wing...")
    wing.visualize(show_sections=True, save_path="wing_geometry_test.png")
    
    # Test optimization array conversion
    print("\nTesting optimization interface...")
    x = np.array([10.0, 0.6, 25.0, 3.0, 2.0, -3.0])
    params_from_array = WingParameters.from_array(x)
    wing2 = Wing3D(params_from_array)
    wing2.summary()
    
    print("✅ Wing geometry test complete!")
