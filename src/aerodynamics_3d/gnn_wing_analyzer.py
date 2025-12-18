"""
GNN-Based 3D Wing Analyzer

Integrates the trained GNN model with 3D wing geometry to predict
aerodynamic performance. Applies GNN to each spanwise section and
aggregates results to compute wing-level forces and moments.

This is the core Stage 2 analysis tool that uses AirFRANS-trained GNN.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from aerodynamics_3d.wing_geometry import Wing3D, WingParameters, WingSection
from ml_models.gnn_model import create_model


class GNNWingAnalyzer:
    """
    3D wing aerodynamic analyzer using trained GNN model.
    
    Workflow:
    1. Take 3D wing geometry (from WingParameters)
    2. For each spanwise section:
       - Generate graph representation of airfoil
       - Predict flow field using GNN
       - Extract section CL, CD
    3. Integrate forces over span (Lifting Line correction)
    4. Return wing CL, CD, L/D
    
    Args:
        model_path: Path to trained GNN checkpoint
        device: Device to run inference on (cuda/cpu)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Load trained model if provided; otherwise create a default model with random weights
        if model_path and Path(model_path).exists():
            print(f"Loading GNN model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)

            # Extract model configuration
            config_path = Path(model_path).parent / 'config.json'
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Default configuration
                config = {
                    'num_features': 10,
                    'num_outputs': 3,  # p, u, v
                    'model_type': 'attention'
                }

            # Create model
            self.model = create_model(
                num_node_features=config['num_features'],
                num_outputs=config['num_outputs'],
                model_type=config.get('model_type', 'attention'),
                device=device
            )

            # Load weights (if present in checkpoint)
            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                except Exception:
                    print("⚠️  Warning: checkpoint keys do not match model architecture — using random weights.")

            self.model.eval()
            print(f"✅ Model loaded successfully!")
            print(f"   Device: {device}")
            print(f"   Model type: {config.get('model_type', 'attention')}")
        else:
            # Create default model with random weights
            print("No valid model checkpoint provided — creating default GNN with random weights.")
            config = {
                'num_features': 10,
                'num_outputs': 3,
                'model_type': 'attention'
            }
            self.model = create_model(
                num_node_features=config['num_features'],
                num_outputs=config['num_outputs'],
                model_type=config.get('model_type', 'attention'),
                device=device
            )
            self.model.eval()
            print(f"Created default model on device: {device}")
    
    def analyze_section(
        self,
        section: WingSection,
        Re: float,
        alpha: float,
        velocity: float = 50.0
    ) -> Dict[str, float]:
        """
        Analyze a single wing section using GNN.
        
        Args:
            section: Wing section to analyze
            Re: Reynolds number
            alpha: Angle of attack (degrees)
            velocity: Freestream velocity (m/s)
            
        Returns:
            Dictionary with CL, CD, CM for this section
        """
        # TODO: Convert airfoil coordinates to graph representation
        # This requires knowing the AirFRANS graph structure
        
        # For now, placeholder that integrates pressure distribution
        # In practice, you need to:
        # 1. Create mesh around airfoil
        # 2. Convert to PyTorch Geometric Data object
        # 3. Run through GNN
        # 4. Integrate pressure to get CL, CD
        
        with torch.no_grad():
            # Placeholder: Create graph from section
            graph_data = self._create_graph_from_section(section, Re, alpha)
            
            # Predict flow field
            flow_field = self.model(graph_data)
            
            # Integrate to get forces
            CL, CD, CM = self._integrate_forces(flow_field, section, velocity)
        
        return {
            'CL': CL,
            'CD': CD,
            'CM': CM,
            'Re': Re,
            'alpha': alpha
        }
    
    def _create_graph_from_section(self, section: WingSection, Re: float, alpha: float):
        """
        Convert wing section to graph representation for GNN.
        
        This needs to match the AirFRANS data format.
        TODO: Implement proper mesh generation and graph construction.
        """
        # Implement a lightweight graph generator from the airfoil surface
        # This will create a chain graph along the airfoil coordinates and
        # a few cross-connections so the GNN has local context.
        from torch_geometric.data import Data

        coords2d = np.asarray(section.airfoil_coords)  # [N, 2], normalized (0..1)
        if coords2d.ndim != 2 or coords2d.shape[1] < 2:
            raise ValueError("section.airfoil_coords must be [N,2]")

        # Scale by chord so feature magnitudes are physical
        chord = float(section.chord)
        coords_scaled = coords2d.copy()
        coords_scaled[:, 0] = coords2d[:, 0] * chord
        coords_scaled[:, 1] = coords2d[:, 1] * chord

        num_nodes = coords_scaled.shape[0]

        # Node positional tensor
        pos = torch.tensor(coords_scaled, dtype=torch.float32).to(self.device)

        # Simple geometric features: x, y, dy/dx (slope), curvature estimate
        dx = np.gradient(coords_scaled[:, 0])
        dy = np.gradient(coords_scaled[:, 1])
        slope = dy / (dx + 1e-8)
        # curvature: derivative of slope (approx)
        curvature = np.gradient(slope)

        # Normalize Re and alpha to reasonable ranges and broadcast to nodes
        Re_norm = float(Re) if Re is not None else 1e6
        alpha_norm = float(alpha) if alpha is not None else 0.0

        # Compose feature matrix (pad/truncate to expected feature size)
        feat_list = [coords_scaled[:, 0], coords_scaled[:, 1], slope, curvature]
        feat = np.vstack(feat_list).T  # shape [N,4]

        # Append Re and alpha as node-wise constant features
        feat = np.hstack([feat, np.full((num_nodes, 1), Re_norm), np.full((num_nodes, 1), alpha_norm)])

        # Target feature dimension for model (read from config when available).
        # We'll pad with zeros to 10 features which matches AirFRANS default.
        target_dim = 10
        if feat.shape[1] < target_dim:
            pad = np.zeros((num_nodes, target_dim - feat.shape[1]), dtype=float)
            feat = np.hstack([feat, pad])
        elif feat.shape[1] > target_dim:
            feat = feat[:, :target_dim]

        x = torch.tensor(feat, dtype=torch.float32).to(self.device)

        # Build edge_index: chain edges plus a few cross links between upper/lower
        edges = []
        # chain connections
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))

        # Connect corresponding upper/lower nodes for basic coupling
        # We'll connect i with (N-1-i) for a simple cross link
        for i in range(num_nodes // 2):
            j = num_nodes - 1 - i
            if j != i:
                edges.append((i, j))
                edges.append((j, i))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)

        data = Data(x=x, pos=pos, edge_index=edge_index)
        # Attach metadata for downstream use
        data.Re = Re_norm
        data.alpha = alpha_norm

        return data
    
    def _integrate_forces(
        self,
        flow_field: torch.Tensor,
        section: WingSection,
        velocity: float
    ) -> Tuple[float, float, float]:
        """
        Integrate pressure distribution to compute section forces.
        
        Args:
            flow_field: GNN output [num_nodes, num_features]
                        Typically [pressure, u_velocity, v_velocity]
            section: Wing section geometry
            velocity: Freestream velocity
            
        Returns:
            (CL, CD, CM) tuple
        """
        # Extract pressure from flow field
        # Assuming flow_field[:, 0] is pressure
        pressure = flow_field[:, 0].cpu().numpy()
        
        # TODO: Proper force integration on airfoil surface
        # For now, placeholder using typical values
        
        # Placeholder coefficients (replace with actual integration)
        CL = 0.8  # Typical cruise CL
        CD = 0.02  # Typical cruise CD
        CM = -0.1  # Typical pitching moment
        
        return CL, CD, CM
    
    def analyze_wing(
        self,
        wing: Wing3D,
        flight_conditions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Analyze complete 3D wing.
        
        Args:
            wing: 3D wing geometry
            flight_conditions: Dict with 'velocity', 'altitude', 'alpha'
            
        Returns:
            Dictionary with wing-level CL, CD, L/D, etc.
        """
        velocity = flight_conditions['velocity']
        alpha_wing = flight_conditions['alpha']
        altitude = flight_conditions.get('altitude', 0.0)
        
        # Air properties at altitude
        rho, mu = self._get_air_properties(altitude)
        
        # Analyze each section
        print(f"Analyzing {len(wing.sections)} spanwise sections...")
        
        section_results = []
        for i, section in enumerate(wing.sections):
            # Local Reynolds number
            Re_local = rho * velocity * section.chord / mu
            
            # Local angle of attack (wing alpha + section twist)
            alpha_local = alpha_wing + section.twist
            
            # Analyze section
            result = self.analyze_section(section, Re_local, alpha_local, velocity)
            result['y_position'] = section.y_position
            result['chord'] = section.chord
            
            section_results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(wing.sections)} sections")
        
        # Integrate along span using Lifting Line Theory
        wing_forces = self._lifting_line_integration(
            wing,
            section_results,
            velocity,
            rho
        )
        
        return wing_forces
    
    def _get_air_properties(self, altitude: float) -> Tuple[float, float]:
        """
        Get air density and viscosity at altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            (rho, mu) - density (kg/m³) and dynamic viscosity (Pa·s)
        """
        # Standard atmosphere (simplified)
        if altitude <= 11000:
            # Troposphere
            T = 288.15 - 0.0065 * altitude  # Temperature (K)
            p = 101325 * (T / 288.15) ** 5.2561  # Pressure (Pa)
        else:
            # Stratosphere (constant temp)
            T = 216.65
            p = 22632 * np.exp(-0.0001577 * (altitude - 11000))
        
        # Ideal gas law
        rho = p / (287.05 * T)
        
        # Sutherland's formula for viscosity
        mu = 1.458e-6 * T**1.5 / (T + 110.4)
        
        return rho, mu
    
    def _lifting_line_integration(
        self,
        wing: Wing3D,
        section_results: list,
        velocity: float,
        rho: float
    ) -> Dict[str, float]:
        """
        Integrate section forces using Lifting Line Theory.
        
        Accounts for:
        - Induced drag from 3D effects
        - Spanwise load distribution
        - Wing efficiency factor (Oswald efficiency)
        
        Args:
            wing: 3D wing geometry
            section_results: List of section analysis results
            velocity: Freestream velocity
            rho: Air density
            
        Returns:
            Wing-level aerodynamic coefficients
        """
        # Extract section data
        y_stations = np.array([r['y_position'] for r in section_results])
        chords = np.array([r['chord'] for r in section_results])
        CL_sections = np.array([r['CL'] for r in section_results])
        CD_sections = np.array([r['CD'] for r in section_results])
        
        # Trapezoidal integration for lift
        # CL_wing = (1 / S) * ∫ CL(y) * c(y) dy
        S = wing.get_area()
        
        # Lift distribution per unit span
        lift_dist = CL_sections * chords
        
        # Integrate (symmetric wing, so multiply by 2)
        CL_wing = 2 * np.trapz(lift_dist, y_stations) / S
        
        # Profile drag integration
        profile_drag_dist = CD_sections * chords
        CDp_wing = 2 * np.trapz(profile_drag_dist, y_stations) / S
        
        # Induced drag (from Lifting Line Theory)
        AR = wing.get_aspect_ratio()
        e = 0.85  # Oswald efficiency factor (typical for swept wing)
        
        CDi_wing = CL_wing**2 / (np.pi * e * AR)
        
        # Total drag
        CD_wing = CDp_wing + CDi_wing
        
        # L/D ratio
        L_D = CL_wing / CD_wing if CD_wing > 0 else 0
        
        # Detailed results
        results = {
            'CL': CL_wing,
            'CD': CD_wing,
            'CDi': CDi_wing,
            'CDp': CDp_wing,
            'L/D': L_D,
            'efficiency': CL_wing**1.5 / CD_wing if CD_wing > 0 else 0,
            'span': wing.params.span,
            'area': S,
            'aspect_ratio': AR,
            'section_results': section_results
        }
        
        return results


def analyze_wing_performance(
    wing_params: WingParameters,
    model_path: str,
    flight_conditions: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Convenience function to analyze wing performance.
    
    Args:
        wing_params: Wing design parameters
        model_path: Path to trained GNN checkpoint
        flight_conditions: Flight conditions (velocity, altitude, alpha)
        
    Returns:
        Aerodynamic performance metrics
    """
    if flight_conditions is None:
        flight_conditions = {
            'velocity': 50.0,  # m/s
            'altitude': 0.0,   # m
            'alpha': 5.0       # degrees
        }
    
    # Create wing geometry
    wing = Wing3D(wing_params)
    
    # Create analyzer
    analyzer = GNNWingAnalyzer(model_path)
    
    # Analyze
    results = analyzer.analyze_wing(wing, flight_conditions)
    
    return results


if __name__ == "__main__":
    """Test the GNN wing analyzer."""
    
    print("="*60)
    print("GNN Wing Analyzer Test")
    print("="*60)
    
    # Create test wing
    params = WingParameters(
        span=10.0,
        taper_ratio=0.6,
        sweep_angle=25.0,
        dihedral=3.0,
        twist_root=2.0,
        twist_tip=-3.0,
        n_sections=20  # Fewer sections for testing
    )
    
    # Note: This test requires a trained model
    # model_path = "training/checkpoints/best.pth"
    
    print("\n⚠️ To run full test, you need to:")
    print("1. Train the GNN model: python training/train_gnn.py")
    print("2. Update model_path to point to trained checkpoint")
    print("3. Implement graph generation from airfoil coordinates")
    
    print("\n✅ GNN Wing Analyzer structure is ready!")
    print("Next step: Train the GNN model on AirFRANS data")
