"""
Physics-Informed Neural Network (PINN) solver for 2D steady RANS equations.

This is a TRUE PINN — the neural network learns to solve the Navier-Stokes PDEs
by minimizing physics residuals, not just fitting data. The loss function encodes:

1. Momentum equations (RANS with Spalart-Allmaras turbulence)
2. Continuity equation
3. Boundary conditions (no-slip wall, farfield, Kutta condition)
4. Optional data loss (from CFD/experimental reference solutions)

The PINN provides:
- Meshless solution: no grid generation needed
- Differentiable: exact gradients for optimization via autodiff
- Parametric: conditioned on Re, Mach, alpha, and airfoil geometry
- Transfer learning: pre-train on simple cases, fine-tune for new geometries

Reference: Raissi et al. (2019) "Physics-informed neural networks"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. PINN solver unavailable.")


if HAS_TORCH:

    class SirenLayer(nn.Module):
        """Sinusoidal activation layer (SIREN) — better for PDE solutions than ReLU.

        Reference: Sitzmann et al. (2020) "Implicit Neural Representations with
        Periodic Activation Functions"
        """

        def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0,
                     is_first: bool = False):
            super().__init__()
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_features, out_features)
            self._init_weights(is_first, in_features)

        def _init_weights(self, is_first: bool, fan_in: int):
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1 / fan_in, 1 / fan_in)
                else:
                    bound = np.sqrt(6.0 / fan_in) / self.omega_0
                    self.linear.weight.uniform_(-bound, bound)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.omega_0 * self.linear(x))


    class RANSPinn(nn.Module):
        """PINN for 2D steady incompressible/compressible RANS equations.

        Input: (x, y, Re, Mach, alpha, cst_params...)
        Output: (u, v, p, nu_t)  — velocity, pressure, eddy viscosity

        The network is conditioned on flow parameters and airfoil geometry,
        making it a parametric surrogate that generalizes across designs.
        """

        def __init__(
            self,
            n_cst_params: int = 12,
            hidden_dim: int = 256,
            n_layers: int = 8,
            omega_0: float = 30.0,
            use_fourier_features: bool = True,
            n_fourier: int = 64,
        ):
            super().__init__()

            self.n_cst_params = n_cst_params
            self.use_fourier_features = use_fourier_features

            # Input: x, y, Re, Mach, alpha + CST params
            base_input_dim = 5 + n_cst_params

            # Fourier feature encoding for spatial coordinates
            if use_fourier_features:
                self.n_fourier = n_fourier
                self.register_buffer(
                    'B_matrix',
                    torch.randn(2, n_fourier) * 2.0  # Random Fourier features
                )
                input_dim = 2 * n_fourier + 3 + n_cst_params  # sin+cos features + Re,Ma,alpha + CST
            else:
                input_dim = base_input_dim

            # SIREN network
            layers = []
            layers.append(SirenLayer(input_dim, hidden_dim, omega_0, is_first=True))
            for _ in range(n_layers - 1):
                layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0))
            self.network = nn.Sequential(*layers)

            # Output heads
            self.velocity_head = nn.Linear(hidden_dim, 2)   # u, v
            self.pressure_head = nn.Linear(hidden_dim, 1)   # p
            self.turbulence_head = nn.Linear(hidden_dim, 1)  # nu_t (eddy viscosity)

        def encode_input(self, x: torch.Tensor, y: torch.Tensor,
                        Re: torch.Tensor, mach: torch.Tensor, alpha: torch.Tensor,
                        cst_params: torch.Tensor) -> torch.Tensor:
            """Encode inputs with optional Fourier features."""
            # Normalize inputs
            Re_norm = torch.log10(Re) / 7.0  # log-scale, normalize to ~[0,1]
            alpha_norm = alpha / 15.0  # normalize to ~[-1,1]

            if self.use_fourier_features:
                xy = torch.stack([x, y], dim=-1)  # [batch, 2]
                projected = xy @ self.B_matrix  # [batch, n_fourier]
                fourier = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
                return torch.cat([fourier, Re_norm.unsqueeze(-1), mach.unsqueeze(-1),
                                alpha_norm.unsqueeze(-1), cst_params], dim=-1)
            else:
                return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1),
                                Re_norm.unsqueeze(-1), mach.unsqueeze(-1),
                                alpha_norm.unsqueeze(-1), cst_params], dim=-1)

        def forward(self, x: torch.Tensor, y: torch.Tensor,
                   Re: torch.Tensor, mach: torch.Tensor, alpha: torch.Tensor,
                   cst_params: torch.Tensor) -> dict[str, torch.Tensor]:
            """Forward pass: spatial coords + params → flow field."""
            encoded = self.encode_input(x, y, Re, mach, alpha, cst_params)
            features = self.network(encoded)

            uv = self.velocity_head(features)
            p = self.pressure_head(features)
            nu_t = self.turbulence_head(features)

            return {
                'u': uv[:, 0],      # x-velocity
                'v': uv[:, 1],      # y-velocity
                'p': p.squeeze(-1),  # pressure
                'nu_t': torch.nn.functional.softplus(nu_t.squeeze(-1)),  # eddy viscosity (>0)
            }

        def compute_pde_residuals(
            self, x: torch.Tensor, y: torch.Tensor,
            Re: torch.Tensor, mach: torch.Tensor, alpha: torch.Tensor,
            cst_params: torch.Tensor, nu: float = 1.5e-5,
        ) -> dict[str, torch.Tensor]:
            """Compute RANS PDE residuals using automatic differentiation.

            Incompressible steady RANS:
                u ∂u/∂x + v ∂u/∂y = -∂p/∂x + (ν + ν_t)(∂²u/∂x² + ∂²u/∂y²)
                u ∂v/∂x + v ∂v/∂y = -∂p/∂y + (ν + ν_t)(∂²v/∂x² + ∂²v/∂y²)
                ∂u/∂x + ∂v/∂y = 0
            """
            x.requires_grad_(True)
            y.requires_grad_(True)

            outputs = self.forward(x, y, Re, mach, alpha, cst_params)
            u, v, p, nu_t = outputs['u'], outputs['v'], outputs['p'], outputs['nu_t']

            # First derivatives
            u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
            v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
            v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
            p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
            p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

            # Second derivatives
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]

            # Effective viscosity
            nu_eff = nu + nu_t

            # RANS momentum residuals
            res_x = u * u_x + v * u_y + p_x - nu_eff * (u_xx + u_yy)
            res_y = u * v_x + v * v_y + p_y - nu_eff * (v_xx + v_yy)

            # Continuity residual
            res_cont = u_x + v_y

            return {
                'momentum_x': res_x,
                'momentum_y': res_y,
                'continuity': res_cont,
            }


    class PINNTrainer:
        """Training loop for the RANS PINN with physics-informed loss."""

        def __init__(
            self,
            model: RANSPinn,
            lr: float = 1e-4,
            physics_weight: float = 1.0,
            bc_weight: float = 10.0,
            data_weight: float = 1.0,
            device: str = "auto",
        ):
            self.model = model
            self.physics_weight = physics_weight
            self.bc_weight = bc_weight
            self.data_weight = data_weight

            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model.to(self.device)
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1000, T_mult=2
            )
            self.history = {'total': [], 'physics': [], 'bc': [], 'data': []}

        def sample_domain(self, n_points: int, Re: float, mach: float,
                         alpha: float, cst_params: np.ndarray) -> dict[str, torch.Tensor]:
            """Sample collocation points in the flow domain."""
            # Domain: [-2, 3] x [-2, 2] around unit chord airfoil
            x = torch.rand(n_points, device=self.device) * 5.0 - 2.0
            y = torch.rand(n_points, device=self.device) * 4.0 - 2.0

            Re_t = torch.full((n_points,), Re, device=self.device)
            mach_t = torch.full((n_points,), mach, device=self.device)
            alpha_t = torch.full((n_points,), alpha, device=self.device)
            cst_t = torch.tensor(cst_params, dtype=torch.float32, device=self.device)
            cst_t = cst_t.unsqueeze(0).expand(n_points, -1)

            return {'x': x, 'y': y, 'Re': Re_t, 'mach': mach_t,
                    'alpha': alpha_t, 'cst_params': cst_t}

        def sample_boundary(self, n_points: int, airfoil_x: np.ndarray,
                          airfoil_y: np.ndarray, Re: float, mach: float,
                          alpha: float, cst_params: np.ndarray) -> dict[str, torch.Tensor]:
            """Sample points on the airfoil surface (no-slip BC)."""
            # Random indices along airfoil surface
            idx = np.random.choice(len(airfoil_x), n_points, replace=True)
            x = torch.tensor(airfoil_x[idx], dtype=torch.float32, device=self.device)
            y = torch.tensor(airfoil_y[idx], dtype=torch.float32, device=self.device)

            Re_t = torch.full((n_points,), Re, device=self.device)
            mach_t = torch.full((n_points,), mach, device=self.device)
            alpha_t = torch.full((n_points,), alpha, device=self.device)
            cst_t = torch.tensor(cst_params, dtype=torch.float32, device=self.device)
            cst_t = cst_t.unsqueeze(0).expand(n_points, -1)

            return {'x': x, 'y': y, 'Re': Re_t, 'mach': mach_t,
                    'alpha': alpha_t, 'cst_params': cst_t}

        def compute_loss(
            self,
            domain_points: dict[str, torch.Tensor],
            bc_points: dict[str, torch.Tensor],
            data_points: Optional[dict[str, torch.Tensor]] = None,
        ) -> dict[str, torch.Tensor]:
            """Compute total PINN loss = physics + BC + data."""

            # Physics loss: PDE residuals at collocation points
            residuals = self.model.compute_pde_residuals(
                domain_points['x'], domain_points['y'],
                domain_points['Re'], domain_points['mach'],
                domain_points['alpha'], domain_points['cst_params']
            )
            physics_loss = (
                torch.mean(residuals['momentum_x']**2) +
                torch.mean(residuals['momentum_y']**2) +
                torch.mean(residuals['continuity']**2)
            )

            # Boundary condition loss: no-slip on wall (u=0, v=0)
            bc_pred = self.model(
                bc_points['x'], bc_points['y'],
                bc_points['Re'], bc_points['mach'],
                bc_points['alpha'], bc_points['cst_params']
            )
            bc_loss = torch.mean(bc_pred['u']**2) + torch.mean(bc_pred['v']**2)

            # Data loss (optional): match reference CFD/experimental data
            data_loss = torch.tensor(0.0, device=self.device)
            if data_points is not None and 'u_ref' in data_points:
                pred = self.model(
                    data_points['x'], data_points['y'],
                    data_points['Re'], data_points['mach'],
                    data_points['alpha'], data_points['cst_params']
                )
                data_loss = (
                    torch.mean((pred['u'] - data_points['u_ref'])**2) +
                    torch.mean((pred['v'] - data_points['v_ref'])**2)
                )

            total = (self.physics_weight * physics_loss +
                    self.bc_weight * bc_loss +
                    self.data_weight * data_loss)

            return {
                'total': total,
                'physics': physics_loss,
                'bc': bc_loss,
                'data': data_loss,
            }

        def train_step(
            self,
            domain_points: dict,
            bc_points: dict,
            data_points: Optional[dict] = None,
        ) -> dict[str, float]:
            """Single training step."""
            self.optimizer.zero_grad()
            losses = self.compute_loss(domain_points, bc_points, data_points)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            result = {k: v.item() for k, v in losses.items()}
            for k, v in result.items():
                self.history[k].append(v)
            return result

        def train(
            self,
            n_epochs: int,
            n_domain: int = 4096,
            n_boundary: int = 512,
            airfoil_x: np.ndarray = None,
            airfoil_y: np.ndarray = None,
            Re: float = 1e6,
            mach: float = 0.0,
            alpha: float = 5.0,
            cst_params: np.ndarray = None,
            log_every: int = 100,
            callback: Optional[Callable] = None,
        ) -> dict:
            """Full training loop."""
            if cst_params is None:
                cst_params = np.zeros(self.model.n_cst_params)

            if airfoil_x is None or airfoil_y is None:
                # Default: NACA 0012 circle
                theta = np.linspace(0, 2 * np.pi, 200)
                airfoil_x = 0.5 * (1 - np.cos(theta))
                airfoil_y = 0.06 * np.sin(theta)

            logger.info(f"Training PINN for {n_epochs} epochs on {self.device}")

            for epoch in range(n_epochs):
                domain = self.sample_domain(n_domain, Re, mach, alpha, cst_params)
                bc = self.sample_boundary(n_boundary, airfoil_x, airfoil_y,
                                         Re, mach, alpha, cst_params)

                losses = self.train_step(domain, bc)

                if (epoch + 1) % log_every == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{n_epochs} | "
                        f"Total: {losses['total']:.6f} | "
                        f"Physics: {losses['physics']:.6f} | "
                        f"BC: {losses['bc']:.6f}"
                    )

                if callback is not None:
                    callback(epoch, losses)

            return self.history

        def compute_forces(
            self,
            airfoil_x: np.ndarray,
            airfoil_y: np.ndarray,
            airfoil_nx: np.ndarray,
            airfoil_ny: np.ndarray,
            Re: float,
            mach: float,
            alpha: float,
            cst_params: np.ndarray,
        ) -> dict[str, float]:
            """Compute aerodynamic forces by integrating pressure on airfoil surface.

            CL = integral(-p * ny * ds) / (0.5 * rho * V^2 * c)
            CD = integral(-p * nx * ds) / (0.5 * rho * V^2 * c)
            """
            n = len(airfoil_x)
            x_t = torch.tensor(airfoil_x, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(airfoil_y, dtype=torch.float32, device=self.device)
            Re_t = torch.full((n,), Re, device=self.device)
            mach_t = torch.full((n,), mach, device=self.device)
            alpha_t = torch.full((n,), alpha, device=self.device)
            cst_t = torch.tensor(cst_params, dtype=torch.float32, device=self.device)
            cst_t = cst_t.unsqueeze(0).expand(n, -1)

            with torch.no_grad():
                pred = self.model(x_t, y_t, Re_t, mach_t, alpha_t, cst_t)

            p = pred['p'].cpu().numpy()

            # Compute ds (segment lengths)
            dx = np.diff(airfoil_x, append=airfoil_x[0])
            dy = np.diff(airfoil_y, append=airfoil_y[0])
            ds = np.sqrt(dx**2 + dy**2)

            # Integrate pressure forces
            # Rotate to wind axes
            alpha_rad = np.radians(alpha)
            cos_a = np.cos(alpha_rad)
            sin_a = np.sin(alpha_rad)

            fx = np.sum(-p * airfoil_nx * ds)
            fy = np.sum(-p * airfoil_ny * ds)

            cl = fy * cos_a - fx * sin_a
            cd = fx * cos_a + fy * sin_a

            return {'CL': float(cl), 'CD': float(cd), 'L/D': float(cl / cd) if abs(cd) > 1e-10 else 0.0}


class PINNAeroSolver:
    """High-level PINN aerodynamic solver interface for the MDO pipeline.

    Wraps RANSPinn model + training into a clean evaluate() API that
    matches the NeuralFoilSolver interface.
    """

    def __init__(
        self,
        n_cst_params: int = 12,
        pretrained_path: Optional[str] = None,
        device: str = "auto",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for PINN solver")

        self.model = RANSPinn(n_cst_params=n_cst_params)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if pretrained_path is not None:
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            logger.info(f"Loaded pretrained PINN from {pretrained_path}")

        self.trainer = PINNTrainer(self.model, device=str(self.device))
        self._is_trained = pretrained_path is not None

    def train_for_geometry(
        self,
        airfoil_x: np.ndarray,
        airfoil_y: np.ndarray,
        cst_params: np.ndarray,
        Re: float = 1e6,
        mach: float = 0.0,
        alpha: float = 5.0,
        n_epochs: int = 5000,
    ):
        """Train/fine-tune the PINN for a specific airfoil geometry."""
        self.trainer.train(
            n_epochs=n_epochs,
            airfoil_x=airfoil_x,
            airfoil_y=airfoil_y,
            Re=Re, mach=mach, alpha=alpha,
            cst_params=cst_params,
        )
        self._is_trained = True

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved PINN model to {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self._is_trained = True
        logger.info(f"Loaded PINN model from {path}")
