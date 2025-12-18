import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage import measure


class SimpleTriplaneDecoder(nn.Module):
    """Small MLP that expands a latent vector into three 2D feature planes.

    This is a lightweight placeholder for a true diffusion U-Net + triplane decoder.
    """

    def __init__(self, latent_dim: int = 64, plane_res: int = 128, feat_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.plane_res = plane_res
        self.feat_dim = feat_dim
        hidden = 256
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3 * plane_res * plane_res * feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.shape[0]
        out = self.mlp(z)
        out = out.view(B, 3, self.plane_res, self.plane_res, self.feat_dim)
        return out


class WingDecoder:
    """Decode a latent vector into an SDF volume and mesh.

    This class provides a small, self-contained triplane-style decoder and
    marching-cubes postprocessor. It is intentionally lightweight so it can
    be used as a development stub until a full Latent Diffusion Model is available.
    """

    def __init__(self, model_path: str = None, latent_dim: int = 64, plane_res: int = 128):
        self.latent_dim = latent_dim
        self.plane_res = plane_res
        self.device = torch.device("cpu")
        self.decoder = SimpleTriplaneDecoder(latent_dim=latent_dim, plane_res=plane_res).to(self.device)
        if model_path and os.path.exists(model_path):
            self.decoder.load_state_dict(torch.load(model_path, map_location=self.device))

    def decode_latent(self, latent_vector: np.ndarray, grid_size: int = 128, bbox: Tuple[float, float] = (-1.5, 1.5)) -> np.ndarray:
        """Decode `latent_vector` into an SDF volume on a cubic grid.

        Args:
            latent_vector: shape (latent_dim,) or (1,latent_dim)
            grid_size: resolution of the 3D SDF grid
            bbox: (min, max) coordinate range for sampling

        Returns:
            sdf_volume: numpy array of shape (grid_size, grid_size, grid_size)
        """
        z = torch.tensor(latent_vector.reshape(1, -1), dtype=torch.float32, device=self.device)
        planes = self.decoder(z)  # (1,3,R,R,F)
        planes = planes[0].detach().cpu().numpy()  # (3,R,R,F)

        # simple triplane query: for each 3D point, sample XY, XZ, YZ planes and average
        xs = np.linspace(bbox[0], bbox[1], grid_size)
        ys = np.linspace(bbox[0], bbox[1], grid_size)
        zs = np.linspace(bbox[0], bbox[1], grid_size)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1)  # (N,N,N,3)

        def sample_plane(plane_feat, u, v):
            # plane_feat: (R,R,F), u,v in [-1,1]
            R = plane_feat.shape[0]
            iu = ((u + 1) * 0.5 * (R - 1)).astype(int)
            iv = ((v + 1) * 0.5 * (R - 1)).astype(int)
            # clamp
            iu = np.clip(iu, 0, R - 1)
            iv = np.clip(iv, 0, R - 1)
            return plane_feat[iu, iv].mean(axis=-1)

        # normalize coords to [-1,1]
        def norm(v):
            return (v - bbox[0]) / (bbox[1] - bbox[0]) * 2 - 1

        U = norm(X)
        V = norm(Y)
        W = norm(Z)

        # sample XY plane with (x,y)
        s_xy = sample_plane(planes[0], U, V)
        # sample XZ plane with (x,z)
        s_xz = sample_plane(planes[1], U, W)
        # sample YZ plane with (y,z)
        s_yz = sample_plane(planes[2], V, W)

        sdf = (s_xy + s_xz + s_yz) / 3.0
        # simple signed distance approximation by shifting and scaling
        sdf = (sdf - np.mean(sdf)) / (np.std(sdf) + 1e-8)
        return sdf

    def sdf_to_mesh(self, sdf_volume: np.ndarray, level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Convert `sdf_volume` into a mesh using marching cubes.

        Returns vertices (N,3) and faces (M,3).
        """
        verts, faces, normals, values = measure.marching_cubes(sdf_volume, level=level)
        return verts, faces


if __name__ == "__main__":
    # quick smoke test
    dec = WingDecoder(latent_dim=64, plane_res=64)
    z = np.random.randn(64)
    sdf = dec.decode_latent(z, grid_size=64)
    v, f = dec.sdf_to_mesh(sdf, level=0.0)
    print("Generated mesh:", v.shape, f.shape)
