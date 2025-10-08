import torch
import torch.nn as nn
import numpy as np

from .measurer import simulate_ray_3d

class BaseMeasurementSimulatorModule(nn.Module):
    def __init__(self, nx, ny, nz, steps, dmin=None, dmax=None, hard_data_locations=None):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.steps = steps
        self.hard_data_locations = hard_data_locations
        self.dmin, self.dmax = dmin, dmax

    def forward(self, preds, m_ens=None):
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_hard_data(self, m_ens, hard_data_locations):
        assert len(m_ens.shape) == 5, "Expected m_ens to have shape (nsample, 1, NX, NY, NZ)."
        hard_data = [m_ens[:, 0, x, y, z] for x, y, z in hard_data_locations]
        return torch.stack(hard_data, dim=-1)

    def normalize(self, d_sim):
        if self.dmin is None or self.dmax is None:
            return d_sim
        return (d_sim - self.dmin) / (self.dmax - self.dmin)


class RayPathMeasurementSimulatorModule(BaseMeasurementSimulatorModule):
    def __init__(self, nx, ny, nz, steps, dmin=None, dmax=None, hard_data_locations=None, vertical_interval=8, horizontal_interval=32, marginal=32):
        super().__init__(nx, ny, nz, steps, dmin, dmax, hard_data_locations)
        self.vertical_interval = vertical_interval
        self.horizontal_interval = horizontal_interval
        self.marginal = marginal
        self.x_start, self.x_end = marginal - 1, ny - marginal
        self.register_buffer("G3D", self.simulate_ray_paths()[0])
        self.register_buffer("G", self.simulate_ray_paths()[1])
        self.register_buffer("mask", self.simulate_ray_paths()[2])
        self.register_buffer("density", self.simulate_ray_paths()[3])

    def simulate_ray_paths(self):
        self.shooters = [[depth, self.ny // 2 - 1, 0] for depth in range(0, self.nx, self.vertical_interval)]
        self.receivers = [[0, surface_loc, 0] for surface_loc in range(self.x_start, self.x_end, self.horizontal_interval) if surface_loc != self.ny // 2]
        
        ray_paths = [simulate_ray_3d(self.nx, self.ny, self.nz, shooter, receiver) for shooter in self.shooters for receiver in self.receivers]
        G3D = torch.tensor(np.stack(ray_paths, axis=-1), dtype=torch.float32)
        G = G3D.view(-1, G3D.shape[-1])
        mask = (G3D.sum(dim=-1).mean(dim=-1) != 0).float()
        density = G3D.sum(dim=-1) # (NX, NY, NZ)

        return G3D, G, mask, density

    def measurement_ray_path(self, st):
        return torch.matmul(st.view(st.shape[0], -1), self.G)

    def forward(self, preds, m_ens=None):
        p_ens, s_ens = preds[:, :, 0], preds[:, :, 1]
        measurements = [self.normalize(self.measurement_ray_path(s_ens[:, step])) for step in self.steps]
        d_obs_sim = torch.cat(measurements, dim=1)
        
        if self.hard_data_locations:
            assert m_ens is not None, "m_ens must be provided if hard_data_locations is specified."
            d_hard_data = self.extract_hard_data(m_ens, self.hard_data_locations)
            return torch.cat((d_obs_sim, d_hard_data), dim=1)
        return d_obs_sim


class CoarseningMeasurementSimulatorModule(BaseMeasurementSimulatorModule):
    def __init__(self, nx, ny, nz, steps, dmin=None, dmax=None, hard_data_locations=None, scale_factor=(1, 1, 1), marginal=None):
        super().__init__(nx, ny, nz, steps, dmin, dmax, hard_data_locations)
        self.scale_factor = scale_factor
        self.marginal = marginal
        self.y_start, self.y_end = (marginal or 0), (ny - marginal if marginal else ny)

    def measurement_coarsen(self, st):
        dx, dy, dz = self.scale_factor
        batch_size, nx, ny, nz = st.shape
        st_cropped = st[:, :, self.y_start:self.y_end, :]
        _, nx_c, ny_c, nz_c = st_cropped.shape
        assert nx_c % dx == 0 and ny_c % dy == 0 and nz_c % dz == 0, "Invalid coarsening dimensions."
        
        reshaped = st_cropped.view(batch_size, nx_c // dx, dx, ny_c // dy, dy, nz_c // dz, dz)
        coarse_st = reshaped.mean(dim=(2, 4, 6))
        return coarse_st.view(coarse_st.shape[0], -1)

    def forward(self, preds, m_ens=None):
        s_ens = preds[:, :, 1]
        measurements = [self.measurement_coarsen(s_ens[:, step].view(-1, self.nx, self.ny, self.nz)) for step in self.steps]
        d_obs_sim = torch.cat(measurements, dim=1)
        
        if self.hard_data_locations:
            assert m_ens is not None, "m_ens must be provided if hard_data_locations is specified."
            d_hard_data = self.extract_hard_data(m_ens, self.hard_data_locations)
            return torch.cat((d_obs_sim, d_hard_data), dim=1)
        return d_obs_sim
