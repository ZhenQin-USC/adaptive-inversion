import numpy as np
from .measurer import simulate_ray_3d


class BaseMeasurementSimulator:
    def __init__(self, nx, ny, nz, steps, dmin=None, dmax=None, hard_data_locations=None):
        """
        Base class for measurement simulators.

        Parameters:
        - nx, ny, nz: Dimensions of the grid (3D space).
        - steps: Time steps for measurement calculation.
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.steps = steps
        self.hard_data_locations = hard_data_locations
        self.dmin, self.dmax = dmin, dmax

    def __call__(self, preds, m_ens=None):
        """
        Generate synthetic measurements from predictions.

        This method must be implemented by subclasses.

        Parameters:
        - preds: Predicted values, assumed to have shape (batch_size, time_steps, spatial_dims).
        - m_ens: Optional additional data for measurement.

        Returns:
        - Measurements as NumPy arrays.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_hard_data(self, m_ens, hard_data_locations):
        """
        Extract hard data values based on the provided well locations.

        Parameters:
        - m_ens: Model ensemble, shape (nsample, 1, NX, NY, NZ).
        - hard_data_locations: List of tuples indicating (x, y, z) well locations.

        Returns:
        - hard_data: Extracted values from m_ens, shape (nsample, nhard).
        """
        # Ensure m_ens has the expected shape
        assert len(m_ens.shape) == 5, "Expected m_ens to have shape (nsample, 1, NX, NY, NZ)."

        # Extract hard data directly based on (x, y, z) indices
        hard_data = []
        for x, y, z in hard_data_locations:
            hard_data.append(m_ens[:, 0, x, y, z])  # Access the (x, y, z) location for all samples

        # Stack the extracted hard data along the last dimension using numpy
        hard_data = np.stack(hard_data, axis=-1)  # Shape: (nsample, nhard)

        return hard_data

    def normalize(self, d_sim, dmin=None, dmax=None):
        self.dmin = dmin if dmin else self.dmin
        self.dmax = dmax if dmax else self.dmax
        if self.dmin is None or self.dmax is None:
            return d_sim
        else:
            return (d_sim - self.dmin) / (self.dmax - self.dmin)
        

class RayPathMeasurementSimulator(BaseMeasurementSimulator):
    def __init__(self, nx, ny, nz, steps, dmin=None, dmax=None, hard_data_locations=None, vertical_interval=8, horizontal_interval=32, marginal=32):
        """
        Ray-path-based measurement simulator.

        Parameters:
        - nx, ny, nz: Dimensions of the grid (3D space).
        - steps: Time steps for measurement calculation.
        - vertical_interval: Spacing between shooters along the vertical axis.
        - horizontal_interval: Spacing between receivers along the horizontal axis.
        - marginal: Margin to avoid edge artifacts when positioning receivers.
        """
        super().__init__(nx, ny, nz, steps, dmin, dmax, hard_data_locations)
        self.vertical_interval = vertical_interval
        self.horizontal_interval = horizontal_interval
        self.marginal = marginal
        self.x_start = marginal - 1
        self.x_end = ny - marginal

        # Simulate ray paths and generate measurement matrix
        self.G3D, self.G, self.mask = self.simulate_ray_paths()

    def simulate_ray_paths(self):
        """
        Simulate ray paths between shooters and receivers to generate the measurement matrix.

        Returns:
        - G3D: 3D measurement matrix.
        - G: Flattened measurement matrix.
        - mask: Binary mask indicating regions of interest.
        """
        # Generate default shooters and receivers
        self.shooters = [
            [depth, self.ny // 2 - 1, 0] for depth in range(0, self.nx, self.vertical_interval)
        ]
        self.receivers = [
            [0, surface_loc, 0]
            for surface_loc in range(self.x_start, self.x_end, self.horizontal_interval)
            if surface_loc != self.ny // 2
        ]

        ray_paths = []
        for shooter in self.shooters:
            for receiver in self.receivers:
                # Simulate the ray path (replace with the actual simulation function)
                ray_path = simulate_ray_3d(self.nx, self.ny, self.nz, shooter, receiver)
                ray_paths.append(ray_path)

        # Stack ray paths into a 3D matrix
        G3D = np.stack(ray_paths, axis=-1)
        G = G3D.reshape(-1, G3D.shape[-1])

        # Generate density and mask
        density = G3D.sum(axis=-1).mean(axis=-1)
        mask = (density != 0).astype(int)

        return G3D, G, mask

    def measurement_ray_path(self, st):
        """
        Perform ray-path-based measurements.

        Parameters:
        - st: Saturation map for time frame t.
        - pt: Pressure map for time frame t.
        - hard_data: Optional additional data.

        Returns:
        - d_obs_sim: Simulated observations.
        """
        d_obs_sim = st.reshape(st.shape[0], -1) @ self.G
        return d_obs_sim

    def __call__(self, preds, m_ens=None):
        """
        Generate synthetic measurements using ray paths.

        Parameters:
        - preds: Predicted values, assumed to have shape (batch_size, time_steps, spatial_dims).
        - m_ens: Optional additional data for measurement.

        Returns:
        - d_obs_sim: Simulated observations as a concatenated NumPy array.
        """
        # Extract p_ens and s_ens components
        p_ens = preds[:, :, 0]
        s_ens = preds[:, :, 1]

        measurements = []
        for step in self.steps:
            st_ens, pt_ens = s_ens[:, step], p_ens[:, step]
            s_obs_sim = self.measurement_ray_path(st_ens)
            measurements.append(self.normalize(s_obs_sim))

        # Concatenate measurements across all time steps
        d_obs_sim = np.concatenate(measurements, axis=1) # (nens, nsoft)

        # Optionally Collect hard data
        if self.hard_data_locations:
            assert m_ens is not None, "m_ens must be provided if hard_data_locations is specified."
            d_hard_data = self.extract_hard_data(m_ens, hard_data_locations=self.hard_data_locations) # (nens, nhard)
            assert d_obs_sim.shape[0] == d_hard_data.shape[0], "Batch size mismatch between soft and hard data."
            return np.concatenate((d_obs_sim, d_hard_data), axis=1) # (nens, nsoft + nhard)
        else: # no hard data
            return d_obs_sim


class CoarseningMeasurementSimulator(BaseMeasurementSimulator):
    def __init__(self, nx, ny, nz, steps, dmin=None, dmax=None, hard_data_locations=None, scale_factor=(1, 1, 1), marginal=None):
        """
        Coarsening-based measurement simulator.

        Parameters:
        - nx, ny, nz: Dimensions of the grid (3D space).
        - steps: Time steps for measurement calculation.
        - scale_factor: Coarsening scale factor (dx, dy, dz).
        """
        super().__init__(nx, ny, nz, steps, dmin, dmax, hard_data_locations)
        self.scale_factor = tuple(scale_factor)
        self.marginal = marginal
        self.y_start = marginal if marginal else 0
        self.y_end = ny - marginal if marginal else ny

        # Ensure y_start and y_end are within bounds
        assert 0 <= self.y_start < ny, f"y_start={self.y_start} is out of bounds!"
        assert 0 < self.y_end <= ny, f"y_end={self.y_end} is out of bounds!"
        assert self.y_start < self.y_end, f"y_start={self.y_start} must be less than y_end={self.y_end}."

    def measurement_coarsen(self, st):
        """
        Perform coarsening-based measurements.

        Parameters:
        - st: Saturation tensor of shape (batch_size, nx, ny, nz).

        Returns:
        - coarse_st: Coarsened tensor of shape based on scale_factor.
        """
        dx, dy, dz = self.scale_factor
        batch_size, nx, ny, nz = st.shape

        # Extract the portion of st along the y-axis
        st_cropped = st[:, :, self.y_start:self.y_end, :]  # Shape: (batch_size, nx, y_end-y_start, nz)

        # Get new dimensions after cropping
        _, self.nx_cropped, self.ny_cropped, self.nz_cropped = st_cropped.shape

        # Validate coarsening divisors
        assert self.nx_cropped % dx == 0, f"nx_cropped={self.nx_cropped} is not divisible by dx={dx}."
        assert self.ny_cropped % dy == 0, f"ny_cropped={self.ny_cropped} is not divisible by dy={dy}."
        assert self.nz_cropped % dz == 0, f"nz_cropped={self.nz_cropped} is not divisible by dz={dz}."
        
        self.nx_fold, self.ny_fold, self.nz_fold = self.nx_cropped // dx, self.ny_cropped // dy, self.nz_cropped // dz

        # Reshape for coarsening
        reshaped = st_cropped.reshape(batch_size, self.nx_fold, dx, self.ny_fold, dy, self.nz_fold, dz)

        # Compute the mean over the coarsened regions
        coarse_st = reshaped.mean(axis=(2, 4, 6))

        return coarse_st.reshape(coarse_st.shape[0], -1)

    def __call__(self, preds, m_ens=None):
        """
        Generate synthetic measurements using coarsening.

        Parameters:
        - preds: Predicted values, assumed to have shape (batch_size, time_steps, spatial_dims).
        - m_ens: Optional additional data for measurement.

        Returns:
        - d_obs_sim: Coarsened observations.
        """
        # Extract saturation component
        s_ens = preds[:, :, 1]

        measurements = []
        for step in self.steps:
            st_ens = s_ens[:, step].reshape(-1, self.nx, self.ny, self.nz)
            s_obs_sim = self.measurement_coarsen(st_ens)
            measurements.append(s_obs_sim)

        # Stack coarsened measurements across time steps
        d_obs_sim = np.concatenate(measurements, axis=1) # (nens, nsoft)

        # Optionally Collect hard data
        if self.hard_data_locations:
            assert m_ens is not None, "m_ens must be provided if hard_data_locations is specified."
            d_hard_data = self.extract_hard_data(m_ens, hard_data_locations=self.hard_data_locations) # (nens, nhard)
            assert d_obs_sim.shape[0] == d_hard_data.shape[0], "Batch size mismatch between soft and hard data."
            return np.concatenate((d_obs_sim, d_hard_data), axis=1) # (nens, nsoft + nhard)
        else: # no hard data
            return d_obs_sim

