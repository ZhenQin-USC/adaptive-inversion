import numpy as np
import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
import torch.nn.functional as F
import os

from typing import Union 
from os.path import join
from tqdm import tqdm
from typing import List, Tuple
from scipy.optimize import minimize
from .measurementSimulatorTorch import RayPathMeasurementSimulatorModule as RayPathMeasurementSimulator


class Regularization:
    def __init__(self, 
                 z, 
                 device, 
                 mode='L2', 
                 weight=1.0):
        """
        Initialize the Regularization class.

        Args:
            mode (str): Type of regularization. Options are 'L2', 'L1', and 'TV' (Total Variation).
            weight (float): Regularization weight factor.
        """
        self.mode = mode
        self.weight = weight
        if isinstance(z, np.ndarray):
            self.z0 = torch.tensor(z, dtype=torch.float32, requires_grad=False).to(device)
        elif isinstance(z, torch.Tensor):
            self.z0 = z.to(device)
        else:
            raise ValueError("Unsupported input type for z")

    def __call__(self, zm):
        """
        Compute the regularization term.

        Args:
            zm (torch.Tensor): The predicted tensor (shape: [1, 1, nx, ny, nz]).
            z0 (torch.Tensor): The target tensor (shape: [1, 1, nx, ny, nz]).

        Returns:
            torch.Tensor: The computed regularization loss.
        """
        if self.mode == 'L2':
            reg = torch.norm(zm - self.z0, p=2)  # L2 regularization
        elif self.mode == 'L1':
            reg = torch.norm(zm - self.z0, p=1)  # L1 regularization
        elif self.mode == 'TV':
            reg = self.total_variation(zm)  # Total Variation regularization
        else:
            raise ValueError(f"Unsupported regularization mode: {self.mode}")

        return self.weight * reg

    def total_variation(self, x):
        """
        Compute the Total Variation (TV) regularization.

        Args:
            x (torch.Tensor): The tensor to regularize (shape: [1, 1, nx, ny, nz]).

        Returns:
            torch.Tensor: The computed TV regularization term.
        """
        dx = torch.abs(x[..., 1:, :, :] - x[..., :-1, :, :]).sum()
        dy = torch.abs(x[..., :, 1:, :] - x[..., :, :-1, :]).sum()
        dz = torch.abs(x[..., :, :, 1:] - x[..., :, :, :-1]).sum()
        return dx + dy + dz


class AdaptiveMask:
    def __init__(self, z0: torch.Tensor, update_map: torch.Tensor):
        """
        Initialize the AdaptiveMask module.
        
        Args:
            z0 (torch.Tensor): The reference tensor, shape (B, C, X, Y, Z).
            update_map (torch.Tensor): The update mask, shape (1, 1, X, Y, Z), where 1 indicates update and 0 means retain z0.
        """
        super(AdaptiveMask, self).__init__()
        self.z0 = z0
        self.update_map = update_map
        assert update_map.shape == (1, 1, z0.shape[2], z0.shape[3], z0.shape[4]), "update_map shape mismatch"

    def __call__(self, zm: torch.Tensor) -> torch.Tensor:
        """
        Apply an adaptive mask based on the update_map.
        
        Args:
            zm (torch.Tensor): The tensor to be updated, shape (B, C, X, Y, Z).
        
        Returns:
            torch.Tensor: The updated tensor with gradients preserved.
        """
        # Broadcast update_map to match zm's shape
        update_map = self.update_map.expand_as(zm)
        
        # Compute the masked update
        return zm * update_map + self.z0 * (1 - update_map)


class GradientBasedSpatioTemporalProxy:
    def __init__(self, 
                 predictor, 
                 autoencoder, 
                 measurement,
                 selector,
                 d_obs: np.ndarray, 
                 contrl, 
                 states, 
                 device, 
                 batch_size=1, 
                 regularization:Regularization=None,
                 masker:AdaptiveMask=None,
                 ):
        
        self.predictor = predictor.to(device)
        self.autoencoder = autoencoder.to(device)
        self.measurement = measurement.to(device)
        self.selector = selector.to(device)
        self.regularizer = regularization
        self.masker = masker

        self.d_obs = torch.tensor(d_obs, dtype=torch.float32).to(device)       # Convert d_obs from np.ndarray to torch.Tensor
        self.Cd_inv_sqrt = torch.eye(self.d_obs.size(-1)).to(device)           # Precompute inverse sqrt covariance
        
        self.contrl = contrl.to(device)
        self.states = states.to(device)

        self.device = device
        self.batch_size = batch_size

        # Reset/Initialize list to store iterative results
        self.initialize_states()

        # Ensure selected mask is properly initialized
        self.selected_mask = self.selector.selected_mask
        if self.selected_mask is None:
            raise ValueError("selected_mask is not initialized")
        
    def initialize_states(self):
        self.s_posteriors, self.p_posteriors = [], []
        self.m_posteriors, self.d_posteriors = [], []
        self.ensemble, self.latents, self.objective = [], [], []
        self.gradients, self.gradients_3D = [], []
        self.monitor_fval, self.monitor_reg = [], []
    
    def reset_states(self, max_iter_states=20):
        """Keep only the latest `max_iter_states` iterative states to save memory."""
        self.s_posteriors = self.s_posteriors[-max_iter_states:]
        self.p_posteriors = self.p_posteriors[-max_iter_states:]
        self.m_posteriors = self.m_posteriors[-max_iter_states:]
        self.d_posteriors = self.d_posteriors[-max_iter_states:]
        self.ensemble = self.ensemble[-max_iter_states:]
        self.latents = self.latents[-max_iter_states:]
        self.objective = self.objective[-max_iter_states:]
        self.gradients = self.gradients[-max_iter_states:]
        self.gradients_3D = self.gradients_3D[-max_iter_states:]
        self.monitor_fval = self.monitor_fval[-max_iter_states:]
        self.monitor_reg = self.monitor_reg[-max_iter_states:]

    def __call__(self, ensemble: np.ndarray, iteration: int=None) -> np.ndarray:

        # Run the decoder to generate ensemble realizations
        fval, grad, preds, m_ens, d_ens, grads_3d, zm, monitor_val = self._compute_fval_and_gradients(ensemble)
        monitor_fval_val, monitor_reg_val = monitor_val

        # Store the posterior model and observable data
        self.ensemble.append(ensemble)
        self.latents.append(zm.detach().cpu().numpy())
        self.p_posteriors.append(preds.detach().cpu().numpy()[:,:,0])
        self.s_posteriors.append(preds.detach().cpu().numpy()[:,:,1])
        self.m_posteriors.append(m_ens.detach().cpu().numpy())
        self.d_posteriors.append(d_ens.detach().cpu().numpy())
        self.gradients.append(grad.detach().cpu().numpy())
        self.gradients_3D.append(grads_3d.detach().cpu().numpy())
        self.objective.append(fval.detach().cpu().numpy())
        self.monitor_fval.append(monitor_fval_val)
        self.monitor_reg.append(monitor_reg_val)

        return fval.detach().cpu().numpy(), grad.detach().cpu().numpy()
    
    def _compute_fval_and_gradients(self, ensemble):
        """
        Computes the objective function and its gradients, without storing results.

        Args:
            ensemble (np.ndarray): The input ensemble.

        Returns:
            tuple: (objective function value, gradients, predictions, model ensemble, simulated data, 3D gradients).
        """

        selected_latent = torch.tensor(ensemble, dtype=torch.float32, requires_grad=True).to(self.device)
        selected_latent.retain_grad()
        
        if selected_latent.grad is not None:
            selected_latent.grad.zero_()  # ✅ Reset computational graph 

        zm = self._ensemble_to_latents(selected_latent)

        zm = self.masker(zm) if self.masker else zm 

        m_ens = self._decoder(zm)

        preds = self._simulator(m_ens) 

        d_sim = self.measurement(preds)

        f_val = self._objective(self.d_obs, d_sim)
        d_mismatch_val = f_val.item()

        if self.regularizer is None:
            reg_term_val = 0.0
        else:
            reg_term = self.regularizer(zm)
            reg_term_val = reg_term.item()
            f_val += reg_term

        # Compute gradients
        f_val.backward()
        grads = selected_latent.grad.clone()  # ✅ This will now contain the correct gradients
        
        # Convert gradients back to 3D space
        grads_3d = self._ensemble_to_latents(grads)
        
        return f_val, grads, preds, m_ens, d_sim, grads_3d, zm, (d_mismatch_val, reg_term_val)

    def _ensemble_to_latents(self, selected_latent: torch.Tensor) -> torch.Tensor:
        return self.selector.ensemble_to_latent(selected_latent)
    
    def _decoder(self, zm, requires_grad=True):
        if requires_grad:
            return self.autoencoder.decoder([zm])[0]  # ✅ retain computational graph
        else:
            with torch.no_grad():
                return self.autoencoder.decoder([zm])[0]  # ✅ no computational graph

    def _simulator(self, m_ens: torch.Tensor, requires_grad=True):
        """
        Run the simulator, controlling gradient computation based on `requires_grad`:
        - requires_grad=True: Compute gradients, supports backpropagation.
        - requires_grad=False: Disable gradient computation to save GPU memory.
        """

        num_samples = m_ens.shape[0]
        all_preds = []

        # Ensure m_ens allows gradient computation when required
        if requires_grad and not m_ens.requires_grad:
            raise ValueError("m_ens must have requires_grad=True when requires_grad is enabled.")

        # Control whether to use no_grad() to reduce memory usage
        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()

        with grad_context:
            for i in range(0, num_samples, self.batch_size):  # Process each batch
                contrl = torch.repeat_interleave(self.contrl, repeats=self.batch_size, dim=0)
                states = torch.repeat_interleave(self.states, repeats=self.batch_size, dim=0)
                _preds, _ = self.predictor(contrl, states, m_ens[i:i + self.batch_size])
                all_preds.append(_preds)  # Keep tensors, do not convert to numpy

        # Concatenate all batches
        preds = torch.cat(all_preds, dim=0)  

        return preds

    def _objective(self, d_obs, d_sim):
        residual = d_obs - d_sim 
        weighted_residual = torch.matmul(residual, self.Cd_inv_sqrt)
        loss = torch.sum(weighted_residual ** 2)
        return loss

    def calculate_zm(self, m):

        if isinstance(m, torch.Tensor):
            m = m.to(self.device)
        elif isinstance(m, np.ndarray):
            m = torch.tensor(m, dtype=torch.float32).to(self.device)
        else:
            raise ValueError("Only support np.array or torch.Tensor")
        
        with torch.no_grad():
            h = self.autoencoder.encoder(m)
        
        selected_latent = self.selector.select_elements(h, self.selected_mask)

        return selected_latent.detach().cpu().numpy()
    

class GradientBasedProxyOptimizer(GradientBasedSpatioTemporalProxy):
    """
    Gradient-Based Proxy Optimization class that extends GradientBasedSpatioTemporalProxy.
    
    This class adds an optimization framework on top of the existing simulation structure.
    """
    def __init__(self, 
                 predictor, 
                 autoencoder, 
                 measurement, 
                 selector, 
                 d_obs: np.ndarray, 
                 contrl, 
                 states, 
                 device, 
                 batch_size=1, 
                 regularization:Regularization=None,
                 masker:AdaptiveMask=None,
                 ):
        """
        Initializes the optimizer by calling the parent class constructor.

        Args:
            predictor: Predictive model.
            autoencoder: Autoencoder for encoding and decoding latent representations.
            measurement: Measurement model.
            selector: Selector for latent variable sampling.
            d_obs (np.ndarray): Observed data.
            contrl: Control variables.
            states: State variables.
            device: Computational device (CPU/GPU).
            batch_size (int, optional): Batch size for processing. Defaults to 1.
        """
        super().__init__(predictor, autoencoder, measurement, selector, 
                         d_obs, contrl, states, 
                         device, batch_size, regularization, masker)
        
        self.expected_shape = (batch_size, -1)
        
        # Optimization-specific storage
        self.SimulateHistory = {'m': [], 'd': [], 's': [], 'p': [], 'z': [], 'f': [], 'r': []}
        self.OptimizeHistory = {'gradients': [], 'gradients_3D': []}
        self.history_x = []
        self.history_f = []
        
    def savedata(self, x, res):
        """
        Stores optimization history.

        Args:
            x (np.ndarray): Optimization variable.
            res: Optimization result.
        """
        # Store the most recent optimization results
        self.history_x.append(x)  # Optimization variable
        self.history_f.append(self.objective[-1])  # Objective function value

        # Store posterior results (already updated in __call__())
        self.SimulateHistory['m'].append(self.m_posteriors[-1])  # Reconstructed realization, m
        self.SimulateHistory['d'].append(self.d_posteriors[-1])  # Simulated observation, d
        self.SimulateHistory['s'].append(self.s_posteriors[-1])  # Simulated saturation, s
        self.SimulateHistory['p'].append(self.p_posteriors[-1])  # Simulated pressure, p
        self.SimulateHistory['z'].append(self.latents[-1])       # Expanded Latent variable, z
        self.SimulateHistory['f'].append(self.monitor_fval[-1])  # Objective function value, f
        self.SimulateHistory['r'].append(self.monitor_reg[-1])   # Regularization term value, r

        # Store gradient results
        self.OptimizeHistory['gradients'].append(self.gradients[-1])       # Last gradient
        self.OptimizeHistory['gradients_3D'].append(self.gradients_3D[-1]) # Last 3D gradient
        
        # Reset proxy states
        self.reset_states()

    def optimize(self, 
                 x0, bounds=None, constraints=(), 
                 constr_penalty=1.0, 
                 tr_radius=1.0, 
                 barrier_para=0.1, 
                 barrier_tol=0.1, 
                 xtol=1e-6, 
                 gtol=1e-6, 
                 maxiter=1e3):
        """
        Runs optimization using `trust-constr` algorithm.

        Args:
            x0 (1D np.ndarray): Initial optimization variables.
            bounds: Constraints on variables.
            lcon: Linear constraints.
            constr_penalty (float, optional): Constraint penalty parameter. Defaults to 1.0.
            tr_radius (float, optional): Trust region radius. Defaults to 1.0.
            barrier_para (float, optional): Initial barrier parameter. Defaults to 0.1.
            barrier_tol (float, optional): Barrier tolerance. Defaults to 0.1.

        Returns:
            tuple: Optimization result, history of x, history of f, simulation history, and optimization history.
        """

        res = minimize(self.objfun, x0, method='trust-constr', jac=self.gradfun, hess=None,
                       bounds=bounds, constraints=constraints, tol=None,
                       callback=self.savedata,
                       options={'xtol': xtol, 'gtol': gtol,
                                'barrier_tol': 1e-08, 'sparse_jacobian': None,
                                'maxiter': maxiter, 'verbose': 2, 'finite_diff_rel_step': None,
                                'initial_constr_penalty': constr_penalty,
                                'initial_tr_radius': tr_radius,
                                'initial_barrier_parameter': barrier_para,
                                'initial_barrier_tolerance': barrier_tol,
                                'factorization_method': None, 'disp': True})
        return res, self.history_x, self.history_f, self.SimulateHistory, self.OptimizeHistory

    def objfun(self, x):
        """
        Computes the objective function.

        Returns:
            float: Objective function value.
        """
        fval, _ = super().__call__(x.reshape(self.expected_shape))
        return fval

    def gradfun(self, x):
        """
        Computes the gradient.

        Returns:
            np.ndarray: Gradient vector.
        """
        _, grad = super().__call__(x.reshape(self.expected_shape))
        return grad.flatten()


class OptimizationStage:
    def __init__(self, 
                 config, 
                 data_loader, 
                 predictor, 
                 autoencoder, 
                 latent_selector, 
                 device, 
                 opt_config=None, 
                 regularization:Regularization=None,
                 masker:AdaptiveMask=None,
                 ):
        
        self.dir_to_results = None
        self.config = config
        self.data_loader = data_loader
        self.predictor = predictor
        self.autoencoder = autoencoder
        self.latent_selector = latent_selector
        self.device = device
        self.regularization = regularization
        self.masker = masker
        
        if opt_config is None:
            self.opt_config = {'tr_radius': 2.0, 'barrier_para': 0.1, 
                               'barrier_tol': 0.1, 'xtol': 1e-3, 'gtol': 1e-3}
        else:
            self.opt_config = opt_config

        # Measurement
        self.measurement = self.build_measurement()

        # Prepare Data
        self.prepare_data()

        # Latent Mask Selection (which will be stored in selector and won't be called explicitly)
        self.select_latent_mask() # -> (scale_indices, selected_mask, selected_density_map)

        # Proxy Optimizer
        self.proxy_optimizer = self.build_proxy_optimizer()

        # Optimization Results
        self.sim_history = None
        self.opt_history = None

    def build_measurement(self):
        """Build measurement simulator based on config."""
        return self.build_raypath_measurement(self.config)

    def build_raypath_measurement(self, config):
        # Basic setup
        nx                  = config['Basic setup']['nx']                        # e.g., 128, 512, 1
        ny                  = config['Basic setup']['ny']                        # e.g., 128, 512, 1 
        nz                  = config['Basic setup']['nz']                        # e.g., 128, 512, 1
        steps               = config['Basic setup']['steps']                     # e.g., [5, 10]
        dmin                = config['Basic setup']['dmin']                      # e.g., 0, 40
        dmax                = config['Basic setup']['dmax']                      # e.g., 0, 40

        # Measurement setup
        vertical_interval   = config['Measurement setup']['vertical_interval']   # e.g., 8
        horizontal_interval = config['Measurement setup']['horizontal_interval'] # e.g., 32
        marginal            = config['Measurement setup']['marginal']            # e.g., 16
        hard_data_locations = config['Measurement setup']['hard_data_locations'] # e.g., [(zi, 255, 0) for zi in range(0, 128, 8)]
        print("Measurement setup")
        print(f" vertical_interval: {vertical_interval},\n \
        horizontal_interval: {horizontal_interval},\n \
        marginal: {marginal},\n hard_data_locations: {hard_data_locations}, \n steps: {steps}")

        # Build Ray-Path Measurement Simulator
        raypath_measurement = RayPathMeasurementSimulator(
            nx=nx, ny=ny, nz=nz, steps=steps, dmin=dmin, dmax=dmax, 
            vertical_interval=vertical_interval, horizontal_interval=horizontal_interval, 
            hard_data_locations=hard_data_locations, marginal=marginal).to(self.device)
        
        return raypath_measurement

    def prepare_data_for_inverse(self, data_loader, initial_idx, target_idx, measurement, steps, hard_data_locations=None):

        # Training data
        training_set = data_loader['train'].dataset
        train_m, train_s, train_p = training_set.M, training_set.S[:,1:].unsqueeze(dim=2), training_set.P[:,1:].unsqueeze(dim=2)
        train_labels = torch.cat((train_p, train_s), dim=2)
        # Testing data
        testing_set = data_loader['test'].dataset 
        test_m, test_s, test_p = testing_set.M, testing_set.S[:,1:].unsqueeze(dim=2), testing_set.P[:,1:].unsqueeze(dim=2)
        test_labels = torch.cat((test_p, test_s), dim=2)
        # Print
        # print(train_m.shape, train_p.shape, train_s.shape)
        # print(test_m.shape, test_p.shape, test_s.shape)
        # print(train_labels.shape)
        # print(test_labels.shape)

        # Extract contrl and states
        testing_set = data_loader['test'].dataset 
        test_m, test_s, test_p = testing_set.M, testing_set.S[:,1:].unsqueeze(dim=2), testing_set.P[:,1:].unsqueeze(dim=2)
        test_s0, test_p0 = testing_set.S[target_idx,:1].unsqueeze(dim=0), testing_set.P[target_idx,:1].unsqueeze(dim=0)
        states = torch.cat((test_p0, test_s0), dim=1)
        contrl = torch.zeros(test_s[target_idx].unsqueeze(dim=0).size(), dtype=torch.float32)
        contrl[:, :, :, 100, 255, 0] = 1

        # Prepare inputs to ESMDA
        # Realizations
        m_true = test_m[[target_idx]].detach().cpu().numpy()
        m_ens = train_m[[initial_idx]].detach().cpu().numpy()
        s_ens = train_s[[initial_idx], steps, 0].unsqueeze(dim=0).detach().cpu().numpy()
        
        m_ref = m_true
        s_true = test_s[[target_idx],:,0].detach().cpu().numpy()
        p_true = test_p[[target_idx],:,0].detach().cpu().numpy()
        # print(m_true.shape, s_true.shape, p_true.shape)

        # Measurements
        s_ref = test_s[[target_idx], steps, 0,...].unsqueeze(dim=0).detach().cpu().numpy()
        p_ref = test_p[[target_idx], steps, 0,...].unsqueeze(dim=0).detach().cpu().numpy()
        d_ref = torch.cat((test_p, test_s), dim=2)[[target_idx]].detach().cpu().numpy()
        
        d_ref_tensor = torch.tensor(d_ref, dtype=torch.float32).to(self.device)
        m_ref_tensor = torch.tensor(m_ref, dtype=torch.float32).to(self.device)
        
        d_obs = measurement(d_ref_tensor, m_ref_tensor) if hard_data_locations else measurement(d_ref_tensor)
        d_obs = d_obs.detach().cpu().numpy()
        
        # print(m_ref.shape, s_ref.shape, p_ref.shape, d_ref.shape)
        # print(d_obs.shape)

        return (contrl, states), (m_true, s_true, p_true), (m_ens, s_ens), (d_obs, m_ref, s_ref, p_ref)

    def prepare_data(self):
        """Extracts control, states, true values, and measurements."""
        (contrl, states), (m_true, s_true, p_true), (m_ens, s_ens), (d_obs, m_ref, s_ref, p_ref) = self.prepare_data_for_inverse(
            self.data_loader, 
            self.config['Basic setup']['initial_idx'], 
            self.config['Basic setup']['target_idx'], 
            self.measurement, 
            self.config['Basic setup']['steps'], 
            hard_data_locations=self.config['Measurement setup']['hard_data_locations']
        )
        
        self.contrl, self.states = contrl, states
        self.m_true, self.s_true, self.p_true = m_true, s_true, p_true
        self.m_ens, self.s_ens = m_ens, s_ens
        self.d_obs, self.m_ref, self.s_ref, self.p_ref = d_obs, m_ref, s_ref, p_ref

    def select_latent_mask(self, density=None):
        """Calculate density-based latent space mask."""
        selected_grain_ratio = self.config['Basic setup'].get('selected_grain_ratio', [0.0, 0.0, 0.0, 1.0])
        density = self.measurement.density.detach().cpu().numpy() if density is None else density

        self.scale_indices, self.selected_mask, self.selected_density_map = self.latent_selector.generate_mask(
            density, 
            self.config['Basic setup']['selected_scale_index'], 
            selected_grain_ratio
        )

    def build_proxy_optimizer(self):
        """Initialize the proxy optimizer."""
        return GradientBasedProxyOptimizer(
            self.predictor, 
            self.autoencoder, 
            self.measurement, 
            self.latent_selector,
            self.d_obs, 
            self.contrl, 
            self.states, 
            self.device,
            batch_size=1,
            regularization=self.regularization,
            masker=self.masker
        )

    def run_optimization(self, x0):
        """Run optimization with given initial latent representation."""
        x = x0.reshape(-1)
        bounds = [(-10, 10)] * len(x)
        constraints = ()

        res, hx, hf, SimHistory, OptHistory = self.proxy_optimizer.optimize(x, 
                                                                            bounds, 
                                                                            constraints, 
                                                                            **self.opt_config)
        
        # Store results
        self.sim_history = SimHistory
        self.opt_history = OptHistory
        self.opt_history['hx'] = hx
        self.opt_history['hf'] = hf

        return res, hx, hf, SimHistory, OptHistory

    def save_results(self, dir_to_results):
        
        self.dir_to_results = dir_to_results

        """Save optimization results."""
        os.makedirs(dir_to_results, exist_ok=True)

        np.save(join(dir_to_results, "s.npy"), np.vstack(self.sim_history["s"]))
        np.save(join(dir_to_results, "p.npy"), np.vstack(self.sim_history["p"]))
        np.save(join(dir_to_results, "d.npy"), np.vstack(self.sim_history["d"]))
        np.save(join(dir_to_results, "m.npy"), np.vstack(self.sim_history["m"]))
        np.save(join(dir_to_results, "z.npy"), np.vstack(self.sim_history["z"]))
        np.save(join(dir_to_results, "r.npy"), np.vstack(self.sim_history["r"]))
        np.save(join(dir_to_results,"f0.npy"), np.vstack(self.sim_history["f"]))

        np.save(join(dir_to_results, "x.npy"), np.array(self.opt_history["hx"]))
        np.save(join(dir_to_results, "f.npy"), np.array(self.opt_history["hf"]))

        np.save(join(dir_to_results, "g.npy"), np.vstack(self.opt_history["gradients"]))
        np.save(join(dir_to_results, "g3D.npy"), np.vstack(self.opt_history["gradients_3D"]))


