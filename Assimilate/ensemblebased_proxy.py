import numpy as np
import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
from tqdm import tqdm


class BaseSpatioTemporalProxy:
    def __init__(self, predictor, autoencoder, contrl, states, device, batch_size=1, measurement=None):
        self.predictor = predictor.to(device)
        self.autoencoder = autoencoder.to(device)
        self.contrl = contrl.to(device)
        self.states = states.to(device)
        self.device = device
        self.batch_size = batch_size
        self.latent_shape = (1, 32, 32, 1)
        self.measurement = measurement
        self.m_posteriors, self.d_posteriors, self.latents, self.s_posteriors, self.p_posteriors = [], [], [], [], []

    def __call__(self, ensemble: np.ndarray, iteration: int=None):
        
        # Update the latent variable with the ensemble and decode
        zm = self._ensemble_to_latents(ensemble)  # Convert ensemble to latents format

        # Run the decoder to generate ensemble realizations
        m_ens = self._decoder(zm)

        # Run the simulation to generate simulated outputs
        preds = self._simulator(m_ens, iteration=iteration)
        self.p_posteriors.append(preds[:,:,0].copy())
        self.s_posteriors.append(preds[:,:,1].copy())

        if self.measurement is not None:
            d_ens = self.measurement(preds, m_ens.detach().cpu().numpy())
        else:
            d_ens = preds.copy()

        # Store the posterior model and observable data
        self.m_posteriors.append(m_ens.detach().cpu().numpy())
        self.d_posteriors.append(d_ens)
        self.latents.append(ensemble)

        return d_ens

    def _ensemble_to_latents(self, ensemble: np.ndarray) -> torch.tensor:
        nens = ensemble.shape[0]
        latent_shape = (nens, 1, 32, 32, 1)
        latents = torch.tensor(ensemble, dtype=torch.float32).view(latent_shape).to(self.device)
        return latents

    def _simulator(self, proxy_static: torch.Tensor, iteration=None):

        batch_size = self.batch_size

        # Number of samples in the input
        num_samples = proxy_static.shape[0]

        # Container for batch outputs
        all_preds = []

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Predictor"):
                # Slice the current batch

                # Run predictor
                contrl = torch.repeat_interleave(self.contrl, repeats=batch_size, dim=0)
                states = torch.repeat_interleave(self.states, repeats=batch_size, dim=0)
                _preds, _ = self.predictor(contrl, states, proxy_static[i:i + batch_size])
                
                # Move batch outputs to CPU and store them
                all_preds.append(_preds.detach().cpu().numpy())

        # Concatenate all batch results along the first dimension (batch dimension)
        preds = np.concatenate(all_preds, axis=0)
        return preds

    def _decoder(self, zm):
        with torch.no_grad():
            m_ens = self.autoencoder.decoder([zm])[0].detach()
        return m_ens

    def calculate_zm(self, m):
        nsample = m.shape[0]

        if isinstance(m, torch.Tensor):
            m = m.to(self.device)
        elif isinstance(m, np.ndarray):
            m = torch.tensor(m, dtype=torch.float32).to(self.device)
        else:
            raise ValueError("Only support np.array or torch.tensor")
        
        with torch.no_grad():
            h = self.autoencoder.encoder(m)
            h_grains, _, _ = self.autoencoder.masker(h)
        zm = h_grains[-1] # 0: Mixed, 1: Finest, -1: Coarsest
        
        return zm.detach().cpu().numpy().reshape(nsample, -1)
    
