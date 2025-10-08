import numpy as np
from os.path import join
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from .functions import sample_mvnormal


class ESMDA:
    def __init__(self, d_obs, X, proxy, d_std=0.01, seed=169, **kwargs):
        self.d_obs = d_obs  # True measurements with shape (1, nd)
        self.X = X  # Prior state ensemble with shape (nens, nz)
        self.proxy = proxy  # Proxy model function
        self.d_std = d_std  # Standard deviation of observation error
        self.iter = 0  # Initialize iteration counter
        self.X_list, self.Y_list = [], []  # Lists to store ensemble X and Y over iterations
        self.CDD_list, self.CMD_list, self.K_list = [], [], []
        self.nens, self.nd = X.shape[0], d_obs.shape[1]
        self.rng = np.random.default_rng(seed)
        self.dir_to_project = kwargs.get('dir_to_project', None)

    def initialize_esmda(self):
        # Compute the observational error covariance matrix R and CD (diagonal of R)
        self.D = np.tile(self.d_obs, (self.X.shape[0], 1))  # Repeat d_obs for each ensemble member
        E = self.d_std * np.random.normal(0, 1, self.D.shape) # Error term
        R = np.cov(E.T)  # Covariance of errors
        self.CD = np.diag(R)  # Diagonal of R for observational error covariance

    def perturb_obs(self, alpha):
        noise = sample_mvnormal(C_d=self.CD, rng=self.rng, size=self.nens)
        D = self.d_obs + np.sqrt(alpha) * noise
        assert D.shape == (self.nens, self.nd), f"Expected shape {(self.nens, self.nd)}, got {D.shape}"
        return D

    def assimilate(self, alpha_list):
        # Initialize ESMDA-related parameters
        self.initialize_esmda()

        # Initialize lists to store ensemble states and simulated data over iterations
        self.Y = self.proxy(self.X, iteration=0)
        self.X_list, self.Y_list = [self.X.copy()], [self.Y.copy()]
        self.save_state(iteration=0)

        for idx, alpha in enumerate(alpha_list):
            print(f' --------------------- Iteration {idx+1} --------------------- ')
            D_obs = self.perturb_obs(alpha)
            self.X = self.X + self.update(alpha=alpha, D=D_obs, X=self.X, Y=self.Y)

            # Update the simulated data ensemble Y using the proxy model
            self.Y = self.proxy(self.X, iteration=idx+1)

            # Store the updated ensembles and covariances
            self.X_list.append(self.X.copy())
            self.Y_list.append(self.Y.copy())

            # Save the state to dir_to_project
            self.save_state(iteration=idx+1)

        return self.X_list, self.Y_list

    def update(self, alpha, D, X, Y):
        # Compute covariance matrices CDD and CMD
        CDD = np.cov(Y.T)  # Covariance of simulated data ensemble (nd, nd)
        DA = X - np.mean(X, axis=0)  # Anomalies of state ensemble (nens, nz)
        DF = Y - np.mean(Y, axis=0)  # Anomalies of simulated data ensemble (nens, nd)
        CMD = DA.T @ DF / (X.shape[0] - 1)  # Cross-covariance between state and data (nz, nd)

        # Compute the Kalman gain K
        K = CMD @ np.linalg.pinv(CDD + alpha * np.diag(self.CD))  # Kalman gain (nz, nd)

        # Use Kalman gain to compute DX based on DY
        DY = (D - Y).T
        DX = (K @ DY).T

        self.CDD_list.append(CDD)
        self.CMD_list.append(CMD)
        self.K_list.append(K)

        # Debugging and information output
        print(f'Shape of CDD: {CDD.shape}')
        print(f'Shape of DA and DF: {DA.shape} and {DF.shape}')
        print(f'Shape of CMD and K: {CMD.shape} and {K.shape}')
        print(f'Shape of CD, DY, and DX: {self.CD.shape}, {DY.shape}, and {DX.shape}')

        print('CDD: min={}, max={}'.format(CDD.min(), CDD.max()))
        print('DA: min={}, max={}'.format(DA.min(), DA.max()))
        print('DF: min={}, max={}'.format(DF.min(), DF.max()))
        print('CMD: min={}, max={}'.format(CMD.min(), CMD.max()))
        print('K: min={}, max={}'.format(K.min(), K.max()))
        print('CD: min={}, max={}'.format(self.CD.min(), self.CD.max()))
        print('DY: min={}, max={}'.format(DY.min(), DY.max()))
        print('DX: min={}, max={}'.format(DX.min(), DX.max()))
        print('X: min={}, max={}'.format(X.min(), X.max()))
        print('Y: min={}, max={}'.format(Y.min(), Y.max()))
        print('D: min={}, max={}'.format(D.min(), D.max()))

        return DX

    def save_state(self, iteration=0):
        if self.dir_to_project is not None:
            np.save(join(self.dir_to_project, f"X_iter{iteration}.npy"), self.X)
            np.save(join(self.dir_to_project, f"Y_iter{iteration}.npy"), self.Y)
