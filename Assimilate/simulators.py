import os
import time
import numpy as np
import matplotlib.pyplot as plt
from .measurer import generate_ray_path, ray_path_cross_well, simulate_ray_3d
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from os.path import join


class LinearSimulator: # Mapping from m to d_sim
    def __init__(self, nx, ny, nz, setup: Optional[Dict]=None, data_type='crosswell'):
        self.nx, self.ny, self.nz = nx, ny, nz
        setup = {} if setup is None else setup
        if data_type == 'horizontal':
            self.G3D, self.G, self.density = self.load_horizontal_raypath_matrix(**setup)
        elif data_type == 'crosswell':
            self.G3D, self.G, self.density = self.load_crosswell_raypath_matrix(**setup)
            
    def __call__(self, m, **kwargs):
        # m: nens, spatial_dims
        nens = m.shape[0]
        d = m.reshape(nens, -1) @ self.G
        return d

    def load_horizontal_raypath_matrix(self, **kwargs):
        nd = kwargs.get('nd', 128)
        d_index = np.array(range(0, 2*nd**2, nd)) 
        sim_ray2D = generate_ray_path(self.nx, self.ny, nd) # np.load(path_to_sim)
        sim2D = sim_ray2D.reshape(self.nx, self.ny, sim_ray2D.shape[-1])[..., d_index]

        G3D = np.zeros((self.nx, self.ny, self.nz, self.nz*len(d_index)))
        for i in range(self.nz):
            G3D[:, :, i, i*len(d_index):(i+1)*len(d_index)] = sim2D

        sim = G3D.reshape(-1, G3D.shape[-1])
        density = G3D.sum(axis=-1).mean(axis=-1)
        return G3D, sim, density

    def load_crosswell_raypath_matrix(self, **kwargs):

        nx, ny, nz = self.nx, self.ny, self.nz
        cross_angles = [np.pi*0/4, np.pi*1/4, np.pi*2/4, np.pi*3/4, 
                        np.pi*4/4, np.pi*5/4, np.pi*6/4, np.pi*7/4] if kwargs.get('cross_angles') is None else kwargs.get('cross_angles')
        depths = [0, 4, 9, 14, 19] if kwargs.get('depth') is None else kwargs.get('depth')
        cross_radius = 40 if kwargs.get('cross_radius') is None else kwargs.get('cross_radius')
        centers = [[(nx-cross_radius)//2, (nx-cross_radius)//2], 
                   [(nx+cross_radius)//2, (nx+cross_radius)//2]] if kwargs.get('centers') is None else kwargs.get('centers')
        
        G_collect = []
        for center in centers:
            cross_center_x, cross_center_y = center
            G_current = ray_path_cross_well(nx, ny, nz, 
                                            cross_center_x, cross_center_y, 
                                            cross_radius, 
                                            depths=depths, 
                                            cross_angles=cross_angles)
            G_collect.append(G_current)
        G3D = np.concatenate(G_collect, axis=-1)
        sim = G3D.reshape(-1, G3D.shape[-1])
        density = G3D.sum(axis=-1).mean(axis=-1)
        return G3D, sim, density

    def plot_config(self, save_dir=None, dpi=400):
        """
        Plot the configuration of the 3D grid and save figures if a directory is specified.

        Args:
            save_dir (str, optional): Directory to save the plots. Defaults to None.
        """
        # Plot the 3D grid
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(self.G3D.sum(axis=-1)[:, :, ::-1], facecolors='k', edgecolor=None, alpha=0.5)
        plt.title('3D Grid')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, '3d_grid.png'), bbox_inches='tight')
        plt.show()

        # Plot the top layer mask
        top_layer_mask = self.G3D.sum(axis=-1)[:, :, 0]
        top_layer_mask[top_layer_mask != 0.0] = 1.0
        plt.figure(dpi=dpi)
        plt.imshow(top_layer_mask)
        plt.title('Top Layer Mask')
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'top_layer_mask.png'), bbox_inches='tight')
        plt.show()

        # Plot the mask and normalized density
        norm_density = self.density / self.density.max()
        mask = self.density.copy()
        mask[mask != 0] = 1
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, dpi=dpi)
        axs[0].imshow(mask, cmap='Greys')
        axs[0].set_title('Mask')
        axs[1].imshow(norm_density, cmap='jet', vmin=0, vmax=1)
        axs[1].set_title('Density')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'mask_and_density.png'), bbox_inches='tight')
        plt.show()


class LinearSimulator2: # Mapping from m to d_sim
    def __init__(self, nx, ny, nz, setup: Optional[Dict]):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.sim_config = setup
        self.squeeze_axis = self.sim_config.get("density_axis", -1)
        self.G3D, self.G, self.density = self.generate_raypath_matrix()
            
    def __call__(self, m, **kwargs):
        # m: nens, spatial_dims
        print(m.shape)
        nens = m.shape[0]
        d = m.reshape(nens, -1) @ self.G
        return d
    
    def generate_raypath_matrix(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        G_collect = []
        for pair in self.sim_config['pair']:
            _shooters, _receivers = pair
            ray_path = []
            for shooter in _shooters:
                for receiver in _receivers:
                    ray_path.append(simulate_ray_3d(nx, ny, nz, shooter, receiver)) 
            G_current = np.stack(ray_path, axis=-1)
        G_collect.append(G_current)
        G3D = np.concatenate(G_collect, axis=-1)
        sim = G3D.reshape(-1, G3D.shape[-1])
        density = G3D.sum(axis=-1).mean(axis=self.squeeze_axis)
        return G3D, sim, density

    def plot_config(self, save_dir=None, dpi=400):
        # Plot the mask and normalized density
        norm_density = self.density / self.density.max()
        mask = self.density.copy()
        mask[mask != 0] = 1
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, dpi=dpi)
        axs[0].imshow(mask, cmap='Greys')
        axs[0].set_title('Mask')
        axs[1].imshow(norm_density, cmap='jet', vmin=0, vmax=1)
        axs[1].set_title('Density')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'mask_and_density.png'), bbox_inches='tight')
        plt.show()


