import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_map(img, figsize=(2., 2.), cmap='jet', title=None, fs=10, vmin=None, vmax=None, cbar_ticks=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=500)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1) 
    
    if vmin is None or vmax is None:
        im = ax.imshow(img, cmap=cmap)
    else:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        
    if title:
        ax.set_title(title, fontsize=fs)

    ax.axis('off') 
    
    if cbar_ticks is None:
        cbar = fig.colorbar(im, cax=cax)
    else:
        cbar = fig.colorbar(im, cax=cax, ticks=cbar_ticks)
        
    cbar.ax.tick_params(labelsize=8)
    plt.show()

    
def plot_mask(selected_mask, scale_indices, dpi=400):
    fig, axes = plt.subplots(nrows=1, ncols=len(selected_mask)+1, dpi=dpi)
    axes = axes.flatten()
    for idx, ax in enumerate(axes[:-1]):
        ax.imshow((idx+1)*selected_mask[::-1][idx].detach().cpu().numpy().squeeze(), 
                vmin=0, vmax=len(selected_mask)+1, cmap='Greys')
        ax.set_title(f'Level {idx + 1}', fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])

    axes[-1].imshow(scale_indices.detach().cpu().numpy().squeeze(), 
                    cmap='Greys', vmin=-1, vmax=len(selected_mask))
    axes[-1].set_xticks([]), axes[-1].set_yticks([])
    axes[-1].set_title('Scale Index', fontsize=10)
    plt.show()


def plot_density(selected_density_map, dpi=400):
    fig, axes = plt.subplots(nrows=1, ncols=len(selected_density_map), dpi=dpi)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.imshow(selected_density_map[idx].detach().cpu().numpy().squeeze(), cmap='Greys')
        ax.set_title(f'Level {idx + 1}', fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    plt.show()


def plot_obs(d_obs, d_before, d_after=None, dpi=400):
    """
    Plots observed, initial, and (optional) post-optimization data.

    Args:
        d_obs (array-like): True observation values.
        d_before (array-like): Initial values before optimization.
        d_after (array-like, optional): Values after optimization.
        dpi (int, optional): Resolution of the plot. Default is 300.
    """
    ms = 5  # Marker size
    plt.figure(dpi=dpi)

    # True observation (Red)
    plt.plot(d_obs, 'r-o', label='True', ms=ms, alpha=0.5)

    # Initial values (Black with gray edges)
    plt.plot(d_before, '-o', label='Before', ms=ms, markeredgecolor='gray', color='k', alpha=0.2)

    # Optimized values (Cyan, optional)
    if d_after is not None:
        plt.plot(d_after, '--', ms=ms, label='After', color='cyan')

    # Formatting
    plt.legend()
    plt.grid(lw=0.3, alpha=0.5)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.show()


def plot_real(stage, titles=None, dpi=400):
    """
    Plots the reference, initial, and post-optimization models.

    Args:
        stage: The optimization stage object (should contain `m_ref` and `sim_history`).
        dpi (int, optional): Resolution of the plot. Default is 400.
    """
    titles = ['Reference', 'Initial Parameter', 'Optimized Parameter'] if titles is None else titles
    # Use gridspec to allocate space for colorbars
    fig = plt.figure(figsize=(8, 2), dpi=dpi)
    spec = gridspec.GridSpec(nrows=2, ncols=4, width_ratios=[1, 1, 1, 0.05])  # 预留 colorbar 位置

    axes = np.empty((2, 3), dtype=object)
    
    # Reference model
    axes[0, 0] = fig.add_subplot(spec[0, 0])
    m_ref = stage.m_ref.squeeze()
    im1 = axes[0, 0].imshow(m_ref, cmap='jet', vmin=0, vmax=1)
    axes[0, 0].set_title(titles[0])

    axes[1, 0] = fig.add_subplot(spec[1, 0])
    axes[1, 0].axis('off')

    # Initial model (first iteration)
    axes[0, 1] = fig.add_subplot(spec[0, 1])
    m_init = stage.sim_history['m'][0].squeeze()
    im2 = axes[0, 1].imshow(m_init, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title(titles[1])

    axes[1, 1] = fig.add_subplot(spec[1, 1])
    im3 = axes[1, 1].imshow(m_ref - m_init, cmap='seismic', vmin=-1, vmax=1)

    # Final model (last iteration)
    axes[0, 2] = fig.add_subplot(spec[0, 2])
    m_opt = stage.sim_history['m'][-1].squeeze()
    im4 = axes[0, 2].imshow(m_opt, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title(titles[2])

    axes[1, 2] = fig.add_subplot(spec[1, 2])
    im5 = axes[1, 2].imshow(m_ref - m_opt, cmap='seismic', vmin=-1, vmax=1)

    # Remove ticks for better visualization
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar for the first row (jet colormap)
    cax1 = fig.add_subplot(spec[0, 3])  # Allocated space for the colorbar
    cbar1 = fig.colorbar(im1, cax=cax1, ax=axes[0, :], ticks=[0.0, 0.5, 1.0])

    # Colorbar for the second row (seismic colormap)
    cax2 = fig.add_subplot(spec[1, 3])  # Allocated space for the colorbar
    cbar2 = fig.colorbar(im3, cax=cax2, ax=axes[1, :], ticks=[-1.0, 0.0, 1.0])

    plt.tight_layout()
    plt.show()


def plot_plume(s_ref, s_ini, s_opt, titles, dpi=400):
    """
    Plots the reference, initial, and post-optimization models.

    Args:
        stage: The optimization stage object (should contain `s_ref` and `sim_history`).
        dpi (int, optional): Resolution of the plot. Default is 400.
    """
    
    # Use gridspec to allocate space for colorbars
    fig = plt.figure(figsize=(8, 2), dpi=dpi)
    spec = gridspec.GridSpec(nrows=2, ncols=4, width_ratios=[1, 1, 1, 0.05])  # 预留 colorbar 位置

    axes = np.empty((2, 3), dtype=object)

    # Reference model
    axes[0, 0] = fig.add_subplot(spec[0, 0])
    im1 = axes[0, 0].imshow(s_ref, cmap='jet', vmin=0, vmax=1)
    axes[0, 0].set_title(titles[0])

    axes[1, 0] = fig.add_subplot(spec[1, 0])
    axes[1, 0].axis('off')
    
    # Initial model (first iteration)
    axes[0, 1] = fig.add_subplot(spec[0, 1])
    im2 = axes[0, 1].imshow(s_ini, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title(titles[1])

    axes[1, 1] = fig.add_subplot(spec[1, 1])
    im3 = axes[1, 1].imshow(s_ref - s_ini, cmap='seismic', vmin=-1, vmax=1)

    # Final model (last iteration)
    axes[0, 2] = fig.add_subplot(spec[0, 2])
    im4 = axes[0, 2].imshow(s_opt, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title(titles[2])

    axes[1, 2] = fig.add_subplot(spec[1, 2])
    im5 = axes[1, 2].imshow(s_ref - s_opt, cmap='seismic', vmin=-1, vmax=1)

    # Remove ticks for better visualization
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar for the first row (jet colormap)
    cax1 = fig.add_subplot(spec[0, 3])  # Allocated space for the colorbar
    cbar1 = fig.colorbar(im1, cax=cax1, ax=axes[0, :], ticks=[0.0, 0.5, 1.0])

    # Colorbar for the second row (seismic colormap)
    cax2 = fig.add_subplot(spec[1, 3])  # Allocated space for the colorbar
    cbar2 = fig.colorbar(im3, cax=cax2, ax=axes[1, :], ticks=[-1.0, 0.0, 1.0])

    plt.tight_layout()
    plt.show()


def plot_fval(*stages, logy_scale=False, dpi=400):
    """
    Plots the optimization history for an arbitrary number of stages.

    Args:
        *stages: Variable number of optimization objects.
        logy_scale (bool, optional): If True, use logarithmic scale for y-axis.
        dpi (int, optional): Resolution of the plot. Default is 400.
    """
    line_kwargs = {'ls': '-', 'lw': 2, 'alpha': 0.9}
    colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson', 'purple', 'goldenrod', 'deepskyblue']
    
    # Ensure there are enough colors for the stages
    num_stages = len(stages)
    if num_stages > len(colors):
        colors = cm.get_cmap('tab10', num_stages).colors  # Generate more colors if needed

    # Define iteration numbers
    init_iter = 0
    iter_ranges = []
    f_values = []

    for stage in stages:
        f_values.append(stage.opt_history['hf'])
        iter_range = np.arange(init_iter, init_iter + len(f_values[-1]))
        iter_ranges.append(iter_range)
        init_iter += len(f_values[-1])  # Update starting point for next stage

    # Plot figure
    plt.subplots(figsize=(5, 3), dpi=dpi)

    for i, (iter_range, f_val) in enumerate(zip(iter_ranges, f_values)):
        plt.plot(iter_range, f_val, color=colors[i], label=f"Stage {i+1}", **line_kwargs)

    # Draw vertical dashed lines at the end of each stage except the last one
    for i in range(len(iter_ranges) - 1):
        plt.axvline(x=iter_ranges[i][-1], color='black', linestyle='--', lw=1.5, alpha=0.2)

    # Formatting
    plt.grid(lw=0.2, alpha=0.5)
    
    if logy_scale:
        plt.yscale('log')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Function Value', fontsize=12)
    plt.legend(loc="best", fontsize=10)  # Automatically place legend in best position
    plt.show()

    