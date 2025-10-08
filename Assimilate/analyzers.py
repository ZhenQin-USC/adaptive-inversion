import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend_handler import HandlerLine2D

save_fig = True # False # 


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
        _m = selected_mask[::-1][idx].detach().cpu().numpy().squeeze(0).squeeze(0)[..., 0]
        ax.imshow((idx+1)*_m, 
                vmin=0, vmax=len(selected_mask)+1, cmap='Greys')
        ax.set_title(f'Level {idx + 1}', fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    axes[-1].imshow(scale_indices.detach().cpu().numpy()[..., 0], 
                    cmap='Greys', vmin=-1, vmax=len(selected_mask))
    axes[-1].set_xticks([]), axes[-1].set_yticks([])
    axes[-1].set_title('Scale Index', fontsize=10)
    plt.show()


def plot_density(selected_density_map, dpi=400):
    fig, axes = plt.subplots(nrows=1, ncols=len(selected_density_map), dpi=dpi)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        ax.imshow(selected_density_map[idx].detach().cpu().numpy().squeeze(0).squeeze(0)[...,0], 
                  cmap='Greys')
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

    
def plot_iterative_updates(posteriors, x_ref, x_prior, num_cases=6, num_iters=5, layer=0, dpi=400,
                           figsize=None, plot_error=False, show_layer=False, save_dir=None):
    cmap = 'jet'
    
    multiplier = 2 if plot_error else 1
    nrows, ncols = num_cases * multiplier, num_iters + 2
    image_shape =  np.squeeze(x_ref).shape[:2] 
    aspect_ratio = image_shape[1] / image_shape[0]  # Assuming x_ref is in (height, width, channels) format
    figsize=(ncols * aspect_ratio, nrows) if figsize is None else figsize
    plt.figure(figsize=figsize, dpi=dpi)
    plt.subplot(nrows, ncols, 1)
    plt.grid(False)
    plt.xticks([]), plt.yticks([])
    plt.imshow(np.squeeze(x_ref[..., layer]), cmap=cmap, vmin=0, vmax=1)
    plt.title("Ref")
    if show_layer is True: 
        plt.ylabel(f"Layer {layer+1}") 

    for count, i in enumerate(range(num_cases)):

        plt.subplot(nrows, ncols, (ncols * (count * multiplier)) + 2)
        plt.grid(False)
        plt.xticks([]), plt.yticks([])
        plt.imshow(np.squeeze(x_prior[i, ..., layer]), cmap=cmap, vmin=0, vmax=1)
        if count == 0:
            plt.title("Prior")
        
        for itr, img_itr in enumerate(posteriors[1:]):
            plt.subplot(nrows, ncols, (ncols * (count * multiplier)) + 3 + itr)
            plt.grid(False)
            plt.xticks([]), plt.yticks([])
            plt.imshow(np.squeeze(img_itr[i, ..., layer]), cmap=cmap, vmin=0, vmax=1)
            if count == 0:
                plt.title(f"Itr-{itr+1}")
            if plot_error:
                # Display the difference image in the row below
                plt.subplot(nrows, ncols, (ncols * (count * multiplier + 1)) + 3 + itr)
                plt.grid(False)
                plt.xticks([]), plt.yticks([])
                difference_image = np.squeeze(img_itr[i, ..., layer]) - np.squeeze(x_ref[..., layer])
                plt.imshow(difference_image, cmap='seismic', vmin=-1, vmax=1)

    plt.tight_layout()
    # Save the figure if a directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'iterative_updates_layer_{layer}.png'))

    plt.show()


def plot_comparisons(m_ref, m_prior, m_posteriors, layer=0, figsize=None, 
                     val_range=None, dif_range=None,
                     var_range_prior=None, var_range_post=None, 
                     pad=.075, aspect=6, shrink=1.0, dpi=100, save_dir=None):
    
    image_shape =  np.squeeze(m_ref).shape[:2] 
    aspect_ratio = image_shape[1] / image_shape[0]  # Assuming x_ref is in (height, width, channels) format
    figsize = (10*aspect_ratio, 3) if figsize is None else figsize

    val_range = [0.0, 1.0] if val_range is None else val_range
    dif_range = [-0.5, 0.5] if dif_range is None else dif_range
    var_range_prior = [0.0, 1.0] if var_range_prior is None else var_range_prior
    var_range_post = [0.0, 1.0] if var_range_post is None else var_range_post

    val_ticks = [val_range[0], (val_range[0] + val_range[1]) / 2, val_range[1]]
    dif_ticks = [dif_range[0], (dif_range[0] + dif_range[1]) / 2, dif_range[1]]
    err_ticks_prior = [var_range_prior[0], (var_range_prior[0] + var_range_prior[1]) / 2, var_range_prior[1]]
    err_ticks_post = [var_range_post[0], (var_range_post[0] + var_range_post[1]) / 2, var_range_post[1]]

    ref = np.squeeze(m_ref[..., layer])
    prior_mean, prior_var = np.mean(np.squeeze(m_prior[..., layer]), axis=0), np.var(np.squeeze(m_prior[..., layer]), axis=0)
    post_mean, post_var = np.mean(np.squeeze(m_posteriors[..., layer]), axis=0), np.var(np.squeeze(m_posteriors[..., layer]), axis=0)
    diff_prior, diff_post = ref - prior_mean, ref - post_mean
    ncols, nrows = 7, 2
    gs_kw = dict(width_ratios=[1., 0.15, 1., 1., 0.15, 1., 1.], height_ratios=[1, 1])
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, gridspec_kw=gs_kw, sharex=True, sharey=True,
                            constrained_layout=True, dpi=dpi)

    im0 = axs[0, 0].imshow(ref, cmap='jet', vmin=val_range[0], vmax=val_range[1], aspect='equal')
    axs[0, 0].set_xticks([]), axs[0, 0].set_yticks([])
    axs[0, 0].set_title(f"Layer {layer + 1}")
    fig.colorbar(im0, ticks=val_ticks, pad=pad, aspect=aspect, shrink=shrink)

    im1 = axs[0, 2].imshow(prior_mean, cmap='jet', vmin=val_range[0], vmax=val_range[1], aspect='equal')
    axs[0, 2].set_title("$\mu_{prior}$")
    fig.colorbar(im1, ticks=val_ticks, pad=pad, aspect=aspect, shrink=shrink)

    im2 = axs[0, 3].imshow(prior_var, cmap='jet', vmin=var_range_prior[0], vmax=var_range_prior[1], aspect='equal')
    axs[0, 3].set_title("$\sigma_{prior}$")
    fig.colorbar(im2, ticks=err_ticks_prior, pad=pad, aspect=aspect, shrink=shrink)

    im3 = axs[0, 5].imshow(post_mean, cmap='jet', vmin=val_range[0], vmax=val_range[1], aspect='equal')
    axs[0, 5].set_title("$\mu_{posterior}$")
    fig.colorbar(im3, ticks=val_ticks, pad=pad, aspect=aspect, shrink=shrink)

    im4 = axs[0, 6].imshow(post_var, cmap='jet', vmin=var_range_post[0], vmax=var_range_post[1], aspect='equal')
    axs[0, 6].set_title("$\sigma_{posterior}$")
    fig.colorbar(im4, ticks=err_ticks_post, pad=pad, aspect=aspect, shrink=shrink)

    im5 = axs[1, 2].imshow(diff_prior, cmap='seismic', vmin=dif_range[0], vmax=dif_range[1], aspect='equal')
    fig.colorbar(im5, ticks=dif_ticks, pad=pad, aspect=aspect, shrink=shrink)
    axs[1, 2].set_title("$\delta_{prior}$")

    im6 = axs[1, 5].imshow(diff_post, cmap='seismic', vmin=dif_range[0], vmax=dif_range[1], aspect='equal')
    fig.colorbar(im6, ticks=dif_ticks, pad=pad, aspect=aspect, shrink=shrink)
    axs[1, 5].set_title("$\delta_{posterior}$")

    axs[1, 0].axis('off'), axs[1, 3].axis('off'), axs[1, 6].axis('off')
    axs[0, 1].axis('off'), axs[0, 4].axis('off')
    axs[1, 1].axis('off'), axs[1, 4].axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'comparison_layer_{layer}.png'), bbox_inches='tight')

    plt.show()


def plot_discrete_matshow(data, vmin=0, vmax=5, figsize=None, tick_font_size=14, cmap_str='RdBu', save_dir=None):
    # Set default figure size if not provided
    figsize = (5, 5) if figsize is None else figsize

    # Create a discrete colormap with appropriate levels
    cmap = plt.get_cmap(cmap_str, vmax - vmin)
    ticks = np.arange(vmin, vmax)
    ticklabels = np.arange(vmin + 1, vmax + 1)
    print(f"Ticks: {ticks}, Tick Labels: {ticklabels}")

    # Create a figure and axis with the specified size and resolution
    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    
    # Display the data as an image
    mat = ax.imshow(data, cmap=cmap, vmin=vmin - 0.5, vmax=vmax - 0.5)

    # Use make_axes_locatable to create an axis for the colorbar that aligns with the figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create colorbar with explicit ticks
    cbar = plt.colorbar(mat, cax=cax, ticks=ticks, spacing='proportional')
    cbar.ax.tick_params(labelsize=tick_font_size)
    cbar.set_ticklabels(ticklabels)  # Ensure the tick labels are set
    
    # Remove x and y ticks from the axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure if a save directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'discrete_matshow.png'), bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_selected_masks(selected_masks, scale=10, dpi=500, save_dir=None, orientation='vertical'):
    """
    Plots a series of selected masks using matplotlib.

    Parameters:
    - selected_masks (list): A list of mask tensors to be visualized.
    - scale (float): Scaling factor for the figure size.
    - dpi (int): Dots per inch for the figure resolution.
    - save_dir (str): Directory to save the plotted figure.
    - orientation (str): 'vertical' or 'horizontal' orientation of the plots.
    """
    if orientation == 'vertical':
        nrows = len(selected_masks)
        ncols = 1
        figsize = (ncols * scale, nrows * scale)
    else:
        nrows = 1
        ncols = len(selected_masks)
        figsize = (ncols * scale, nrows * scale)
        
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axs = np.array([axs]).reshape(nrows, ncols)
    
    for idx, ax in enumerate(axs.flatten()):
        _mask = selected_masks[idx][:, :, 0]
        ax.pcolor(_mask.squeeze().detach().cpu().numpy()[::-1].astype(int), cmap='Greys', 
                  edgecolors='w', linewidths=0.25, vmin=0, vmax=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=0)
        ax.set_aspect('equal')
        ax.set_title(f"{int(_mask.sum())} Parameters", fontsize=14)

    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'selected_masks.png'), bbox_inches='tight')

    plt.show()


def plotmvar(data1, data2, layer=0, name='', pad=.075, aspect=6, shrink=1.0, err_range=None, val_range=None, show_layer=False, save_dir=None):
    err_range = [0.0, 0.02] if err_range is None else err_range
    val_range = [0.0, 1.0] if val_range is None else val_range

    val_ticks = [val_range[0], (val_range[0] + val_range[1]) / 2, val_range[1]]
    err_ticks = [err_range[0], (err_range[0] + err_range[1]) / 2, err_range[1]]

    fig = plt.figure(figsize=(3, 6), dpi=400)

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(np.squeeze(data1[..., layer]), cmap='jet', vmin=val_range[0], vmax=val_range[1], aspect='equal')
    plt.xticks([]), plt.yticks([])
    plt.title(name + "_$\mu$")
    fig.colorbar(im1, ticks=val_ticks, pad=pad, aspect=aspect, shrink=shrink)
    if show_layer is True:
        plt.ylabel(f"Layer {layer + 1}")

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(np.squeeze(data2[..., layer]), cmap='seismic', vmin=err_range[0], vmax=err_range[1])
    plt.xticks([]), plt.yticks([])
    plt.title(name + "_$\sigma$")
    plt.tight_layout()
    fig.colorbar(im2, ticks=err_ticks, pad=pad, aspect=aspect, shrink=shrink)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'mvar_layer_{layer}_{name}.png'), bbox_inches='tight')

    plt.show()


def plot_data_posteriors(d_posteriors, d_ref, d_prior, alpha=0.01, xmin=None, xmax=None, titles=None, save_dir=None):
    titles = ["", ""] if titles is None else titles
    title1, title2 = titles
    nens = d_prior.shape[0]
    d_posteriors = d_posteriors[-1]
    xmin = min(d_prior.min(), d_ref.min(), d_posteriors.min()) if xmin is None else xmin
    xmax = max(d_prior.max(), d_ref.max(), d_posteriors.max()) if xmax is None else xmax
    fig = plt.figure(figsize=(8, 3.5), dpi=400)
    ax = fig.add_subplot(1, 2, 1)
    plt.scatter(d_ref.flatten(), d_ref.flatten(), s=10, color='red', alpha=1, label="$d_{obs}$", zorder=20)
    for i in range(nens):
        plt.scatter(d_ref.flatten(), d_prior[i, :].flatten(), s=30, color='gray', alpha=alpha)
    for i in range(nens):
        plt.scatter(d_ref.flatten(), d_posteriors[i, :].flatten(), s=30, color='orange', alpha=alpha)
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel("$d_{obs}$")
    plt.ylabel("$d_{obs}$" + "+$\epsilon$")
    plt.legend()
    plt.title(title1)

    ax = fig.add_subplot(1, 2, 2)
    timesteps = np.linspace(0, d_ref.shape[0] - 1, d_ref.shape[0])
    plt.plot(timesteps, d_ref, ls=':', c='k', alpha=1, label="$d_{obs}$", zorder=20)
    for i in range(nens):
        plt.plot(timesteps, d_prior[i, :], color='gray', alpha=alpha)
    for i in range(nens):
        plt.plot(timesteps, d_posteriors[i, :], color='orange', alpha=alpha)
    plt.ylim([xmin, xmax])
    plt.title(title2)
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'data_posteriors.png'), bbox_inches='tight')

    plt.show()


def plot_recons(test_recon, test_true, figure_settings=None, titles=None, save_dir=None):
    nrows = test_recon.shape[1] + 2
    if figure_settings is None:
        ncols = 10
        dpi = 400
        figure_settings = {
            'nrows': nrows,
            'ncols': ncols,
            'figsize': (ncols, nrows),
            'dpi': dpi,
        }

    fig, axs = plt.subplots(**figure_settings)
    for i in range(figure_settings['ncols']):
        axs[0, i].imshow(test_true[i, 0, :, :, 0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
        if titles is not None:
            axs[0, i].set_title(titles[i])
        for j in range(test_recon.shape[1]):
            axs[j + 1, i].imshow(test_recon[i, j, 0, :, :, 0].detach().cpu().numpy(), vmin=0, vmax=1.0, cmap='jet')
        error = test_recon[i, -1, 0, :, :, 0].detach().cpu().numpy() - test_true[i, 0, :, :, 0].detach().cpu().numpy()
        axs[-1, i].imshow(error, vmin=-0.2, vmax=0.2, cmap='seismic')

        for j in range(nrows):
            axs[j, i].set_xticks([]), axs[j, i].set_yticks([])

    axs[0, 0].set_ylabel('True', fontsize=10)
    for j in range(1, test_recon.shape[1] + 1):
        axs[j, 0].set_ylabel('Level {}'.format(j), fontsize=10)
    axs[-1, 0].set_ylabel('Error', fontsize=10)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'reconstructions.png'), bbox_inches='tight')

    plt.show()


def plot_hists(loss_list_collect, figure_settings=None, bin_settings=None, save_dir=None):
    if bin_settings is None:
        bin_settings = {'bins': 30, 'histtype': 'stepfilled'}

    if figure_settings is None:
        figure_settings = {'figsize': (6, 3), 'dpi': 200}

    plt.figure(**figure_settings)

    colors = ['skyblue', 'salmon', 'lightgreen', 'gold']
    edge_color = 'black'

    for i, loss_array in enumerate(loss_list_collect):
        plt.hist(loss_array, **bin_settings, color=colors[i % len(colors)],
                 edgecolor=edge_color, alpha=0.75, label=f'Level {i + 1}')

    plt.legend(title='Scales')
    plt.title('Distributions of Test Errors')
    plt.xlabel('RMSE Values')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'histograms.png'), bbox_inches='tight')

    plt.show()


def plot0(y_true, y_pred, index, layer, 
          tstep0=0, dtstep=1, fs=12, figsize=(10, 4.5), vmin=0, vmax=1, error_vmin=-0.2, error_vmax=0.2, 
          aspect=15, shrink=1, dpi=300, cmap='jet', error_cmap='coolwarm', title_prefix='Year',
          space_adjust={'wspace': None, 'hspace': None}, fname=None, ylabels=None):
    ylabels = ['True', 'Pred', 'Error'] if ylabels is None else ylabels
    label1, label2, label3 = ylabels
    time_steps = list(range(tstep0, y_true.shape[1], dtstep)) # y_true.shape[1]
    print(time_steps)
    nrows = 3
    ncols = len(time_steps)
    kwargs = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    err_kwargs = {'vmin': error_vmin, 'vmax': error_vmax, 'cmap': error_cmap}
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(**space_adjust)
    for row in range(3):
        for col, i_tstep in enumerate(time_steps):
            if row == 0:
                im = axs[row,col].imshow(y_true[index, i_tstep, :, :, layer], **kwargs)
                axs[row,col].set_title(f'{title_prefix} {i_tstep+1} ', fontsize=fs)
                ticks = list(np.linspace(vmin, vmax, 3)) # [0.2, 0.4, 0.6, 0.8]
            elif row == 1:
                im = axs[row,col].imshow(y_pred[index, i_tstep, :, :, layer], **kwargs)
                ticks = list(np.linspace(vmin, vmax, 3)) # [0.2, 0.4, 0.6, 0.8]
            else: 
                error = y_true[index, i_tstep, :, :, layer] - y_pred[index, i_tstep, :, :, layer]
                im = axs[row,col].imshow(error, **err_kwargs)
                ticks = [error_vmin, 0.0, error_vmax]
            axs[row,col].set_xticks([])
            axs[row,col].set_yticks([])
        fig.colorbar(im, ax=axs[row, :], ticks=ticks, pad=.009, aspect=aspect, shrink=shrink)
    axs[0,0].set_ylabel(label1, fontsize=fs)
    axs[1,0].set_ylabel(label2, fontsize=fs)
    axs[2,0].set_ylabel(label3, fontsize=fs)

    if fname == None:
        pass
    else:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
     
    plt.show()


def plot_mean_and_CI(mean, lb, ub, steps, lw=2, alpha=.25, alpha2=1.0, color_mean=None, 
                     markeredgecolor='auto', dashed=False, marker=None, color_shading=None, ms=8):
    """
    Plots the mean with confidence intervals (CI) as shaded regions.
    
    :param mean: Mean values
    :param lb: Lower bound of the CI
    :param ub: Upper bound of the CI
    :param steps: X-axis values
    :param lw: Line width
    :param alpha: Transparency for shaded region
    :param color_mean: Color of the mean line
    :param dashed: If True, use dashed lines ('--')
    :param marker: Marker style ('dot', 'circle', 'triangle', 'square', 'inverted_triangle')
    :param color_shading: Color for the shaded CI region
    :param ms: Marker size
    """
    # Define line style
    ls = '--' if dashed else '-'
    
    # Define marker styles
    marker_styles = {
        'dot': '.',         # Small dot
        'circle': 'o',      # Open circle
        'triangle': '^',    # Upward triangle
        'square': 's',      # Square
        'inverted_triangle': 'v'  # Downward triangle
    }
    
    # Get the corresponding marker symbol
    marker_symbol = marker_styles.get(marker, None)
    
    # Plot the shaded confidence interval
    plt.fill_between(steps, ub, lb, color=color_shading, alpha=alpha, edgecolor=None)
    
    # Plot the mean line with the appropriate marker
    plt.plot(steps, mean, color=color_mean, marker=marker_symbol, markeredgecolor=markeredgecolor, ms=ms, ls=ls, lw=lw, alpha=alpha2)


class LegendObject:
    def __init__(self, facecolor='red', edgecolor='white', dashed=False, marker=None, 
                 markeredgecolor='none', lw=1):
        """
        Custom legend handler with optional dashed or marker-based patterns.
        
        :param facecolor: Main fill color
        :param edgecolor: Border color
        :param dashed: If True, adds a dash inside the legend entry
        :param marker: Marker style ('dot', 'circle', 'triangle', 'square', 'inverted_triangle')
        """
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
        self.marker = marker
        self.markeredgecolor = markeredgecolor
        self.lw = lw
        
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        # Create the main background rectangle
        patch = mpatches.Rectangle(
            [x0, y0], width, height, facecolor=self.facecolor,  # create a rectangle that is filled with color
            edgecolor=self.edgecolor, lw=4)                     # and whose edges are the faded color
        handlebox.add_artist(patch)

        # Manually add the dash in to the rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
            
        # Add marker in the center of the legend entry
        if self.marker:
            marker_x = x0 + width / 2  # Center of the width
            marker_y = y0 + height / 2 # Center of the height
            marker_radius = height / 2  # Adjust marker size

            if self.marker == 'dot':
                marker_patch = mpatches.Circle((marker_x, marker_y), radius=marker_radius, lw=self.lw,
                                               facecolor=self.facecolor, edgecolor=self.markeredgecolor, transform=handlebox.get_transform())
            elif self.marker == 'circle':
                marker_patch = mpatches.Circle((marker_x, marker_y), radius=marker_radius, 
                                               facecolor=self.facecolor, edgecolor=self.markeredgecolor, lw=self.lw, transform=handlebox.get_transform())
            elif self.marker == 'triangle':
                marker_patch = mpatches.RegularPolygon((marker_x, marker_y), numVertices=3, radius=marker_radius*1.2, lw=self.lw,
                                                       facecolor=self.facecolor, edgecolor=self.markeredgecolor, transform=handlebox.get_transform())
            elif self.marker == 'square':
                marker_patch = mpatches.Rectangle((marker_x - marker_radius, marker_y - marker_radius),
                                                  2 * marker_radius, 2 * marker_radius, lw=self.lw,
                                                  facecolor=self.facecolor, edgecolor=self.markeredgecolor, transform=handlebox.get_transform())
            elif self.marker == 'inverted_triangle':
                marker_patch = mpatches.RegularPolygon((marker_x, marker_y), numVertices=3, radius=marker_radius*1.2, lw=self.lw,
                                                       orientation=np.pi, facecolor=self.facecolor, 
                                                       edgecolor=self.markeredgecolor, 
                                                       transform=handlebox.get_transform())
            else:
                marker_patch = None  # No marker

            if marker_patch:
                handlebox.add_artist(marker_patch)

        return patch


class Line2DLegendObject(HandlerLine2D):
    def __init__(self, color, linestyle='--', marker=None, lw=2):
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.lw = lw
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = mlines.Line2D([xdescent + width / 2.0, xdescent + width], 
                             [ydescent + height / 2.0, ydescent + height / 2.0],
                             linestyle=self.linestyle, color=self.color, lw=self.lw, marker=self.marker,
                             transform=trans)
        return [line]

