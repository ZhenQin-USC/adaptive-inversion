import numpy as np
import matplotlib.pyplot as plt


def plot0(y_true, y_pred, index, layer, 
          tstep0=1, fs=12, figsize=(10, 4.5), vmin=0, vmax=1, error_vmin=-0.2, error_vmax=0.2, 
          aspect=15, shrink=1, dpi=300, cmap='jet', error_cmap='coolwarm', 
          space_adjust={'wspace': None, 'hspace': None}, fname=None):

    time_steps = y_true.shape[1]
    nrows = 3
    ncols = time_steps
    kwargs = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    err_kwargs = {'vmin': error_vmin, 'vmax': error_vmax, 'cmap': error_cmap}

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(**space_adjust)

    for row in range(3):
        for i_tstep in range(time_steps):
            if row == 0:
                im = axs[row,i_tstep].imshow(y_true[index, i_tstep, :, :, layer], **kwargs)
                axs[row,i_tstep].set_title('Year {} '.format(i_tstep+tstep0), fontsize=fs)
                ticks = list(np.linspace(vmin, vmax, 3)) # [0.2, 0.4, 0.6, 0.8]
            elif row == 1:
                im = axs[row,i_tstep].imshow(y_pred[index, i_tstep, :, :, layer], **kwargs)
                ticks = list(np.linspace(vmin, vmax, 3)) # [0.2, 0.4, 0.6, 0.8]
            else: 
                error = y_true[index, i_tstep, :, :, layer] - y_pred[index, i_tstep, :, :, layer]
                im = axs[row,i_tstep].imshow(error, **err_kwargs)
                ticks = [error_vmin, 0.0, error_vmax]
            axs[row,i_tstep].set_xticks([])
            axs[row,i_tstep].set_yticks([])
        fig.colorbar(im, ax=axs[row, :], ticks=ticks, pad=.009, aspect=aspect, shrink=shrink)

    axs[0,0].set_ylabel('True', fontsize=fs)
    axs[1,0].set_ylabel('Pred', fontsize=fs)
    axs[2,0].set_ylabel('Error', fontsize=fs)

    if fname == None:
        pass
    else:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')
     
    plt.show()


def plot1(y_true, y_preds, index, layer, cbar_title=None, err_cbar_title=None, offset=0.01,
          tstep0=0, dtstep=1, fs=12, figsize=(10, 4.5), vmin=0, vmax=1, error_vmin=-0.2, error_vmax=0.2, rotation=0, labelpad=30, 
          colorbar_height=0.02, shrink=1, dpi=300, cmap='jet', error_cmap='coolwarm', title_prefix='Year', time_steps=None,
          space_adjust={'wspace': None, 'hspace': None}, fname=None, ylabels=None, plot_axis=True):
    """
    Plot multiple y_pred along with y_true and error maps, with y_preds as a dictionary.
    Each key in y_preds is used as a label.
    """

    def get_symmetric_cax(fig_width, shrink, height, colorbar_index, 
                          total=2, bottom=0.06, gap=0.05, offset=0.0):
        """
        Returns [left, bottom, width, height] for symmetric horizontal colorbars.
        - colorbar_index: 0 (left), 1 (right)
        - total: total number of colorbars (default = 2)
        """
        bar_width = shrink * (1.0 - gap * (total - 1)) / total
        left = (
            0.5 - (bar_width * total + gap * (total - 1)) / 2
            + colorbar_index * (bar_width + gap) + offset 
            )
        return [left, bottom, bar_width, height]

    if not isinstance(y_preds, dict):
        raise ValueError("y_preds must be a dictionary {label: prediction_array}")

    if cbar_title is None:
        cbar_title = 'Prediction'
    if err_cbar_title is None:
        err_cbar_title = 'Error'

    pred_keys = list(y_preds.keys())
    y_pred_list = list(y_preds.values())
    num_preds = len(y_pred_list)

    time_steps = list(range(tstep0, y_true.shape[1], dtstep)) if time_steps is None else time_steps

    nrows = 1 + 2 * num_preds
    ncols = len(time_steps)

    if ylabels is None:
        ylabels = ['True'] + [k for key in pred_keys for k in (f'{key}', 'Error')]

    kwargs = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap}
    err_kwargs = {'vmin': error_vmin, 'vmax': error_vmax, 'cmap': error_cmap}

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(**space_adjust)

    for row in range(nrows):
        for col, i_tstep in enumerate(time_steps):
            if row == 0:
                im_sat = axs[row, col].imshow(y_true[index, i_tstep, :, :, layer], **kwargs)
                axs[row, col].set_title(f'{title_prefix} {i_tstep+1}', fontsize=fs)
            else:
                pred_idx = (row - 1) // 2
                if (row - 1) % 2 == 0:
                    im = axs[row, col].imshow(y_pred_list[pred_idx][index, i_tstep, :, :, layer], **kwargs)
                else:
                    error = y_true[index, i_tstep, :, :, layer] - y_pred_list[pred_idx][index, i_tstep, :, :, layer]
                    im_err = axs[row, col].imshow(error, **err_kwargs)
                    
            if plot_axis is False:
                axs[row, col].axis("off")
                
            axs[row, col].set_xticks([]), axs[row, col].set_yticks([])

    for i, label in enumerate(ylabels):
        axs[i, 0].set_ylabel(label, fontsize=fs, rotation=rotation, labelpad=labelpad, va='center')

    fig_width_inch = figsize[0]

    cax1 = fig.add_axes(get_symmetric_cax(fig_width_inch, shrink, colorbar_height, colorbar_index=0, offset=offset))
    cax2 = fig.add_axes(get_symmetric_cax(fig_width_inch, shrink, colorbar_height, colorbar_index=1, offset=offset))

    fig.colorbar(im_sat, cax=cax1, orientation='horizontal', ticks=np.linspace(vmin, vmax, 3)).set_label(cbar_title, fontsize=fs)
    fig.colorbar(im_err, cax=cax2, orientation='horizontal', ticks=np.linspace(error_vmin, error_vmax, 3)).set_label(err_cbar_title, fontsize=fs)
    
    if fname is not None:
        plt.savefig(fname, dpi=dpi, bbox_inches='tight')

    plt.show()

