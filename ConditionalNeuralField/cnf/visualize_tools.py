import matplotlib.pyplot as plt
import torch
def plot_fig(pred:torch.Tensor,gt:torch.Tensor,x_coord=None,y_coord=None):
    if x_coord is None:
        
        x_min, x_max = x_coord.min(), x_coord.max()
        y_min, y_max = y_coord.min(), y_coord.max()
        extent = [x_min, x_max, y_min, y_max]
    else:
        extent = None
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    vmin, vmax = gt.min(), gt.max()
    im0 = ax[0].imshow(gt, cmap='jet',vmin=vmin, vmax=vmax,extent=extent,origin='lower')
    im1 = ax[1].imshow(pred, cmap='jet',extent=extent,origin='lower', vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth')
    ax[1].set_title('Prediction')
    abs_err = torch.abs(pred - gt).squeeze().cpu().numpy()
    fig.colorbar(im0, ax=ax[0],fraction=0.035)
    fig.colorbar(im1, ax=ax[1],fraction=0.035)
    im2 = ax[2].imshow(abs_err, cmap='jet',extent=extent,origin='lower')
    fig.colorbar(im2, ax=ax[2],fraction=0.035)
    ax[2].set_title(f'Absolute Error {abs_err.mean():.2e}')
    return fig
