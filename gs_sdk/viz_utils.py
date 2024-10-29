import matplotlib.pyplot as plt
import numpy as np


def plot_gradients(fig, ax, gx, gy, mask=None, mode="rgb", **kwargs):
    """
    Plot the gradients.

    :params fig: plt.figure; the figure to plot the gradients.
    :params ax: plt.axis; the axis to plot the gradients.
    :params gx: np.array (H, W); the x gradient.
    :params gy: np.array (H, W); the y gradient.
    :params mask: np.array (H, W); the mask for gradients to be plotted
    :params mode: str {"rgb", "quiver"}; the mode to plot the gradients.
    """
    if mode == "rgb":
        # Plot the gradient in red and blue
        grad_range = kwargs.get("grad_range", 3.0)
        red = gx * 255 / grad_range + 127
        red = np.clip(red, 0, 255)
        blue = gy * 255 / grad_range + 127
        blue = np.clip(blue, 0, 255)
        image = np.stack((red, np.zeros_like(red), blue), axis=-1).astype(np.uint8)
        if mask is not None:
            image[np.logical_not(mask)] = np.array([127, 0, 127])
        ax.imshow(image)
    elif mode == "quiver":
        # Plot the gradient in quiver
        n_skip = kwargs.get("n_skip", 5)
        quiver_scale = kwargs.get("quiver_scale", 10.0)
        imgh, imgw = gx.shape
        X, Y = np.meshgrid(np.arange(imgw)[::n_skip], np.arange(imgh)[::n_skip])
        U = gx[::n_skip, ::n_skip] * quiver_scale
        V = -gy[::n_skip, ::n_skip] * quiver_scale
        if mask is None:
            mask = np.ones_like(gx)
        else:
            mask = np.copy(mask)
        mask = mask[::n_skip, ::n_skip]
        ax.quiver(X[mask], Y[mask], U[mask], V[mask], units="xy", scale=1, color="red")
        ax.set_xlim(0, imgw)
        ax.set_ylim(imgh, 0)
    else:
        raise ValueError("Unknown plot gradient mode %s" % mode)
