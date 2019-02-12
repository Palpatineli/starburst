import numpy as np
import matplotlib.pyplot as plt

def _plot_conic(frame, coef):
    if len(frame) > 2:
        x_size, y_size = frame.shape[1], frame.shape[0]
    else:
        x_size, y_size = frame[0], frame[1]
    x_mesh, y_mesh = np.meshgrid(np.arange(x_size), np.arange(y_size))
    mask = coef[0] * (x_mesh ** 2) + coef[1] * x_mesh * y_mesh \
        + coef[2] * y_mesh ** 2 + coef[3] * x_mesh + coef[4] * y_mesh + 1 < 0
    plt.imshow(mask)
    return mask

def _plot_ellipse(frame, coef):
    if len(frame) > 2:
        x_size, y_size = frame.shape[1], frame.shape[0]
    else:
        x_size, y_size = frame[0], frame[1]
    x_mesh, y_mesh = np.meshgrid(np.arange(x_size), np.arange(y_size))
    x, y, a, b, Θ = coef
    cosΘ, sinΘ = np.cos(Θ), np.sin(Θ)
    mask = (((x_mesh - x) * cosΘ - (y_mesh - y) * sinΘ) / a) ** 2\
        + (((x_mesh - x) * sinΘ + (y_mesh - y) * cosΘ) / b) ** 2 < 1
    plt.imshow(mask)
    return mask
