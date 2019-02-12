from typing import Tuple
import numpy as np
from scipy.optimize import minimize, Bounds
import cv2
from .conic import Conic, conic2ellipse

ConicCoef = Tuple[float, float, float, float, float]
ANGLE_SCALE = 15

def __pupil_score(conic_coef: ConicCoef, image) -> float:
    """calculate the sum of pixel values just outside and inside the conic, try to maximize inside/outside"""
    cx, cy, a, b, Θ = conic_coef
    sinΘ, cosΘ = np.sin(Θ / ANGLE_SCALE), np.cos(Θ / ANGLE_SCALE)
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    ellipse_mask = ((((x - cx) * cosΘ - (y - cy) * sinΘ) / a) ** 2
                    + (((y - cy) * cosΘ + (x - cx) * sinΘ) / b) ** 2 < 1).astype(np.uint8)
    inner = ellipse_mask - cv2.erode(ellipse_mask, np.ones((3, 3), np.uint8), iterations=1)
    outer = cv2.dilate(ellipse_mask, np.ones((3, 3), np.uint8), iterations=1) - ellipse_mask
    return image[inner.astype(np.bool_)].sum() / image[outer.astype(np.bool_)].sum()

def _pupil_score(conic_coef: ConicCoef, image) -> float:
    """calculate the sum of pixel values just outside and inside the conic, try to maximize inside/outside"""
    cx, cy, a, b, Θ = conic_coef
    mask_size = int(max(a, b)) + 5
    start_x, start_y = int(cx) - mask_size, int(cy) - mask_size
    sinΘ, cosΘ = np.sin(Θ / ANGLE_SCALE), np.cos(Θ / ANGLE_SCALE)
    x, y = np.meshgrid(np.arange(mask_size * 2), np.arange(mask_size * 2))
    x -= mask_size
    y -= mask_size
    ellipse_mask = (((x * cosΘ - y * sinΘ) / a) ** 2
                    + ((y * cosΘ + x * sinΘ) / b) ** 2 < 1).astype(np.uint8)
    inner = ellipse_mask - cv2.erode(ellipse_mask, np.ones((7, 7), np.uint8), iterations=1)
    outer = cv2.dilate(ellipse_mask, np.ones((7, 7), np.uint8), iterations=1) - ellipse_mask
    cutout = image[start_y: start_y + 2 * mask_size, start_x: start_x + 2 * mask_size]
    return cutout[inner.astype(np.bool_)].sum() / cutout[outer.astype(np.bool_)].sum()

def refine_pupil(image: np.ndarray, conic: Conic) -> ConicCoef:
    """gradient descent to fit the conic to actual image
    Args:
        image: image in uint8
        conic: it's .coef is [A, B, C, D, E] assuming F = 1
    """
    x_size, y_size = image.shape[1], image.shape[0]
    bounds = Bounds((x_size // 5, y_size // 5, 30, 30, -np.pi * ANGLE_SCALE),
                    (int(x_size * 0.8), int(y_size * 0.8), 100, 100, np.pi * ANGLE_SCALE))
    init = conic2ellipse(conic.coef)
    result = minimize(_pupil_score, init, args=(image), bounds=bounds, options={'eps': 3.0})
    return result.x
