from typing import Optional
import numpy as np
import cv2
from .utils import Rect, ΔΘ

class Circle(object):
    x: float = 0
    y: float = 0
    r: float = 0

    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r

class CorneaReflection(Circle):
    @classmethod
    def fit(cls, image: np.ndarray, search_window: Optional[Rect] = None) -> Optional["CorneaReflection"]:
        """detect glare by gradual change of pixel intensity"""
        if search_window is None:
            search_window = Rect(image.shape[0] // 2, image.shape[1] // 2, 201)
        image = search_window.take(image)
        score = list()
        for threshold in range(image.max(), 0, -1):
            object_no, labeled, stats, centroid = cv2.connectedComponentsWithStats((image > threshold).astype(np.uint8),
                                                                                   connectivity=8)
            if object_no >= 2:
                max_idx = stats[:, 4].argmax()
                max_area = stats[max_idx, 4]
                score.append(max_area / (stats[:, 4].sum() / max_area))
                if len(score) > 1 and score[-1] < score[-2]:
                    x, y = search_window.translate(*centroid[max_idx, :])
                    return cls(x, y, np.sqrt(max_area / np.pi))
        return None

    def remove(self, image: np.ndarray) -> Optional[np.ndarray]:
        """remove glare by using edge values to fill the circle"""
        if (self.x + self.r > image.shape[1]) | (self.y + self.r > image.shape[0]) | (self.x <= self.r) \
                | (self.y <= self.r):
            return None
        Θ = np.arange(0, np.pi * 2, ΔΘ)
        mesh = np.arange(self.r)[:, np.newaxis] * np.exp(1j * Θ[np.newaxis, :]) + self.x * 1j + self.y
        x_edge, y_edge = np.rint(mesh[:, -1].real).astype(np.int), np.rint(mesh[:, -1].imag).astype(np.int)
        edge = image[x_edge, y_edge]
        coef = (np.arange(self.r) / self.r)[np.newaxis, :]
        x_mesh, y_mesh = np.rint(mesh.real).astype(np.int), np.rint(mesh.imag).astype(np.int)
        image[x_mesh, y_mesh] = edge.mean() * (1 - coef) + edge[:, np.newaxis] * coef
        return image
