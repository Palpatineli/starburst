from typing import Optional, Union, Tuple
import numpy as np

ΔΘ = np.pi / 180
Num = Union[float, int]
Ellipse = Tuple[complex, complex, float]

class Rect(object):
    def __init__(self, x: int, y: int, a: int, b: Optional[int] = None):
        b = a if b is None else b
        self.x0, self.x1 = max(x - a // 2, 0), x + a - a // 2
        self.y0, self.y1 = max(y - b // 2, 0), y + b - b // 2

    @classmethod
    def like(cls, image: np.ndarray):
        return cls(image.shape[1] // 2, image.shape[0] // 2, image.shape[1], image.shape[0])

    def take(self, image: np.ndarray) -> np.ndarray:
        y_max, x_max = image.shape[0: 2]
        return image[self.y0: min(self.y1, y_max), self.x0: min(self.x1, x_max)]

    def contains(self, point: Union[complex, Tuple[float, float]]) -> bool:
        if isinstance(point, complex):
            return (self.x1 > int(round(point.real)) >= 0) and (self.y1 > int(round(point.imag)) >= 0)
        else:
            return (self.x1 > int(round(point[0])) >= 0) and (self.y1 > int(round(point[1])) >= 0)

    def translate(self, x: float, y: float) -> Tuple[float, float]:
        return self.x0 + x, self.y0 + y
