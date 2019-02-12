from typing import Tuple, Optional
import numpy as np
from numpy.linalg import solve, pinv
from scipy.stats import norm
from .ransac import BaseModel, Processor

ConicCoef = Tuple[float, float, float, float, float]

def ellipse2conic(ellipse: ConicCoef) -> ConicCoef:
    x, y, a, b, Θ = ellipse
    sinΘ, cosΘ = np.sin(Θ), np.cos(Θ)
    A = (sinΘ ** 2 / b ** 2 + cosΘ ** 2 / a ** 2)
    B = (2 * sinΘ * cosΘ / b ** 2 - 2 * sinΘ * cosΘ / a ** 2)
    C = ((cosΘ ** 2 / b ** 2 + sinΘ ** 2 / a ** 2))
    D = ((-2 * x * sinΘ ** 2 / b ** 2 - 2 * y * sinΘ * cosΘ / b ** 2
         - 2 * x * cosΘ ** 2 / a ** 2 + 2 * y * sinΘ * cosΘ / a ** 2))
    E = ((-2 * x * sinΘ * cosΘ / b ** 2 - 2 * y * cosΘ ** 2 / b ** 2
          + 2 * x * sinΘ * cosΘ / a ** 2 - 2 * y * sinΘ ** 2 / a ** 2))
    F = - 1 + (x ** 2 * sinΘ ** 2 / b ** 2 + 2 * x * y * sinΘ * cosΘ / b ** 2
               + y ** 2 * cosΘ ** 2 / b ** 2 + x ** 2 * cosΘ ** 2 / a ** 2
               - 2 * x * y * sinΘ * cosΘ / a ** 2 + y ** 2 * sinΘ ** 2 / a ** 2)
    return A / F, B / F, C / F, D / F, E / F

def conic2ellipse(conic: ConicCoef) -> ConicCoef:
    A, B, C, D, E = conic  # scales F to be 1
    Θ = np.arctan2(B, C - A) / 2
    sinΘ2, cosΘ2 = np.sin(Θ) ** 2, np.cos(Θ) ** 2
    x = (2 * D * C - B * E) / (B * B - 4 * A * C)
    y = (2 * A * E - D * B) / (B * B - 4 * A * C)
    F = 1 / (- 1 + x ** 2 * A + y ** 2 * C + x * y * B)  # for the real F
    AF = A * F
    CF = C * F
    denom = sinΘ2 ** 2 - cosΘ2 ** 2
    a = np.sqrt(denom / (CF * sinΘ2 - AF * cosΘ2))
    b = np.sqrt(denom / (AF * sinΘ2 - CF * cosΘ2))
    return x, y, a, b, Θ

class EllipseNormalizer(Processor):
    def preprocess(self, samples: np.ndarray) -> np.ndarray:
        """Normalizes samples"""
        self.mean = samples.mean()
        self.scale = np.sqrt(2) / np.mean(np.abs(samples - self.mean))
        return self.scale * (samples - self.mean)

    def threshold(self) -> float:
        return norm.ppf(0.975) * self.scale  # scale factor * normal cdf at 0.975

    def postprocess(self, coef: ConicCoef) -> ConicCoef:
        """Denormalizes conic
            A*s**2*x**2 + B*s**2*x*y + C*s**2*y**2\
            + (-2*A*cx*s**2 - B*cy*s**2 + D*s)*x\
            + (-B*cx*s**2 - 2*C*cy*s**2 + E*s)*y\
            + A*cx**2*s**2 + B*cx*cy*s**2 + C*cy**2*s**2 - D*cx*s - E*cy*s + 1
        """
        a, b, c, d, e = coef
        s, s2, cx, cy = self.scale, self.scale ** 2, self.mean.real, self.mean.imag
        A, B, C = a * s2, b * s2, c * s2
        D = d * s - 2 * a * cx * s2 - b * cy * s2
        E = e * s - b * cx * s2 - 2 * c * cy * s2
        F = a * cx ** 2 * s2 + b * cx * cy * s2 + c * cy ** 2 * s2 - d * cx * s - e * cy * s + 1
        return A / F, B / F, C / F, D / F, E / F

class Conic(BaseModel):
    min_sample_no = 5
    _ellipse = None
    _coef: Optional[ConicCoef] = None

    def __init__(self, coef: ConicCoef, processor: Optional[EllipseNormalizer] = None):
        """coef of conic curve such as (A, B, C, D, E)"""
        self.processor = processor
        self.set(coef)

    def set(self, coef: ConicCoef):
        self._coef = coef
        A, B, C, D, E = coef
        self.det = B ** 2 - 4 * A * C
        B, D, E = B / 2, D / 2, E / 2
        self.mat = np.array([[A, B, D], [B, C, E], [D, E, 1]])

    @property
    def coef(self) -> ConicCoef:
        if self._coef is None:
            raise ValueError("conic coefficients not set yet!")
        return self._coef if self.processor is None else self.processor.postprocess(self._coef)

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        mat = np.vstack([samples.real, samples.imag, np.ones_like(samples, dtype=np.float)]).T
        return (mat @ self.mat * mat).sum(axis=1)

    def is_valid(self) -> bool:
        if self._coef is None:
            raise ValueError("Conic testing before initialization!")
        if self.det >= 0:
            return False
        ellipse = conic2ellipse(self.coef)
        return 0.75 < ellipse[2] / ellipse[3] < 1.35

    @classmethod
    def fit(cls, points: np.ndarray, processor: Optional[EllipseNormalizer] = None):
        """
        Args:
            points: array of complex for points on R^2
            processor: normalzier for ellipse
        """
        x, y = points.real, points.imag
        A = np.vstack([x * x, x * y, y * y, x, y]).T
        coef = (pinv(A) @ (-np.ones(points.shape[0])[:, np.newaxis]))[:, 0]
        return cls(coef, processor)

    @classmethod
    def solve(cls, points: np.ndarray, processor: Optional[EllipseNormalizer] = None):
        """
        Args:
            points: array of complex for points on R^2
            processor: normalzier for ellipse
        """
        x, y = points.real, points.imag
        A = np.vstack([x * x, x * y, y * y, x, y]).T
        coef = solve(A, -np.ones(5))
        return cls(coef, processor)
