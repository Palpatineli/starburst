from typing import List, Tuple
from sys import maxsize
import numpy as np
from .utils import ΔΘ, Rect

class StarBurst(object):
    _ray_no: int = 18
    _feature_candidate_no: int = 10
    _angle_spread: float = 100 * ΔΘ

    def __init__(self, start: complex, edge_threshold: float):
        self.start, self.threshold = start, edge_threshold

    def detect(self, image: np.ndarray, ray_length: float = 7.0) -> np.ndarray:
        """
        Returns:
            points: np.ndarray[complex] for points (mostly) circling a ellipse on the edge of pupil
        """
        rect = Rect.like(image)

        def get_points(start: complex, angles: List[float], threshold: float) -> Tuple[np.ndarray, np.ndarray]:
            points: List[complex] = list()
            diffs: List[float] = list()
            for angle in angles:
                ray_section = ray_length * np.exp(1j * angle)
                point_0 = start + ray_section
                if not rect.contains(point_0):
                    continue
                for _ in range(maxsize):
                    point_1 = point_0 + ray_section
                    if not rect.contains(point_1):
                        break
                    try:
                        diff = int(image[int(round(point_1.imag)), int(round(point_1.real))]) -\
                            int(image[int(round(point_0.imag)), int(round(point_0.real))])
                    except IndexError as e:
                        print(rect.x1, rect.y1, rect.contains(point_0), rect.contains(point_1))
                        print(point_0, point_1, image.shape)
                        raise e
                    if diff >= threshold:
                        points.append(point_0)
                        diffs.append(diff)
                        break
                    point_0 = point_1
            return np.array(points), np.array(diffs)

        start = self.start
        for _ in range(10):
            threshold = self.threshold
            while True:
                points, diffs = get_points(start, np.linspace(-np.pi, np.pi, self._ray_no, False), threshold)
                if len(points) >= self._feature_candidate_no:
                    break
                if threshold < 2:
                    print("Adaptive Threshold too low!")
                    return None
                threshold -= 1
            angle_spreads = np.angle(self.start - points)[:, np.newaxis]\
                + np.array([-np.pi / 3.6, np.pi / 3.6])[np.newaxis, :]
            angle_steps = diffs * self._ray_no * (5.0 / 36.0) / threshold
            back = [get_points(point, np.linspace(angle_spread[0], angle_spread[1], angle_step, False), threshold)[0]
                    for point, angle_spread, angle_step in zip(points, angle_spreads, angle_steps)]
            points = np.hstack([points] + back)
            new_start = points.mean()
            if np.abs(start - new_start) < 10:
                break
            start = new_start
        else:
            print("Edge Points not converging")
            return None
        return points
