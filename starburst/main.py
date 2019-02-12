from os.path import splitext
import numpy as np
from scipy.signal import gaussian, convolve
from tqdm import tqdm
from tiffreader import TiffReader
from uifunc import FileSelector
from .starburst import StarBurst
from .ransac import fit
from .conic import Conic, EllipseNormalizer
from .refine import refine_pupil

def reduce_noise_temporal_shift(image, column_factor, hysteresis_factor, max_pixel=255):
    line_mean = image.mean(axis=1)
    normalizer = line_mean * hysteresis_factor + column_factor * (1 - hysteresis_factor)
    adjuster = np.tile(normalizer - line_mean, [1, image.shape[1]])
    image = np.minimum(image + adjuster, max_pixel)
    return image, normalizer

def _smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = int(round(2.5 * sigma))
    gaussian_1d = gaussian(kernel_size, sigma)
    kernel = np.outer(gaussian_1d, gaussian_1d)
    kernel /= kernel.sum()
    return convolve(np.pad(image, kernel_size, 'symmetric'), kernel, "valid")

def convert_image(image: np.ndarray) -> np.ndarray:
    """drop image to u8 grayscale and cutoff the highlight part"""
    image = image.astype(np.int32)
    dist = np.sort(image.ravel())
    minimum, maximum = dist[int(dist.size * 0.025)], dist[int(dist.size * 0.75)]
    image = ((image - minimum) * (256.0 / (maximum - minimum)))
    image[image > 255] = 255
    image[image < 0] = 0
    return image.astype(np.uint8)

@FileSelector([".tiff", ".tif"])
def main(file_path: str):
    tif = TiffReader.open(file_path)
    cy, cx = 600, 640
    radii = list()
    for frame, idx in tqdm(zip(tif, range(tif.length)), total=tif.length):
        frame = convert_image(_smooth(frame, 1))
        points = StarBurst(cx + cy * 1j, 30).detect(frame)
        conic = fit(Conic, EllipseNormalizer(), points)
        refined_ellipse = refine_pupil(frame, conic)
        radii.append(np.sqrt(refined_ellipse[2] * refined_ellipse[3]))
    np.save(splitext(file_path)[0] + '.npy', np.asarray(radii, dtype=np.float))
##
