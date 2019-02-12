from os.path import join
from pkg_resources import Requirement, resource_filename
from pytest import fixture
import numpy as np

from tiffreader import TiffReader
from starburst.main import StarBurst, convert_image, _smooth
from starburst.main import fit, Conic, EllipseNormalizer, refine_pupil
from starburst.conic import conic2ellipse
from utils import _plot_conic, _plot_ellipse
from mplplot.importer import MplFigure

DATA_FOLDER = "starburst/test/data"

@fixture
def data():
    data_file = resource_filename(Requirement.parse("starburst"), join(DATA_FOLDER, "pupil.tif"))
    data = TiffReader.open(data_file)
    return data

data = TiffReader.open("/media/data/2018-hiroki-pupil-tracking/animal4229pupil2.tif")

# noinspection PyShadowingNames
def test_detect(data):
    cx, cy = data.shape[1] // 2, data.shape[0] // 2
    idx = 0
    radii = list()
    for idx in range(60, 70):
        frame = (convert_image(_smooth(data[idx], 1)))
        y_size, x_size = frame.shape
        points = StarBurst(cx + cy * 1j, 30).detect(frame)
        result = fit(Conic, EllipseNormalizer(), points)
        min_result = refine_pupil(frame, result)
        x, y, a, b, Θ = min_result  # conic2ellipse(result.coef)
        radii.append(np.sqrt(a * b))
        cosΘ, sinΘ = np.cos(Θ), np.sin(Θ)
        x_mesh, y_mesh = np.meshgrid(np.arange(x_size), np.arange(y_size))
        mask = (((x_mesh - x) * cosΘ - (y_mesh - y) * sinΘ) / a) ** 2\
            + (((x_mesh - x) * sinΘ + (y_mesh - y) * cosΘ) / b) ** 2 < 1
        with MplFigure(f"sample_frame_{idx}.png", (6, 4)) as ax:
            ax.imshow(frame / 2 + mask * 128)
            ax.scatter(points.real, points.imag)
    return radii
##
from cProfile import Profile
profile = Profile()
profile.run("test_detect(data)")
profile.dump_stats("stats.prof")
print('done')
##
radii = test_detect(data)
