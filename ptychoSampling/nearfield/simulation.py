import numpy as np
import dataclasses as dt
from typing import Tuple, Any
from ptychoSampling.obj import Simulated2DObj, Obj
from ptychoSampling.probe import GaussianSpeckledProbe, Probe
from ptychoSampling.grid import RectangleGrid, ScanGrid
from ptychoSampling.detector import Detector
from ptychoSampling.logger import logger

@dt.dataclass
class ObjParams:
    shape: Tuple = (192, 192) # y, x
    border_shape: Any = ((32, 32), (32, 32))
    border_const: float = 1.0

@dt.dataclass
class ProbeParams:
    n_photons: float = 2e9
    width_dist: Tuple = (65 * 3e-7, 65 * 3e-7) # y, x
    speckle_window_npix: int = 20

@dt.dataclass
class GridParams:
    step_dist: Tuple = (44 * 3e-7, 44 * 3e-7) # y, x
    full_field_probe: bool = True
    scan_grid_boundary_pix: np.ndarray = np.array([[20,492],[20, 492]])

@dt.dataclass
class DetectorParams:
    shape: Tuple[int, int] = (512, 512)
    obj_dist: float = 0.0468
    pixel_size: Tuple[float, float] = (3e-7, 3e-7)


class Simulation:
    def __init__(self,
                 wavelength: float = 1.5e-10,
                 obj: Obj = None,
                 obj_params: dict = {},
                 probe: Probe = None,
                 probe_params: dict = {},
                 scan_grid: ScanGrid = None,
                 scan_grid_params: dict = {},
                 detector: Detector = None,
                 detector_params: dict = {},
                 poisson_noise: bool = True,
                 upsampling_factor: int = 1) -> None:

        self.wavelength = wavelength
        self.upsampling_factor = upsampling_factor
        self.poisson_noise = poisson_noise

        if obj or probe or scan_grid or detector:
            logger.warning("If one (or all) of obj, probe, scan_grid, or detector is supplied, "
                           + "then the corresponding _params parameter is ignored.")

        if detector is not None:
            self.detector = detector
            self._detector_params = {}
        else:
            self._detector_params = DetectorParams(**detector_params)
            self.detector = Detector(**dt.asdict(self._detector_params))

        obj_pixel_size = np.array(self.detector.pixel_size) * self.upsampling_factor
        probe_shape = np.array(self.detector.shape) * self.upsampling_factor

        if obj is not None:
            if obj.pixel_size is not None and np.any(obj.pixel_size != obj_pixel_size):
                e = ValueError("Mismatch between the provided pixel size and the pixel size calculated from scan "
                               + "parameters.")
                logger.error(e)
                raise e
            obj.pixel_size = obj_pixel_size
        else:
            self._obj_params = ObjParams(**obj_params)
            self.obj = Simulated2DObj(**dt.asdict(self._obj_params))

        if probe is not None:
            check = (np.any(probe.shape != probe_shape)
                     or (probe.wavelength != self.wavelength)
                     or np.any(probe.pixel_size != obj_pixel_size))
            if check:
                e = ValueError("Supplied probe parameters do not match with supplied scan and detector parameters.")
                logger.error(e)
                raise e
            self.probe = probe
        else:
            self._probe_params = ProbeParams(**probe_params)
            self.probe = GaussianSpeckledProbe(wavelength=wavelength,
                                               pixel_size=obj_pixel_size,
                                               shape=probe_shape,
                                               **dt.asdict(self._probe_params))

        if scan_grid is not None:
            self.scan_grid = scan_grid
            self._scan_grid_params = None
        else:
            self._scan_grid_params = GridParams(**scan_grid_params)
            self.scan_grid = RectangleGrid(obj_w_border_shape=self.obj.bordered_array.shape,
                                           probe_shape=self.probe.shape,
                                           obj_pixel_size=obj_pixel_size,
                                           **dt.asdict(self._scan_grid_params))

        self.scan_grid.checkOverlap()
        self._calculateDiffractionPatterns()

    def _calculateDiffractionPatterns(self):
        wv = self.probe.wavefront
        intensities_all = []
        self._transfer_function = None
        ny, nx = self.obj.bordered_array.shape

        if self.scan_grid.subpixel_scan:
            e = ValueError("Subpixel scan not supported for nearfield.")
            logger.error(e)
            raise e

        for i, (py, px) in enumerate(self.scan_grid.positions_pix):
            exit_wave = wv.ifftshift
            exit_wave[py:py + ny, px: px + nx] *= self.obj.bordered_array
            exit_wave = exit_wave.fftshift
            if self._transfer_function is None:
                self._transfer_function = np.zeros_like(exit_wave)
                det_wave = exit_wave.propTF(prop_dist=self.detector.obj_dist, transfer_function=self._transfer_function)
            else:
                det_wave = exit_wave.propTF(reuse_transfer_function=True,
                                            transfer_function=self._transfer_function)
            intensities_all.append(det_wave.intensities)


        self.intensities = np.random.poisson(intensities_all) if self.poisson_noise else np.array(intensities_all)










