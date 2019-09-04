import numpy as np
import dataclasses as dt
from typing import Tuple, Any
from ptychoSampling.obj import Simulated2DObj, Obj
from ptychoSampling.probe import RectangleProbe, Probe
from ptychoSampling.grid import RectangleGrid, ScanGrid
from ptychoSampling.detector import Detector
from ptychoSampling.logger import logger

@dt.dataclass
class ObjParams:
    shape: Tuple = (192, 192) # y, x
    border_shape: Any = ((32, 32),(32, 32))
    border_const: float = 1.0

@dt.dataclass
class ProbeParams:
    n_photons: float = 1e8
    defocus_dist: float = 0.15
    width_dist: Tuple = (7e-6, 7e-6) # x, y

@dt.dataclass
class GridParams:
    step_dist: Tuple = (3.5e-6, 3.5e-6) # x, y

@dt.dataclass
class DetectorParams:
    npix: int = 64
    obj_dist: float = 14.0
    pixel_size: float = 55e-6


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

        obj_pixel_size = (self.wavelength * self.detector.obj_dist / (self.detector.pixel_size * self.detector.npix
                          * self.upsampling_factor))
        probe_npix = self.detector.npix * self.upsampling_factor

        if obj is not None:
            if obj.pixel_size is not None and obj.pixel_size != obj_pixel_size:
                e = ValueError("Mismatch between the provided pixel size and the pixel size calculated from scan "
                               + "parameters.")
                logger.error(e)
                raise e
            obj.pixel_size = obj_pixel_size
        else:
            self._obj_params = ObjParams(**obj_params)
            self.obj = Simulated2DObj(**dt.asdict(self._obj_params))

        if probe is not None:
            check = ((probe.shape[0] != probe_npix)
                     or (probe.wavelength != self.wavelength)
                     or (probe.pixel_size != obj_pixel_size))
            if check:
                e = ValueError("Supplied probe parameters do not match with supplied scan and detector parameters.")
                logger.error(e)
                raise e
            self.probe = probe
        else:
            self._probe_params = ProbeParams(**probe_params)
            self.probe = RectangleProbe(wavelength=wavelength,
                                        pixel_size=(obj_pixel_size,obj_pixel_size),
                                        shape=(probe_npix,probe_npix),
                                        **dt.asdict(self._probe_params))

        if scan_grid is not None:
            self.scan_grid = scan_grid
            self._scan_grid_params = None
        else:
            self._scan_grid_params = GridParams(**scan_grid_params)
            self.scan_grid = RectangleGrid(obj_w_border_shape=self.obj.bordered_array.shape,
                                           probe_npix=self.probe.shape[0],
                                           obj_pixel_size=obj_pixel_size,
                                           **dt.asdict(self._scan_grid_params))
        self.scan_grid.checkOverlap()
        self._calculateDiffractionPatterns()

    def _calculateDiffractionPatterns(self):
        wv = self.probe.wavefront.copy()#.array
        intensities_all = []

        for i, (px, py) in enumerate(self.scan_grid.positions_pix):
            if self.scan_grid.subpixel_scan:
                u = np.fft.fftshift(np.arange(self.probe.shape[0]))
                spx, spy = self.scan_grid.subpixel_scan[i]
                phase_factor = (-2 * np.pi * (u * spx + u[:,None] * spy) / self.probe.shape[0])
                phase_ramp = np.complex(np.cos(phase_factor), np.sin(phase_factor))
                wv = (wv.fft2() * phase_ramp).ifft2()
                #wv = np.fft.ifft2(np.fft.fft2(wv, norm='ortho') * phase_ramp, norm='ortho')

            obj_slice = np.fft.fftshift(self.obj.bordered_array[py: py + self.probe.shape[0],
                                        px: px + self.probe.shape[0]])
            wv = wv * obj_slice
            #det_wave = np.fft.fft2(wv, norm='ortho')
            #intensity = np.abs(det_wave)**2
            intensities = wv.fft2().intensities
            intensities_all.append(intensities)

        self.intensities = np.random.poisson(intensities_all) if self.poisson_noise else np.array(intensities_all)










