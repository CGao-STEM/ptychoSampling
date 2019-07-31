#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import propagators
import utils
from typing import Optional, Tuple, List
import abc

class Probe(abc.ABC):
    def __init__(self, 
                 wavelength: float,
                 pixel_pitch: float,
                 npix: int,
                 n_photons: int=1e6,
                 defocus_dist: float = 0,
                 center_x: float = 0,
                 center_y: float = 0) -> None:

        self.wavelength = wavelength
        self.pixel_pitch = pixel_pitch
        self.npix = npix
        self.n_photons = n_photons
        self.defocus_dist = defocus_dist
        self.center_x = center_x
        self.center_y = center_y

        self.photons_flux = n_photons / npix**2
        self.wavefront = np.zeros((npix, npix), dtype='complex64')

    @abc.abstractmethod
    def _calculateWavefront(self) -> None:
        pass

    @property
    def gaussian_fit(self) -> dict:
        if not hasattr(self, '_gaussian_fit'):
            self._calculateGaussianFit()
        return self._gaussian_fit

    @property
    def gaussian_fwhm_max(self) -> float:
        if not hasattr(self, '_gaussian_fwhm_max'):
            self._calculateGaussianFit()
        return self._gaussian_fwhm_max


    def _calculateGaussianFit(self) -> None:
        from scipy.optimize import curve_fit

        intensities = np.abs(self.wavefront)**2

        x = np.arange(-self.npix // 2, self.npix // 2) * self.pixel_pitch
        xx, yy = np.meshgrid(x, x)
        popt, _ = curve_fit(utils.generalized2dGaussian, (xx, yy), intensities.flatten())
        amplitude, center_x, center_y, sigma_x, sigma_y, theta, offset = popt

        self._gaussian_fit = {"amplitude": amplitude,
                             "center_x": center_x,
                             "center_y": center_y,
                             "sigma_x": sigma_x,
                             "sigma_y": sigma_y,
                             "theta": theta,
                             "offset": offset}
        self._gaussian_fwhm_max = 2.355 * max(sigma_x, sigma_y)


    def _propagateWavefront(self) -> None:
        if self.defocus_dist> 0:
            propagation_type = propagators.checkPropagationType(self.npix,
                                                                self.wavelength,
                                                                self.defocus_dist,
                                                                self.pixel_pitch)
            if propagation_type == propagators.PropagationTypes.FARFIELD:
                raise ValueError("Defocus distance too large. Only near field defocus supported.")
            self.wavefront, _ = propagators.propagate(self.wavefront,
                                                      self.pixel_pitch,
                                                      self.wavelength,
                                                      self.defocus_dist)
            if hasattr(self, '_gaussian_fit'):
                self._calculateGaussianFit()

class CustomProbeFromArray(Probe):
    def __init__(self, wavefront_array: np.ndarray,
                 wavelength: float,
                 pixel_pitch: float,
                 defocus_dist: float=0) -> None:
        super().__init__(wavelength, pixel_pitch,
                       wavefront_array.shape[0],
                       np.sum(np.abs(wavefront_array)**2),
                       defocus_dist)
        self.wavefront = wavefront_array.copy()
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        self._propagateWavefront()


class RectangleProbe(Probe):
    def __init__(self, *args: List,
                width_x: float,
                width_y: float,
                **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.width_x = width_x
        self.width_y = width_y
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:

        x = np.arange(-self.npix // 2, self.npix // 2) * self.pixel_pitch
        y = np.arange(-self.npix // 2, self.npix // 2)[:, np.newaxis] * self.pixel_pitch
        xarr = np.where(np.abs(x - self.center_x) <= self.width_x / 2, 1, 0)
        yarr = np.where(np.abs(y - self.center_y) <= self.width_y / 2, 1, 0)
        wavefront= (xarr * yarr).astype('complex64')

        scaling_factor = np.sqrt(self.n_photons / (np.abs(wavefront)**2).sum())
        self.wavefront = scaling_factor * wavefront
        self._propagateWavefront()

class CircularProbe(Probe):
    def __init__(self, *args: List,
                 radius: float,
                 **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self._calculateWavefront()

    def _calculateWavefront(self):

        x = np.arange(-self.npix // 2, self.npix // 2) * self.pixel_pitch
        y = np.arange(-self.npix // 2, self.npix // 2)[:, np.newaxis] * self.pixel_pitch
        radsq = (x - self.center_x)**2 + (y - self.center_y)**2

        wavefront = np.zeros(radsq.shape, dtype='complex64')
        wavefront[radsq < self.radius**2] = 1.0
        wavefront[radsq == self.radius**2] = 0.5

        scaling_factor = np.sqrt(self.n_photons / (np.abs(wavefront)**2).sum())
        self.wavefront = scaling_factor * wavefront
        self._propagateWavefront()
        

class GaussianProbe(Probe):
    def __init__(self, *args: List,
                 sigma_x: float,
                 sigma_y: float,
                 theta: float = 0,
                 **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.theta = theta
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        x = np.arange(-self.npix // 2, self.npix // 2) * self.pixel_pitch
        xx, yy = np.meshgrid(x, x)
        intensity = utils.generalized2dGaussian((xx, yy),
                                               amplitude=1,
                                               center_x=self.center_x,
                                               center_y=self.center_y,
                                               sigma_x = self.sigma_x,
                                               sigma_y = self.sigma_y,
                                               theta=self.theta,
                                               offset=0)

        scaling_factor = np.sqrt(self.n_photons / intensity.sum())
        self.wavefront = scaling_factor * np.sqrt(intensity).astype('complex64')
        self.wavefront = np.reshape(self.wavefront, (self.npix, self.npix))
        self._propagateWavefront()

class GaussianSpeckledProbe(Probe):
    def __init__(self, *args: List,
                 sigma_x: float,
                 sigma_y: float,
                 speckle_window_npix: int,
                 theta: float = 0,
                 **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.speckle_window_npix = speckle_window_npix
        self.theta = theta
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        x = np.arange(-self.npix // 2, self.npix // 2) * self.pixel_pitch
        xx, yy = np.meshgrid(x, x)
        intensity = utils.generalized2dGaussian((xx, yy),
                                                amplitude=1,
                                                center_x=self.center_x,
                                                center_y=self.center_y,
                                                sigma_x=self.sigma_x,
                                                sigma_y=self.sigma_y,
                                                theta=self.theta,
                                                offset=0)
        amplitude = np.sqrt(intensity).astype('complex64').reshape(self.npix, self.npix)

        speckle = utils.getSpeckle(self.npix, self.speckle_window_npix)
        wavefront = amplitude * speckle
        scaling_factor = np.sqrt(self.n_photons / np.sum(np.abs(wavefront)**2))
        self.wavefront = scaling_factor * wavefront
        self._propagateWavefront()

class FocusCircularProbe(Probe):
    def __init__(self, *args: List,
                 focal_length: Optional[float] = None,
                 aperture_radius: Optional[float] = None,
                 focus_radius_npix: Optional[float] = None,
                 oversampling: bool = True,
                 oversampling_npix: float = 1024,
                 **kwargs: dict,) -> None:
        super().__init__(*args, **kwargs)
        self.focus_radius_npix = focus_radius_npix
        self.focal_length = focal_length
        self.aperture_radius = aperture_radius
        self.oversampling = oversampling
        self.oversampling_npix = oversampling_npix
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:

        if self.focus_radius_npix is not None:
            print('Warning: if focus_radius_npix float is supplied, '
                  + 'we ignore any supplied focal_length and aperture radius.')
            self.aperture_radius = None
            self.focal_length = None
            # Assuming resolution equal to pixel pitch and focal length of 0.1 m.
            delta_r = self.pixel_pitch
            focal_length = 10.0
            focus_radius = delta_r * self.focus_radius_npix
            aperture_radius = (self.wavelength * 1.22 * np.pi * focal_length
                               / 2 / np.pi / focus_radius)
        else:
            if self.aperture_radius is None or self.focal_length is None:
                raise ValueError('Either focus_radius_npix or '
                                 + 'BOTH aperture_radius and focal_length must be supplied.')
            aperture_radius = self.aperture_radius
            focal_length = self.focal_length

        npix_oversampled = max(self.oversampling_npix, self.npix) if self.oversampling else self.npix
        pixel_pitch_aperture = self.wavelength * focal_length / (npix_oversampled * self.pixel_pitch)

        x = np.arange(-npix_oversampled // 2, npix_oversampled // 2) * pixel_pitch_aperture

        r = np.sqrt(x**2 + x[:,np.newaxis]**2).astype('float32')
        circ_wavefront = np.zeros(r.shape)
        circ_wavefront[r < aperture_radius] = 1.0
        circ_wavefront[r == aperture_radius] = 0.5

        probe_vals = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ_wavefront), norm='ortho'))

        n1 = npix_oversampled // 2 - self.npix // 2
        n2 = npix_oversampled // 2 + self.npix // 2

        scaling_factor = np.sqrt(self.n_photons / np.sum(np.abs(probe_vals)**2))
        self.wavefront = probe_vals[n1:n2, n1:n2].astype('complex64') * scaling_factor
        self._propagateWavefront()

