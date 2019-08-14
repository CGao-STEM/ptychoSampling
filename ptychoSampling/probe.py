#Author - Saugat Kandel
# coding: utf-8


import numpy as np
from ptychoSampling import propagators, utils
from typing import Optional, List, Tuple
import abc

class Probe(abc.ABC):
    """Abstract class that can be inherited as necessary for arbitrary probe structures.

    Assumes a square probe array.

    Parameters
    ----------
    wavelength : float
        Wavelength of the probe wavefront.
    pixel_pitch : float
        Pixel pitch at the sample plane.
    npix : int
        Number of pixels in each side of the probe array.
    n_photons : float
        Total number of photons in the probe wavefront, at the sample plane.
    defocus_dist : float, optional
        Distance to further propagate the probe once the initial structure is defined (in m). For example,
        if we want to simulate the probe beam due to a square aperture close to the sample, then we can first create
        the square probe structure (exit wave from the aperture), and then use this `defocus_dist` parameter to
        propagate the wavefront to the sample plane. Default is set to 0 m.
    center_x : float, optional
        Displacement (along the x-axis) of the center of the probe wavefront from the center of the pixellation in the
        sample plane (in m). Defaults to 0 m.
    center_y : float, optional
        Displacement (along the y-axis) of the center of the probe wavefront from the center of the pixellation in
        the sample plane (in m). Defaults to 0 m.

    Attributes
    ----------
    wavelength, pixel_pitch, n_photons, defocus_dist, center_x, center_y : see Parameters
    photons_flux : float
        Average number of photons per pixel (at the sample plane).
    wavefront : ndarray(complex)
        Probe wavefront at the sample plane, after any supplied defocus.
    """
    def __init__(self, 
                 wavelength: float,
                 pixel_pitch: float,
                 npix: int,
                 n_photons: float,
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
        """Abstract method that, when inherited, should calculate the probe wavefront from the supplied parameters."""
        pass

    @property
    def gaussian_fit(self) -> dict:
        r"""Fit a 2-d gaussian to the probe intensities (not amplitudes) and return the fit parameters.

        The returned dictionary contains the following fit parameters (as described in [1]_):
            * ``amplitude`` : Amplitude of the fitted gaussian.
            * ``center_x`` : X-offset of the center of the fitted gaussian.
            * ``center_y`` : Y-offset of the center of the fitted gaussian.
            * | ``theta`` : Clockwise rotation angle for the gaussian fit. Prior to the rotation, primary axes of the \
              | gaussian  are aligned to the X and Y axes.
            * ``sigma_x`` : Spread (standard deviation) of the gaussian fit along the x-axis (prior to rotation).
            * ``sigma_y`` : Spread of the gaussian fit along the y-axis (prior to rotation).
            * | ``offset`` : Constant level of offset applied to the intensity throughout the probe array. This could,
              | for instance, represent the level of background radiation.

        Returns
        -------
        out : dict
            Dictionary containing the fit parameters.

        See also
        --------
        _calculateGaussianFit

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        """
        if not hasattr(self, '_gaussian_fit'):
            self._calculateGaussianFit()
        return self._gaussian_fit

    @property
    def gaussian_fwhm(self) -> Tuple[float, float]:
        r"""Fit a 2-d gaussian the the probe intensities (not amplitudes) and return the FWHM along the primary axes.

        The full width at half-maximum (FWHM) is calculated as :math:`\text{FWHM}_x = \sigma_x * 2.355` and similarly for
        :math:`\textrm{FWHM}_y`.

        Returns
        -------
        out : Tuple[float, float]
            FWHM along the primary axes.

        See also
        --------
        gaussian_fit
        _calculateGaussianFit
        """
        if not hasattr(self, '_gaussian_fwhm_max'):
            self._calculateGaussianFit()
        return self._gaussian_fwhm


    def _calculateGaussianFit(self) -> None:
        r"""Fit a 2d gaussian to the probe intensities.

        Performs a least-squares fit (using ``scipy.optimize.curve_fit``) to fit a 2d gaussian to the probe
        intensities. Uses the calculated gaussian spread to calculate the FWHM as well.

        See also
        --------
        gaussian_fit
        utils.generalized2dGaussian
        """
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
        self._gaussian_fwhm = 2.355 * np.array((sigma_x, sigma_y))


    def _propagateWavefront(self) -> None:
        """Propagate the probe wavefront by `defocus_dist`, and, if the gaussian fit has been priorly calculated,
        recalculate the fit."""
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
    r"""Create a Probe object using a supplied wavefront.

    See documentation for `Probe` for information on the attributes.

    Parameters
    ----------
    wavefront_array : array_like(complex)
        Square 2D array that contains the probe wavefront.
    wavelength, pixel_pitch, defocus_dist : see documentation for `Probe`.

    See also
    --------
    Probe
    """
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
        """Simply propagates the supplied wavefront by `defocus_dist`."""
        self._propagateWavefront()


class RectangleProbe(Probe):
    r"""Create a Probe object using the exit wave from a rectangular aperture.

    Also see documentation for `Probe` information on other parameters and attributes.

    Parameters
    ----------
    width_x : float
        Width of the aperture along the x-direction (in m).
    width_y : float
        Width of the aperture along the y-direction (in m).
    *args : List
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : dict
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    width_x, width_y : see Parameters

    See also
    --------
    Probe
    """
    def __init__(self, *args: List,
                width_x: float,
                width_y: float,
                **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.width_x = width_x
        self.width_y = width_y
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Calculates and propagates the exit wave from a rectangular aperture of the supplied dimensions."""

        x = np.arange(-self.npix // 2, self.npix // 2) * self.pixel_pitch
        y = np.arange(-self.npix // 2, self.npix // 2)[:, np.newaxis] * self.pixel_pitch
        xarr = np.where(np.abs(x - self.center_x) <= self.width_x / 2, 1, 0)
        yarr = np.where(np.abs(y - self.center_y) <= self.width_y / 2, 1, 0)
        wavefront= (xarr * yarr).astype('complex64')

        scaling_factor = np.sqrt(self.n_photons / (np.abs(wavefront)**2).sum())
        self.wavefront = scaling_factor * wavefront
        self._propagateWavefront()

class CircularProbe(Probe):
    r"""Create a Probe object using the exit wave from a circular aperture.

        Also see documentation for `Probe` information on other parameters and attributes.

        Parameters
        ----------
        radius : float
            Radius of the aperture (in m).
        *args : List
            Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
        **kwargs : dict
            Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

        Attributes
        ----------
        radius : see Parameters

        See also
        --------
        Probe

        Notes
        -----
        Uses the :math:`\text{circ}` function for the aperture. If :math:`r` is the radius of the aperture and
        :math:`a` is the distance of a point :math:`(x,y)` from the center of the aperture, this is defined as:

        .. math::

            \text{circ}(a) =    \left\{
                                    \begin{array}{ll}
                                        0 \text{ if } a < r\\
                                        0.5 \text{ if } a =r\\
                                        1 \text{ otherwise}
                                    \end{array}
                                \right.

        """
    def __init__(self, *args: List,
                 radius: float,
                 **kwargs: dict):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self._calculateWavefront()

    def _calculateWavefront(self):
        """Calculates and propagates the exit wave from a circular aperture of the supplied radius.
        """

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
    r"""Create a Probe object with a gaussian wavefront.

    Also see documentation for `Probe` information on other parameters and attributes.

    Parameters
    ----------
    sigma_x : float
        Spread (standard deviation) of the gaussian in the x-direction (in m).
    sigma_y : float
        Spread of the gaussian in the y-direction (in m).
    theta : float, optional
        Angle with which to clockwise rotate the primary axes of the 2d gaussian after generation.
    *args : List
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : dict
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    sigma_x, sigma_y, theta : see Parameters

    See also
    --------
    Probe
    utils.generalized2dGaussian

    Notes
    -----
    The probe wavefront is generated with constant phase and with intensity calculated according to the equation [2]_:

        .. math:: f(x,y) = A \exp\left(-a(x-x_0)^2 - 2b(x-x_0)(y-y_0) - c(y-y_0)^2\right),

    where :math:`A` is the amplitude, :math:`a=\cos^2\theta/(2\sigma_x^2) + \sin^2\theta/(2\sigma_y^2)`, with
    :math:`b=-\sin 2\theta/(4\sigma_x^2) + \sin 2\theta/(4\sigma_y^2)`, and with
    :math:`c=\sin^2\theta/(2\sigma_x^2) + \cos^2\theta/(2\sigma_y^2)`.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

    """
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
        """Calculate the probe wavefront with constant phase and with a gaussian intensity. Propagate the wavefront."""
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
    """Create a gaussian probe, then modulate it with a speckle pattern.

    First calculates a gaussian wavefront, then multiplies it with a speckle pattern for the modulation. The speckle
    pattern is randomly generated and varies between instances of `GaussianSpeckledProbe`.

    Also see documentation for `Probe` information on other parameters and attributes.

    Parameters
    ----------
    sigma_x, sigma_y, theta : see the documentation for the corresponding parameters in GaussianProbe.
    speckle_window_npix : int
        Aperture size used to generate a speckled pattern (in pixels).
    *args : List
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : dict
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    sigma_x, sigma_y, theta, speckle_window_npix : see Parameters

    Notes
    -----
    In ptychography, probe structures with *phase diversity* (i.e. with large angular ranges) are known to reduce
    the dynamic range of the detected signal by spreading the zero-order light over an extended area of the detector
    [3]_. This makes the reconstruction more robust and increases the rate of convergence of the reconstruction
    algorithm. This is generally accomplished experimentally via a diffuser, which introduces speckles into
    probe wavefront. The *speckled probe* used here thus emulates a gaussian probe wavefront modulated by a diffuser.

    Additionally, for successful near-field ptychography [4]_, it is important to ensure that the diffraction patterns
    generated in consecutive scan positions are sufficiently diverse (or different). This can be accomplished by
    using a probe structure that varies rapidly as we traverse the spatial structure. A speckle pattern accomplishes
    this and can be effectively used for near-field ptychography.

    See also
    --------
    GaussianProbe
    Probe
    utils.getSpeckle

    References
    ----------
    .. [3] Morrison, G. R., Zhang, F., Gianoncelli, A. & Robinson, I. K. X-ray ptychography using randomized zone
        plates. Opt. Express 26, 14915 (2018).
    .. [4] Richard M. Clare, Marco Stockmar, Martin Dierolf, Irene Zanette, and Franz Pfeiffer,
        "Characterization of near-field ptychography," Opt. Express 23, 19728-19742 (2015)
    """
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
        """Calculate a wavefront with gaussian intensity, then modulate the wavefront using a speckle pattern."""
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
    r"""Create a Probe object containing an airy pattern obtained due to a lens and a circular aperture.

    Given a lens of focal length :math:`f`, if we place a circular aperture at a distance :math:`f` *before*
    the lens, then we obtain an airy pattern at the focus *after* the lens [5]_. The `FocusCircularProbe` class
    simulates such an experimental condition and populates the probe wavefront with the airy pattern obtained. It
    then allows for subsequent defocus of the wavefront.

    The probe wavefront can be generated by either supplying *both* `focal_length` and `aperture_radius`,
    or by supplying `focus_radius_npix`. If `focus_radius_npix` is supplied, then any supplied `focal_length` and
    `aperture_radius` are ignored; the `focal_length` is set to a default value of 10 m, and the aperture radius is
    calculated such that the generated airy pattern in the wavefront has its first minimum at `focus_radius_npix`
    pixels from the center of the pattern.

    If the size of the probe array is not large enough, then the probe generated contains significant aliasing. To
    reduce this aliasing, we can artificially oversample the probe wavefront by increasing the size of the
    probe array. Using this oversampled array to calculate the probe wavefront, we reduce the aliasing. We can then
    the central :math:`\text{npix}\times\text{npix}` pixels of the oversampled array to get the desired wavefront.

    See the `Probe` documentation for information on the other parameters and attributes.

    Notes
    -----
    To describe the steps used for the wavefront calculation, we denote the focal length as :math:`f`, the aperture
    radius as :math:`r_a`, the number of pixels in the probe array as :math:`N`, the pixel pitch at the probe focus
    as :math:`\Delta p_f`, the pixel pitch at the aperture as :math:`\Delta p_a`.

    #. If `focus_radius_npix` (:math:`N_f`) is supplied, then, assume that the resolution (:math:`\delta_r`) at the
        sample plane is equal to the pixel pitch :math:`\delta_r = \Delta p_f`, and that :math:`f=10` m, use the Rayleigh
        resolution criterion to calculate the aperture radius (Equations 4.161-4.163 in [6]_):

            .. math:: r_a = \frac{1.22 \lambda \pi f}{2 \pi N_f \delta_r}

        Otherwise, if the `aperture_radius` is supplied, use the supplied value.

    #. Calculate the pixel pitch at the aperture plane (which is at a distance :math:`f` before the lens):

            .. math:: \Delta p_a = \frac{\lambda f}{N \Delta p_f}

    #. Use the :math:`\text{circ}` function with the aperture radius :math:`r_a` as the wavefront at the aperture
        plane. See documentation for `CircularProbe` for more on the `\text{circ}` function.

    #. Use a fourier transform of this wavefront to get the airy wavefront at the focus plane.

    #. If necessary, propagate by `defocus_dist` to get the wavefront at the sample plane.

    Parameters
    ----------
    focal_length : float, optional
        Focal length of the lens used to generate the probe (in m).
    aperture_radius : float, optional
        Radius of the circular aperture placed *before* the lens, at a distance `focal_length` from the lens.
    focus_radius_npix : float, optional
        Distance (in pixels) of the first minimum of the airy pattern from the center of the pattern.
    oversampling : bool
        Whether to use oversampling to reduce aliasing in the probe wavefront. Default value is `True`.
    oversampling_npix : int
        Number of pixels in each side of the oversampled array. Default value is 1024. If
        :math:`\text{npix} > \text{oversampling_npix}`, then `oversampling_npix` is set to `npix`.
    *args : List
        Parameters required to initialize a `Probe` object. Supplied to the parent `Probe` class.
    **kwargs : dict
        Optional arguments for a `Probe` object. Supplied to the parent `Probe` class.

    Attributes
    ----------
    focal_length, aperture_radius, focus_radius_npix, oversampling, oversampling_npix : see Parameters

    See also
    --------
    Probe

    References
    ----------
    .. [5] Schmidt, J. D. & Jason D.Schmidt. Numerical Simulation of optical wave propagation (pp. 55-64). (1975).
        doi:10.1117/3.866274
    .. [6] Jacobsen, C. & Kirz, J. X-ray Microscopy_2018_01_15. (2017).

    .. todo::

        Allow for combinations of `focus_radius_npix` with either `focal_length` or `aperture_radius`.

    """
    def __init__(self, *args: List,
                 focal_length: Optional[float] = None,
                 aperture_radius: Optional[float] = None,
                 focus_radius_npix: Optional[float] = None,
                 oversampling: bool = True,
                 oversampling_npix: int = 1024,
                 **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.focus_radius_npix = focus_radius_npix
        self.focal_length = focal_length
        self.aperture_radius = aperture_radius
        self.oversampling = oversampling
        self.oversampling_npix = oversampling_npix
        self._calculateWavefront()

    def _calculateWavefront(self) -> None:
        """Calculating the airy pattern then propagating it by `defocus_dist`."""
        if self.focus_radius_npix is not None:
            print('Warning: if focus_radius_npix float is supplied, '
                  + 'we ignore any supplied focal_length and aperture radius.')
            self.aperture_radius = None
            self.focal_length = None
            # Assuming resolution equal to pixel pitch and focal length of 10.0 m.
            delta_r = self.pixel_pitch
            focal_length = 10.0
            focus_radius = delta_r * self.focus_radius_npix
            # note that jinc(1,1) = 1.22 * np.pi
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

