#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import dataclasses as dt
from ptychoSampling.logger import logger
from enum import Enum

def wfft2(array):
    u1 = np.fft.fftshift(array, axes=(-1,-2))
    u2 = np.fft.fft2(u1, axes=(-1,-2), norm='ortho')
    u3 = np.fft.ifftshift(u2, axes=(-1, -2))
    return u3

def wifft2(array):
    u1 = np.fft.ifftshift(array, axes=(-1,-2))
    u2 = np.fft.ifft2(u1, axes=(-1, -2), norm='ortho')
    u3 = np.fft.fftshift(u2, axes=(-1, -2))
    return u3


class PropagationType(Enum):
    """Contains convenient flags.
    """
    FARFIELD: int = 0
    TRANSFER_FUNCTION: int = 1
    IMPULSE_RESPONSE: int = 2
    UNSUPPORTED: int = 3

#@dt.dataclass
#class Wavefront:
#    array: np.ndarray
#    wavelength: float = None
#    pixel_size: float = None
#
#    def update(self, **kwargs) -> 'Wavefront':
#        new_wavefront = dt.replace(self, **kwargs)
#        return new_wavefront
class Wavefront(np.ndarray):

    def __new__(cls, input_array: np.ndarray,
                wavelength: float=None,
                pixel_size: float=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.wavelength = wavelength
        obj.pixel_size = pixel_size
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.wavelength = getattr(obj, 'wavelength', None)
        self.pixel_size = getattr(obj, 'pixel_size', None)

    #@staticmethod
    def fft2(self):#wv: 'Wavefront'):
        out = np.fft.fft2(self, norm='ortho')
        return Wavefront(out, wavelength=self.wavelength, pixel_size=self.pixel_size)

    def ifft2(self):#wv: 'Wavefront'):
        out = np.fft.ifft2(self, norm='ortho')
        return Wavefront(out, wavelength=self.wavelength, pixel_size=self.pixel_size)

    def propFF(self, apply_phase_factor: bool = False,
               prop_dist: float = None,
               reuse_phase_factor: bool = False,
               quadratic_phase_factor: np.ndarray = None,
               backward=False) -> 'Wavefront':
        """Fraunhofer propagation of the supplied wavefront.

        Parameters
        ----------
        apply_phase_factor: bool
            Whether to apply the quadratic phase factor. The phase factor is not necessary if we only want the wavefront
            at the detector plane to calculate the intensity patterns recorded. If this is set to `True`, then we need to
            supply either the quadratic phase factor for the `quadratic_phase_factor` parameter, or a `Wavefront`
            object with the `pixel_size` and `wavelength` attributes as well as the `prop_dist` function parameter.
        prop_dist : float
            Propagation distance (in m). Only required when `apply_quadratic_phase_factor` is `True` and
            `quadratic_phase_factor` is `None`.
        reuse_phase_factor : bool
            Whether to reuse the quadratic phase factor supplied through the `quadratic_phase_factor` parameter. Default
            is `False`. If set to `True`, then the quadratic phase factor must be supplied through the
            `quadratic_phase_factor` parameter.
        quadratic_phase_factor : array_like(complex)
            Use the supplied quadratic phase factor. Only required when `apply_quadratic_phase_factor` is `True`. The
            function either reuses or mutates (updates) this array, depending on the 'reuse_quadratic_phase_factor`
            parameter. This parameter can be used when we want to avoid recalculating the phase factor for multiple
            propagations for the same experimental parameters.
        backward : bool
            Propagate backward instead of forward.
        Returns
        -------
        out_wavefront : Wavefront
            Output wavefront.
        """

        new_pixel_size = None
        npix = self.shape[0]

        if None not in [prop_dist, self.wavelength, self.pixel_size]:
            new_pixel_size = self.wavelength * prop_dist / (npix * self.pixel_size)

        if apply_phase_factor:
            if quadratic_phase_factor is None:
                quadratic_phase_factor = np.zeros(self.shape, dtype='complex64')
            if not reuse_phase_factor:
                k = 2 * np.pi / self.wavelength

                # reciprocal space pixel size
                rdx = self.pixel_size if backward else new_pixel_size

                # reciprocal space coordinates
                x = np.arange(-npix // 2, npix // 2) * rdx
                y = np.arange(-npix // 2, npix // 2)[:,np.newaxis] * rdx

                # quadratic phase factor
                q = np.fft.fftshift(np.exp(1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
                quadratic_phase_factor[:] = q[:]
        else:
            quadratic_phase_factor = np.ones_like(self)

        if backward:
            new_wavefront = (self / quadratic_phase_factor).ifft2()
        else:
            new_wavefront = self.fft2() * quadratic_phase_factor

        new_wavefront.pixel_size = new_pixel_size
        return new_wavefront

    def propTF(self,
               prop_dist: float = None,
               reuse_transfer_function: bool = False,
               transfer_function: np.ndarray = None,
               backward: bool = False) -> 'Wavefront':
        """Propagation of the supplied wavefront using the Transfer Function function.

        This propagation method is also referred to as *angular spectrum* or *fresnel* propagation.

        Parameters
        ----------
        reuse_transfer_function : bool
            Reuse provided transfer function.
        transfer_function : array_like(complex)
            Transfer function after fftshift of reciprocal coordinates.
        prop_dist : float
            Propagation distance (in m).
        backward : bool
            Backward propagation.
        Returns
        -------
        wavefront : Wavefront
            Wavefront after propagation
        """
        if transfer_function is None:
            transfer_function = np.zeros(self.shape, dtype='complex64')
        if not reuse_transfer_function:
            npix = self.shape[0]
            k = 2 * np.pi / self.wavelength

            # reciprocal space pixel size
            rdx = self.wavelength * prop_dist / (npix * self.pixel_size)

            # reciprocal space coords
            x = np.arange(-npix // 2, npix // 2) * rdx
            y = np.arange(-npix // 2, npix // 2)[:, np.newaxis] * rdx

            H = np.fft.fftshift(np.exp(-1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
            transfer_function[:] = np.fft.fftshift(H)[:]

        if backward:
            out = (self.ifft2() / transfer_function).fft2()
        else:
            out = (transfer_function * self.fft2()).ifft2()

        return out

    def propIR(self,
               prop_dist: float = None,
               reuse_transfer_function=False,
               transfer_function=None) -> 'Wavefront':
        """Propagation of the supplied wavefront using the Impulse Response function.

        This function follows the convention of shifting array in real space before performing the Fourier transform.

        Parameters
        ----------
        prop_dist : float
            Propagation distance (in m).
        reuse_transfer_function : bool
            Whether to reuse the transfer function supplied through the `transfer_function` parameter. Default is `False`.
            If set to `True`, then the transfer function must be supplied through the `transfer_function` parameter.
        transfer_function: array_like(complex)
            If supplied, then the function either either reuses or mutates (updates) this transfer function, depending on
            the `reuse_transfer_function` parameter. Note that this should be provided without any fftshift

        Returns
        -------
        wavefront_out : Wavefront
            Output wavefront object. For near-field propagation, the pixel size remains the same.

        .. todo::

            Improve the error output.

        """
        logger.warning("I have not ever seen the impulse response method mentioned in x-ray imaging literature. "
                       + "This is not in use in the ptychography simulations or in the reconstruction codes.")
        ft = self.fft2()
        if transfer_function is None:
            transfer_function = np.zeros(self.shape, dtype='complex64')
        if not reuse_transfer_function:
            npix = self.shape[0]
            k = 2 * np.pi / self.wavelength

            # real space coordinates
            x = np.arange(-npix // 2, npix // 2) * self.pixel_size

            # quadratic phase factor (or impulse)
            h = np.exp(1j * k / (2 * prop_dist) * (x ** 2 + x[:, np.newaxis] ** 2))
            H = np.fft.fft2(np.fft.fftshift(h), norm='ortho')

            transfer_function[:] = H[:]
        try:
            out = (transfer_function * ft).ifft2()
        except Exception as e:
            e2 = ValueError('Invalid transfer function')
            logger.error([e, e2])
            raise e2 from e
        return out



def checkPropagationType(wavelength: float,
                         prop_dist: float,
                         source_support_size: float,
                         source_pixel_size: float,
                         max_feature_size: float = None) -> PropagationType:
    """Check the parameters to decide whether to use farfield, transfer function, or impulse response propagation.
    This is an experimental feature.

    Calculates the Fresnel number to decide whether to use nearfield or farfield propagation. Then,
    for the near-field, we use the critical sampling criterion to decide whether to use the transfer function (TF)
    method (if :math:`\Delta x > \lambda z/L`) or the impulse response (IR) method (if :math:`\Delta x < \lambda
    z/L`) [1]_.

    Notes
    -----
    Following [1]_, we calculate the Fresnel number :math:`F_N = w^2/\lambda z`, where :math:`w` is the half-width of
    the maximum feature size in the source plane (e.g the half-width of a square aperture, or the radius of a
    circular aperture), :math:`\lambda` is the wavelength, and :math:`z` is the propagation distance. If
    :math:`F_N < 0.1`, we can safely use the Fraunhofer (far-field) method, otherwise we need to use one of the
    near-field methods.

    When the maximum feature size is not provided, the function uses a minimal estimate---twice the pixel size at the
    source plane, or two pixels. This assumes that Fraunhofer propagation is the ost likely propagation method.

    To choose the near-field method, we follow the criterion in [1]_ (Pg. 79). Defining :math:`a = \lambda z/L`,
    where :math:`L` is the source support size, we get:

    * If :math:`\Delta x > a`, then the TF approach works well with some loss of observation plane support.
    * If :math:`\Delta x = a`, then TF approach gives best use of bandwidth and spatial support.
    * If :math:`\Delta x < a`, then IR approach with loss of available source bandwidth and artifacts.

    Parameters
    ----------
    wavelength : float
        Wavelength of the source wavefront (in m).
    prop_dist : float
        Propagation distance (in m).
    source_support_size : float
        Support size (in m).
    source_pixel_size : float
        Pixel pitch (in m).
    max_feature_size : float, optional
        Maximum feature size (e.g. diameter of a circular aperture) (in m). For the default value (`None`),
        the function assumes that the features are two pixels wide, at maximum.

    Returns
    -------
    out : PropagationType
        Type of propagation, as defined in `PropagationType`.

    See also
    --------
    PropagationType

    References
    ----------
    .. [1] Voelz, D. "Computational Fourier Optics: A MATLAB Tutorial". doi:https://doi.org/10.1117/3.858456

    """
    logger.warn("The checkPropagationType function is experimental and should not be relied upon.")
    max_feature_size = 2 * source_pixel_size if max_feature_size is None else max_feature_size
    feature_half_width = max_feature_size / 2
    fresnel_number = feature_half_width ** 2 / (wavelength * prop_dist)
    if fresnel_number > 50:
        prop_type = PropagationType.UNSUPPORTED
    elif fresnel_number < 0.1:
        prop_type = PropagationType.FARFIELD
    elif source_pixel_size >= wavelength * prop_dist / source_support_size:
        prop_type = PropagationType.TRANSFER_FUNCTION
    else:
        prop_type = PropagationType.IMPULSE_RESPONSE
    logger.info(f'Fresnel number is {fresnel_number} and propagation type is {prop_type.name}')
    return prop_type


def propagate(wavefront: Wavefront,
              prop_dist: float,
              max_feature_size: float = None) -> Wavefront:
    """Choose between Fraunhofer propagation, Transfer function propagation, and Impulse Response propagation.
    This is an experimental feature.

    Notes
    -----
    In typical references (e.g. [2]_), the propagation is scaled appropriately to propagate the *field* values at
    the wavefront. For example, for a source wavefront :math:`U_{in}`, we can define the *irradiance*
    :math:`I_{in} = |U_{in}|^2`, and the total optical power

    .. math:: P_{in} = \sum_i^M\sum_j^N I_{in}\Delta x_{in} \Delta y_{in}

    where :math:`(i,j)` index the pixels along the :math:`(x,y)` axes, and :math:`(\Delta x_{in}, \Delta y_{in})`
    are the pixel sizes. After propagation, we obtain :math:`U_{out}, I_{out}, P_{out}` , the output wavefront,
    irradiance and total optical power respectively. A typical propagation algorithm scales these quantities so that
    :math:`P_{in}=P_{out}`, which means that we need to keep track of both the wavefront as well as the respective pixel
    sizes.

    Alternatively, we use wavefronts scaled so that :math:`\sum_i^M\sum_j^N I_{in} = \sum_i^M\sum_j^M I_{out}`. With
    this scaling, we can directly use the orthonormalized fft without worrying about the additional scaling factors.
    The algorithms implemented here use this approach.

    Parameters
    ----------
    wavefront : Wavefront
        Namedtuple that contains the wavefront to be propagated.
    prop_dist : float
        Propagation distance (in m).
    max_feature_size : float
        Maximum feature size (e.g. diameter of a circular aperture) (in m). For function behavior when
        `max_feature_size` is not provided, see documentation for `checkPropagationType`.

    Returns
    -------
    wavefront : Wavefront
        Output wavefront object. Note that the pixel size of the wavefront changes for the farfield case.

    See Also
    --------
    checkPropagationType

    References
    ----------
    .. [2] Voelz, D. "Computational Fourier Optics: A MATLAB Tutorial". doi:https://doi.org/10.1117/3.858456
    """
    logger.warning("The propagate function is an experimental feature.")
    prop_fns = {PropagationType.FARFIELD: wavefront.propFF,
                PropagationType.TRANSFER_FUNCTION: wavefront.propTF,
                PropagationType.IMPULSE_RESPONSE: wavefront.propIR}

    support_npix = np.max(wavefront.shape)
    support_size = support_npix * wavefront.pixel_size
    propagation_type = checkPropagationType(wavelength=wavefront.wavelength,
                                            prop_dist=prop_dist,
                                            source_support_size=support_size,
                                            source_pixel_size=wavefront.pixel_size,
                                            max_feature_size=max_feature_size)
    if propagation_type == PropagationType.UNSUPPORTED:
        e = ValueError("Fresnel number too high. Propagation type is not supported.")
        logger.error(e)
        raise e

    wavefront = prop_fns[propagation_type](prop_dist=prop_dist)
    return wavefront

def propIR(wavefront: Wavefront,
           prop_dist: float = None,
           reuse_transfer_function = False,
           transfer_function = None) -> Wavefront:
    """Propagation of the supplied wavefront using the Impulse Response function.

    This function follows the convention of shifting array in real space before performing the Fourier transform.

    Parameters
    ----------
    wavefront : Wavefront
        Wavefront object that contains the wavefront to be propagated, the source pixel size, and the wavelength.
    prop_dist : float
        Propagation distance (in m).
    reuse_transfer_function : bool
        Whether to reuse the transfer function supplied through the `transfer_function` parameter. Default is `False`.
        If set to `True`, then the transfer function must be supplied through the `transfer_function` parameter.
    transfer_function: array_like(complex)
        If supplied, then the function either either reuses or mutates (updates) this transfer function, depending on
        the `reuse_transfer_function` parameter. Note that this should be provided without any fftshift

    Returns
    -------
    wavefront_out : Wavefront
        Output wavefront object. For near-field propagation, the pixel size remains the same.

    .. todo::

        Improve the error output.

    """
    logger.warning("I have not ever seen the impulse response method mentioned in x-ray imaging literature. "
                   + "This is not in use in the ptychography simulations or in the reconstruction codes.")
    ft = np.fft.fft2(wavefront, norm='ortho')
    if transfer_function is None:
        transfer_function = np.zeros(wavefront.array.shape, dtype='complex64')
    if not reuse_transfer_function:
        npix = np.shape(wavefront.array)[0]
        k = 2 * np.pi / wavefront.wavelength

        # real space coordinates
        x = np.arange(-npix // 2, npix // 2) * wavefront.pixel_size

        # quadratic phase factor (or impulse)
        h = np.exp(1j * k / (2 * prop_dist) * (x**2 + x[:,np.newaxis]**2))
        H = np.fft.fft2(np.fft.fftshift(h), norm='ortho')

        transfer_function[:] = H[:]
    try:
        out_array = np.fft.ifft2(transfer_function * ft, norm='ortho')
    except Exception as e:
        e2 = ValueError('Invalid transfer function')
        logger.error([e, e2])
        raise e2 from e
    return wavefront.update(array=out_array)

def propFF(wavefront: Wavefront,
           apply_phase_factor: bool = True,
           prop_dist: float = None,
           reuse_phase_factor: bool = False,
           quadratic_phase_factor: np.ndarray = None,
           backward=False) -> Wavefront:
    """Fraunhofer propagation of the supplied wavefront.

    Parameters
    ----------
    wavefront : Wavefront
        Wavefront to be propagated.
    apply_phase_factor: bool
        Whether to apply the quadratic phase factor. The phase factor is not necessary if we only want the wavefront
        at the detector plane to calculate the intensity patterns recorded. If this is set to `True`, then we need to
        supply either the quadratic phase factor for the `quadratic_phase_factor` parameter, or a `Wavefront`
        object with the `pixel_size` and `wavelength` attributes as well as the `prop_dist` function parameter.
    prop_dist : float
        Propagation distance (in m). Only required when `apply_quadratic_phase_factor` is `True` and
        `quadratic_phase_factor` is `None`.
    reuse_phase_factor : bool
        Whether to reuse the quadratic phase factor supplied through the `quadratic_phase_factor` parameter. Default
        is `False`. If set to `True`, then the quadratic phase factor must be supplied through the
        `quadratic_phase_factor` parameter.
    quadratic_phase_factor : array_like(complex)
        Use the supplied quadratic phase factor. Only required when `apply_quadratic_phase_factor` is `True`. The
        function either reuses or mutates (updates) this array, depending on the 'reuse_quadratic_phase_factor`
        parameter. This parameter can be used when we want to avoid recalculating the phase factor for multiple
        propagations for the same experimental parameters.
    backward : bool
        Propagate backward instead of forward.
    Returns
    -------
    out_wavefront : Wavefront
        Output wavefront.
    """

    new_pixel_size = None
    npix = wavefront.array.shape[0]

    if None not in [prop_dist, wavefront.wavelength, wavefront.pixel_size]:
        new_pixel_size = wavefront.wavelength * prop_dist / (npix * wavefront.pixel_size)

    if apply_phase_factor:
        if quadratic_phase_factor is None:
            quadratic_phase_factor = np.zeros(wavefront.array.shape, dtype='complex64')
        if not reuse_phase_factor:
            k = 2 * np.pi / wavefront.wavelength

            # reciprocal space pixel size
            rdx = wavefront.pixel_size if backward else new_pixel_size

            # reciprocal space coordinates
            x = np.arange(-npix // 2, npix // 2) * rdx
            y = np.arange(-npix // 2, npix // 2)[:,np.newaxis] * rdx

            # quadratic phase factor
            q = np.fft.fftshift(np.exp(1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
            quadratic_phase_factor[:] = q[:]
    else:
        quadratic_phase_factor = np.ones_like(wavefront.array)

    if backward:
        new_array = np.fft.ifft2(wavefront.array / quadratic_phase_factor, norm='ortho')
    else:
        new_array = np.fft.fft2(wavefront.array, norm='ortho') * quadratic_phase_factor

    wavefront = wavefront.update(array=new_array, pixel_size=new_pixel_size)
    return wavefront

def propTF(wavefront: Wavefront,
           prop_dist: float = None,
           reuse_transfer_function: bool = False,
           transfer_function : np.ndarray = None,
           backward: bool = False) -> Wavefront:
    """Propagation of the supplied wavefront using the Transfer Function function.

    This propagation method is also referred to as *angular spectrum* or *fresnel* propagation.

    Parameters
    ----------
    wavefront : array_like(complex)
        Wavefront to be propagated.
    reuse_transfer_function : bool
        Reuse provided transfer function.
    transfer_function : array_like(complex)
        Transfer function after fftshift of reciprocal coordinates.
    prop_dist : float
        Propagation distance (in m).
    backward : bool
        Backward propagation.
    Returns
    -------
    wavefront : Wavefront
        Wavefront after propagation
    """
    if transfer_function is None:
        transfer_function = np.zeros(wavefront.array.shape, dtype='complex64')
    if not reuse_transfer_function:
        npix = np.shape(wavefront.array)[0]
        k = 2 * np.pi / wavefront.wavelength

        # reciprocal space pixel size
        rdx = wavefront.wavelength * prop_dist / (npix * wavefront.pixel_size)

        # reciprocal space coords
        x = np.arange(-npix // 2, npix // 2) * rdx
        y = np.arange(-npix // 2, npix // 2)[:, np.newaxis] * rdx

        H = np.fft.fftshift(np.exp(-1j * k / (2 * prop_dist) * (x ** 2 + y ** 2)))
        transfer_function[:] = np.fft.fftshift(H)[:]

    if backward:
        out_array = np.fft.fft2(np.fft.ifft2(wavefront.array) / transfer_function)
    else:
        out_array = np.fft.ifft2(transfer_function * np.fft.fft2(wavefront.array))

    return wavefront.update(array=out_array)