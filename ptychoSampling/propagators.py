#Author - Saugat Kandel
# coding: utf-8


import numpy as np
from typing import Tuple

class PropagationTypes:
    """Contains convenient flags.
    """
    FARFIELD: int = 0
    TRANSFER_FUNCTION: int = 1
    IMPULSE_RESPONSE: int = 2


def checkPropagationType(support_npix: int,
                         wavelength: float,
                         prop_dist: float,
                         source_pixel_size: float) -> int:
    """Check the parameters to decide whether to use farfield, transfer function, or impulse response propagation.

    Calculates the Fresnel number to decide whether to use nearfield or farfield propagation. Then,
    for the near-field, we use the critical sampling criterion to decide whether to use the transfer function (TF)
    method (if :math:`\Delta x > \lambda z/L`) or the impulse response (IR) method (if :math:`\Delta x < \lambda
    z/L`) [1]_.

    Notes
    -----
    Following [1]_, we calculate the Fresnel number :math:`F_N = L^2/\lambda z`, where :math:`L` is the source
    support size, :math:`\lambda` is the wavelength, and :math:`z` is the propagation distance. If :math:`F_N < 0.1`,
    we can safely use the Fraunhofer (far-field) method, otherwise we need to use one of the near-field methods.

    To choose the near-field method, we follow the criterion in [1]_ (Pg. 79). Defining :math:`a = \lambda z/L`, we get:

    * If :math:`\Delta x > a`, then the TF approach works well with some loss of observation plane support.
    * If :math:`\Delta x = a`, then TF approach gives best use of bandwidth and spatial support.
    * If :math:`\Delta x < a`, then IR approach with loss of available source bandwidth and artifacts.

    Parameters
    ----------
    support_npix : int
        Number of pixels in each side of the support.
    wavelength : float
        Wavelength of the source wavefront (in m).
    prop_dist : float
        Propagation distance (in m).
    source_pixel_size : float
        Pixel pitch (in m).

    Returns
    -------
    out : int
        Can have values of 0, 1, or 2, as defined in `PropagationTypes`.

    See also
    --------
    PropagationTypes

    References
    ----------
    .. [1] Voelz, D. "Computational Fourier Optics: A MATLAB Tutorial". doi:https://doi.org/10.1117/3.858456

    """
    source_support_size = support_npix * source_pixel_size
    fresnel_number = source_support_size** 2 / (wavelength * prop_dist)
    print('Fresnel number is', fresnel_number)
    if fresnel_number < 0.1:
        return PropagationTypes.FARFIELD
    if source_pixel_size >= wavelength * prop_dist / source_support_size:
        return PropagationTypes.TRANSFER_FUNCTION
    return PropagationTypes.IMPULSE_RESPONSE


def propagate(wavefront: np.ndarray,
              pixel_size: float,
              wavelength: float,
              prop_dist: float) -> Tuple[np.ndarray, float]:
    """Choose between Fraunhofer propagation, Transfer function propagation, and Impulse Response propagation.

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
    wavefront : array_like(complex)
        Array that contains the wavefront to be propagated.
    pixel_size : float
        Pixel pitch (in m).
    wavelength : float
        Wavelength of the wavefront to be propagated (in m).
    prop_dist : float
        Propagation distance (in m).

    Returns
    -------
    out : ndarray(complex)
        Output array of the same shape as `wavefront`.
    new_pixel_size : float
        Pixel size after wavefront propagation. Only changes in the farfield case.

    References
    ----------
    .. [2] Voelz, D. "Computational Fourier Optics: A MATLAB Tutorial". doi:https://doi.org/10.1117/3.858456
    """

    prop_fns = {PropagationTypes.FARFIELD: propFF,
                PropagationTypes.TRANSFER_FUNCTION: propTF,
                PropagationTypes.IMPULSE_RESPONSE: propIR}

    support_npix = np.max(wavefront.shape)
    propagation_type = checkPropagationType(support_npix, wavelength, prop_dist, pixel_size)

    out, new_pixel_size = prop_fns[propagation_type](wavefront, pixel_size, wavelength, prop_dist)
    return out, new_pixel_size

def propIR(wavefront: np.ndarray, 
           pixel_size: float,
           wavelength: float, 
           prop_dist: float) -> Tuple[np.ndarray, float]:
    """Propagation of the supplied wavefront using the Impulse Response function.

    This function follows the convention of shifting array in real space before performing the Fourier transform.

    Parameters
    ----------
    wavefront : array_like(complex)
        Wavefront to be propagated.
    pixel_size : float
        Pixel pitch (in m).
    wavelength : float
        Wavelength of the wavefront to be propagated (in m).
    prop_dist : float
        Propagation distance (in m).

    Returns
    -------
    out : ndarray(complex)
        Output array of the same shape as `wavefront`.
    pixel_size : float
        Returns the input pixel size. This is for consistency with far-field propagation results.
    """
    print('Using the Impulse Response method.')
    npix = np.shape(wavefront)[0]

    k = 2 * np.pi / wavelength

    # real space coordinates
    x = np.arange(-npix // 2, npix // 2) * pixel_size

    # quadratic phase factor
    h = np.exp(1j * k / (2 * prop_dist) * (x**2 + x[:,np.newaxis]**2))

    H = np.fft.fft2(np.fft.fftshift(h), norm='ortho')
    wavefront_ft = np.fft.fft2(np.fft.fftshift(wavefront), norm='ortho')
    out_beam = np.fft.ifftshift(np.fft.ifft2(H * wavefront_ft, norm='ortho'))
    return out_beam, pixel_size



def propFF(wavefront: np.ndarray,
           pixel_size: float,
           wavelength: float,
           prop_dist: float) -> Tuple[np.ndarray, float]:
    """Fraunhofer propagation of the supplied wavefront.

    Note that the side length in the  observation plane is no longer the same as the side length of the input plane.

    Parameters
    ----------
    wavefront : array_like(complex)
        Wavefront to be propagated.
    pixel_size : float
        Pixel pitch (in m).
    wavelength : float
        Wavelength of the wavefront to be propagated (in m).
    prop_dist : float
        Propagation distance (in m).

    Returns
    -------
    out : ndarray(complex)
        Output array of the same shape as `wavefront`.
    new_pixel_size : float
        New pixel size after far-field propagation.
    """
    print('Using Fraunhofer diffraction method.')
    npix = np.shape(wavefront)[0]
    k = 2 * np.pi / wavelength

    # reciprocal space pixel size
    rdx = wavelength * prop_dist / (npix * pixel_size)

    # reciprocal space coordinates
    x = np.arange(-npix // 2, npix // 2) * rdx
    y = np.arange(-npix // 2, npix // 2)[:,np.newaxis] * rdx

    # quadratic phase factor
    q = np.exp(1j * k /(2 * prop_dist) * (x**2 + y**2))
    out = q * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(wavefront), norm='ortho'))
    return out, rdx



def propTF(wavefront: np.ndarray,
           pixel_size: float,
           wavelength: float,
           prop_dist: float) -> Tuple[np.ndarray, float]:
    """Propagation of the supplied wavefront using the Transfer Function function.

    Parameters
    ----------
    wavefront : array_like(complex)
        Wavefront to be propagated.
    pixel_size : float
        Pixel pitch (in m).
    wavelength : float
        Wavelength of the wavefront to be propagated (in m).
    prop_dist : float
        Propagation distance (in m).

    Returns
    -------
    out : ndarray(complex)
        Output array of the same shape as `wavefront`.
    pixel_size : float
        Returns the input pixel size. This is for consistency with far-field propagation results.
    """
    print('Using Transfer Function method.')
    M, N = np.shape(wavefront)

    # wave number
    k = 2 * np.pi / wavelength

    # reciprocal space pixel size
    rdx = wavelength * prop_dist / (N * pixel_size)

    x = np.arange(-N // 2, N // 2) * rdx
    y = np.arange(-M // 2, M // 2)[:, np.newaxis] * rdx

    H = np.exp(-1j * k / (2. * prop_dist) * (x ** 2 + y ** 2))
    U1 = np.fft.fft2(np.fft.fftshift(wavefront), norm='ortho')
    U2 = U1 * np.fft.fftshift(H)
    out = np.fft.ifftshift(np.fft.ifft2(U2, norm='ortho'))
    return out, pixel_size
