#Author - Saugat Kandel
# coding: utf-8


import numpy as np
from typing import Tuple

class PropagationTypes:
    FARFIELD: int = 0
    TRANSFER_FUNCTION: int = 1
    IMPULSE_RESPONSE: int = 2


def checkPropagationType(support_npix: int,
                         wavelength: float,
                         prop_dist: float,
                         source_pixel_size: float) -> int:
    fresnel_number = (support_npix * source_pixel_size)** 2 / (wavelength * prop_dist)
    print('Fresnel number is', fresnel_number)
    if fresnel_number < 0.1:
        return PropagationTypes.FARFIELD
    if source_pixel_size > (wavelength * prop_dist):
        return PropagationTypes.TRANSFER_FUNCTION
    return PropagationTypes.IMPULSE_RESPONSE


def propagate(wavefront: np.ndarray,
              pixel_size: float,
              wavelength: float,
              prop_dist: float) -> Tuple[np.ndarray, float]:
    """Choose between Fraunhofer propagation, Transfer function propagation, and Impulse Response propagation.

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

    This function follows the convention of shifting array in real space before performing the
    Fourier transform.

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
    return out, new_pixel_size



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
