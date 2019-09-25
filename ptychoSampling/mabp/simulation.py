import numpy as np
import dataclasses as dt
from typing import Tuple, Any
from ptychoSampling.obj import Simulated3DCrystalCell, CustomObjFromArray, Obj
from ptychoSampling.probe import Probe, Probe3D, GaussianSpeckledProbe, CustomProbe3DFromArray
from ptychoSampling.grid import RectangleGrid, BraggPtychoGrid
from ptychoSampling.detector import Detector
from ptychoSampling.logger import logger
from scipy import ndimage

@dt.dataclass
class ObjParams:
    mesh_shape: Tuple[int, int, int] = (128, 128, 128) # y, x
    mod_const: float = 0.4
    border_shape: Tuple[int, int, int] = ((1,1), (10,10), (10,10))
    border_const: float = 0.0

@dt.dataclass
class Probe2DParams:
    """
    Assuming that the incident probe has 56 pixels along the y-direction. For a two_theta of 60 degrees (
    :math:`\pi/3` radians), this would give us, post-rotation, a y-length of
    :math:`56 * \cos{\pi/2 - \pi/3}\approx 64` pixels.
    """
    n_photons: float = 1e4
    width_npix: Tuple = (5, 5) # sigma_y, sigma_x
    speckle_window_npix: int = 20
    shape: Tuple[int, int] = (56, 56)
    pixel_size: Tuple[float, float] = (1, 1) # this shouldn't actually matter as long as it is non-zero

@dt.dataclass
class Scan2DGridParams:
    obj_pixel_size : Tuple[float, float]
    step_npix : dt.InitVar[Tuple[int, int]] = (4, 3) # y, z
    step_dist: Tuple[float, float] = dt.field(init=False)
    def __post_init__(self, step_npix):
        self.step_dist = tuple(np.array(self.obj_pixel_size) * step_npix)

@dt.dataclass
class DetectorParams:
    shape: Tuple[int, int] = (64, 64) # y, x, z
    obj_dist: float = 1.5
    pixel_size: Tuple[float, float] = (55e-6, 55e-6)

@dt.dataclass
class AngleGridParams:
    two_theta: float = 60 * np.pi / 180
    del_omega: float = 0.01 * np.pi / 180
    n_rc_angles: int = 15



class Simulation:
    # noinspection DuplicatedCode
    r"""Assumes that the same reference frame can be used (without rotation) for the object and the detector.

    See Fig. 5 in the paper by Kandel et al [1]_ for reference.

    Default experimental setup:
    * Detector is placed perpendicular to the y-axis of the object reference frame, *i.e.* the projection operator
        acts on the exit waves along the y-axis (exit waves are summed along the y-axis).
    * Ptychographic scan is along the y-z plane.
    * Incidence direction :math:`\mathbf{k_i}` and exit wave direction :math:`\mathbf{k_f}` all line on the x-y plane.
    * For consistency, we assume that numpy arrays index the 2D coordinates as :math:`(y, x)` (row-major format),
    and 3D coordinates as :math:`(y, x, z)`.
    * The axis of rotation for the probe is the z-axis.
    * In the z-direction, the probe width does not change even if we change the incident angle. This means that if,
    say, we have a incident probe of FWHM 10 pixels (for the intensity), ie :math:`\approx 20` pixels for the
    amplitude, then if we set the probe array to be :math:`>2\times FWHM= 40` pixels, we should have minimal aliasing
    in the Fourier transforms. As such, a detector with 64 pixel along the :math:`q_z`-direction should sufficiently
    capture the diffraction intensities.
    * (I am not sure if this is correct - ) Note that, in ptychography, the oversampling comes not from the
    application of a large enough support, but from the ptychographic overlap. For example, although the use of larger
    probe sizes can introduce aliasing during the Fourier transforms, this is compensated for by the increased probe
    overlap for larger probe sizes [2]_. As such, for the x-direction, we only need to ensure that the probe slice
    we take into consideration is large enough to account for all possible non-zero probe-object interaction in this
    direction. Ordinarily, this is a function of the object size, the probe width, and the angle of incidence of the
    probe (and is :math:`\leq` largest possible object  x-width). For simplicity, however, we can just take the
    x-width of the reconstruction volume, which is also the largest possible x-width.
    * For the y-axis, as we change two-theta from :math:`90^\circ` to smaller values, the number of scan positions
    that could give probe-object interaction also increases. If :math:`L_x` and :math:`L_y` are the width of the
    unknown reconstruction volume along the :math:`x` and :math:`y` directions respectively, then the


    Notes
    -----
    * We assume that we have knowledge of the object support along the exit wave direction. In this case,
    we accommodate  for possible reconstruction errors due to shot noise by adding a padding of 1 pixel.
    * In the other dimensions, we only have a general idea of the object dimensions. To represent this,
    we add a padding of 10 pixels.
    * Since we do not know, in advance, what the object shape is, we operate on the assumption that the entire
    reconstruction box is filled by the crystal cell. The ptychographic scan thus has to has to cover the entire
    reconstruction box. Afterwards, we can discard the scan positions where we do not see any diffracted photons.
    * For the detection, we can start out with a large detector (say :math:`128\times 128`. If all the recorded
    diffraction patterns are contained within a smaller central subset of the detector (say :math:`64\times64`
    pixels), then we can only use that as the data.

    References
    ----------
    .. [1] Using automatic differentiation as a general framework for ptychographic reconstruction.
    .. [2] Batey et all Upsampling paper.
    """

    def __init__(self,
                 wavelength: float = 1.5e-10,
                 obj: Obj = None,
                 probe_3d: Probe3D = None,
                 scan_grid: BraggPtychoGrid = None,
                 detector: Detector = None,
                 poisson_noise: bool = True,
                 upsampling_factor: int = 1) -> None:
        self.wavelength = wavelength
        self.upsampling_factor = upsampling_factor
        self.poisson_noise = poisson_noise

        two_theta = scan_grid.two_theta if scan_grid is not None else AngleGridParams().two_theta

        if detector is not None:
            logger.info("Using supplied detector info.")
            self.detector = detector
        else:
            logger.info("Creating new detector.")
            self.detector = Detector(**dt.asdict(DetectorParams()))
        det_3d_shape = (1, *self.detector.shape)

        obj_xz_nyquist_support = self.wavelength * self.detector.obj_dist / np.array(self.detector.pixel_size)
        obj_xz_pixel_size = obj_xz_nyquist_support / (np.array(self.detector.shape) * self.upsampling_factor)

        # The y pixel size is very ad-hoc
        obj_y_pixel_size = obj_xz_nyquist_support[1] / self.detector.shape[1]
        obj_pixel_size = (obj_y_pixel_size, *obj_xz_pixel_size)

        if obj is not None:
            logger.info("Using supplied object.")
            self.obj = obj
        else:
            self.obj = self.createObj(dt.asdict(ObjParams()), det_3d_shape, obj_pixel_size, upsampling_factor)

        probe_xz_shape = np.array(self.detector.shape) * self.upsampling_factor

        if probe_3d is not None:
            logger.info("Using supplied 3d probe values.")
            self.probe_3d = probe_3d
            # Note that the probe y shape cannot be determined from the other supplied parameters (I think).
            if (np.any(probe_3d.shape[1:] != probe_xz_shape)
                    or (probe_3d.wavelength != self.wavelength)
                    or np.any(probe_3d.pixel_size != obj_pixel_size)):
                e = ValueError(f"Mismatch between the supplied probe and the supplied scan parameters.")
                logger.error(e)
                raise e
        else:
            self.probe_3d = self.createProbe3D(dt.asdict(Probe2DParams()),
                                               wavelength,
                                               np.pi / 2 - two_theta,
                                               probe_xz_shape,
                                               obj_pixel_size)

        rotate_angle = np.pi / 2 - two_theta
        probe_y_pix_before_rotation = self.probe_3d.shape[0] * np.cos(rotate_angle) // 1
        pady, bordered_obj_ypix = self.calculateObjBorderAfterRotation(rotate_angle,
                                                                       probe_y_pix_before_rotation,
                                                                       self.obj.shape[0],
                                                                       self.obj.shape[1])

        if self.obj.bordered_array.shape[0] < bordered_obj_ypix:
            logger.warning(
                "Adding zero padding to the object in the y-direction so that the overall object y-width "
                + "covers the entirety of the feasible probe positions.")
            self.obj.border_shape = ((pady, pady), self.obj.border_shape[1], self.obj.border_shape[2])
        if scan_grid is not None:
            logger.info("Using supplied scan grid.")
            self.scan_grid = scan_grid
        else:
            elems_yz = lambda tup: np.array(tup)[[0, 2]]
            scan_grid_params_dict = dt.asdict(Scan2DGridParams(obj_pixel_size=elems_yz(self.obj.pixel_size)))
            self.scan_grid = self.createScanGrid(scan_grid_params_dict,
                                                 dt.asdict(AngleGridParams()),
                                                 elems_yz(self.obj.bordered_array.shape),
                                                 elems_yz(self.probe_3d.shape))
        self._calculateDiffractionPatterns()

    @staticmethod
    def createScanGrid(scan_grid_params_dict: dict,
                       angle_grid_params_dict: dict,
                       bordered_obj_yz_shape: Tuple[int, int],
                       probe_yz_shape: Tuple[int, int]) -> BraggPtychoGrid:

        logger.info("creating new 2d scan grid based on object and probe shapes.")
        scan_grid_2d = RectangleGrid(obj_w_border_shape=bordered_obj_yz_shape,
                                     probe_shape=probe_yz_shape,
                                     **scan_grid_params_dict)
        scan_grid_2d.checkOverlap()

        logger.info("Using created 2d scan grid to create full RC scan grid.")
        scan_grid = BraggPtychoGrid.fromPtychoScan2D(scan_grid_2d, grid2d_axes=("y", "z"), **angle_grid_params_dict)
        return scan_grid

    @staticmethod
    def createObj(obj_params_dict: dict,
                  det_3d_shape: Tuple[int, int, int],
                  obj_pixel_size: Tuple[float, float, float],
                  upsampling_factor: float = 1.0) -> Obj:
        logger.info("Creating new crystal cell.")
        while True:
            obj_crystal = Simulated3DCrystalCell(**obj_params_dict)
            if det_3d_shape[1] * upsampling_factor >= obj_crystal.bordered_array.shape[1]:
                break
            else:
                logger.warning("Generated object width is larger than detector x-width. Trying again.")

        logger.info('Adding x and z borders to crystal cell based on detector parameters.')

        # Ensuring that the number of pixels in the x dimension matches that in the detector.
        padx = (det_3d_shape[1] - obj_crystal.bordered_array.shape[1]) // 2

        # Calculating the padding needed to accomodate all feasible probe translations (w probe-obj interaction)
        padz, _ = Simulation.calculateObjBorderAfterRotation(0,
                                                             det_3d_shape[2],
                                                             obj_crystal.bordered_array.shape[1],
                                                             0)

        pad_shape = (0, padx, padz)
        obj = CustomObjFromArray(array=obj_crystal.bordered_array,
                                 border_shape=np.vstack((pad_shape, pad_shape)).T,
                                 border_const=0,
                                 pixel_size=obj_pixel_size)
        return obj

    @staticmethod
    def createProbe3D(probe_params_dict: dict,
                      wavelength: float,
                      rotate_angle: float,
                      probe_xz_shape: Tuple[int, int],
                      obj_pixel_size: Tuple[int, int, int]) -> Probe:
        logger.info("Creating new guassian, speckled, 2d probe.")
        probe_yz = GaussianSpeckledProbe(wavelength=wavelength, **probe_params_dict)
        ny = probe_yz.shape[0]

        nx, nz = probe_xz_shape
        # overdoing the repeat and interpolation just for safety
        logger.info("Rotating and interpolating the 2d probe to generate the 3d probe.")
        probe_yz_stack = np.repeat(probe_yz.wavefront.ifftshift[:, None, :], nx * 2, axis=1)

        rdeg = rotate_angle * 180 / np.pi
        rotated_real = ndimage.rotate(np.real(probe_yz_stack), rdeg, axes=(0, 1), mode='constant', order=1)
        rotated_imag = ndimage.rotate(np.imag(probe_yz_stack), rdeg, axes=(0, 1), mode='constant', order=1)
        rotated = rotated_real + 1j * rotated_imag

        # Calculating the number of pixels required to capture the y-structure of the rotated probe
        rny = ny / np.cos(rotate_angle) // 1
        rshape = np.array([rny, nx, nz]).astype('int')

        # Calculating the extent of the probe (relative to the center of rotation) in the x and y dimensions and
        # also ensuring that the probe array has an even number of pixels.
        # Adding any required padding so that the z dimension of the probe matches up with the detector.
        a = (np.array(rotated.shape) - rshape) // 2
        b = ((np.array(rotated.shape) - rshape) // 2 + (np.array(rotated.shape) - rshape) % 2)
        z_pad = (rshape[2] - rotated.shape[2]) // 2

        rotated_centered = rotated[a[0]:-b[0], a[1]:-b[1]]
        rotated_centered = np.pad(rotated_centered, [[0, 0], [0, 0], [z_pad, z_pad]],
                                  mode='constant', constant_values=0)
        probe_3d = CustomProbe3DFromArray(array=rotated_centered,
                                          wavelength=wavelength,
                                          pixel_size=obj_pixel_size)
        return probe_3d

    @staticmethod
    def calculateObjBorderAfterRotation(rotation_angle: float,
                                        py: int,
                                        oy: int,
                                        ox: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assumptions:
        * For simplicity, we assume that the probe is rotated in the x-y plane, with the z-axis as the axis of
        rotation. Additionally, we also assume that the scan direction is along the y-axis.
        * We want the extremum scan positions to be such that the center of the probe array is just touching the edge
        of the unknown reconstruction box, both before and after the rotation.
        * This will result in an overestimate of the size of the scanned area. We will need to filter out the scan
        positions that do not give rise to any diffraction effects.
        * If the rotation angle is 0, then
        Parameters
        ----------
        rotation_angle: float
            The rotation angle (in rad). Should always be between 0 and pi/2.
        py : int
            Number of probe y pixels before rotation.
        oy: int
            Number of obj y pixels (without border).
        ox: int
            Number of obj x pixels (without border).
        Returns
        -------
        border_pad : int
            Required border padding.
        shape_bordered_obj: int
            Shape of the final obj with the border.
        """

        py_projection_y = py / np.cos(rotation_angle)

        # border added due to the rotation of the probe for a probe of width 1 pixel
        dy = 2 * ox / np.tan(np.pi / 2 - rotation_angle)

        border_pad = (dy + py_projection_y) // 2
        shape_bordered_obj = oy + 2 * border_pad
        return border_pad.astype('int'), shape_bordered_obj.astype('int')

    def _calculatePhaseModulationsForRCAngles(self):
        logger.info("Calculating the phase modulations for the rc angles.")

        ttheta = self.scan_grid.two_theta
        domega = self.scan_grid.del_omega

        ki = 2 * np.pi / self.wavelength * np.array([np.cos(ttheta), np.sin(ttheta), 0])
        kf = 2 * np.pi / self.wavelength * np.array([1, 0, 0])
        q = (kf - ki)[:, None]

        ki_new = 2 * np.pi / self.wavelength * np.array([np.cos(ttheta + self.scan_grid.rc_angles),
                                                         np.sin(ttheta + self.scan_grid.rc_angles),
                                                         0 * self.scan_grid.rc_angles])
        kf_new = 2 * np.pi / self.wavelength * np.array([np.cos(self.scan_grid.rc_angles),
                                                         np.sin(self.scan_grid.rc_angles),
                                                         0 * self.scan_grid.rc_angles])
        q_new = kf_new - ki_new
        delta_q = q_new - q
        # Probe dimensions in real space (assumes even shape)
        position_grids = [np.arange(-s // 2, s // 2) * ds for (s, ds) in zip(self.probe_3d.shape,
                                                                             self.probe_3d.pixel_size)]
        Ry, Rx, Rz = np.meshgrid(*position_grids, indexing='ij')
        phase_modulations_all = np.exp(1j * np.array([delta_q[0, i] * Ry
                                                      + delta_q[1, i] * Rx
                                                      + delta_q[2, i] * Rz
                                                      for i in range(self.scan_grid.n_rc_angles)]))
        self._phase_modulations_all = phase_modulations_all

    def _calculateDiffractionPatterns(self):

        # Calculating the wave vectors
        self._calculatePhaseModulationsForRCAngles()

        intensities_all = []

        logger.info("Calculating the generated diffraction patterns.")
        for ia in range(self.scan_grid.n_rc_angles):
            for ib, (py, pz) in enumerate(self.scan_grid.positions_pix):
                obj_slice = self.obj.bordered_array[py: py + self.probe_3d.shape[0], :, pz: pz + self.probe_3d.shape[2]]
                exit_wave = obj_slice * self.probe_3d.wavefront * self._phase_modulations_all[ia]
                exit_wave_proj = np.sum(exit_wave, axis=0).fftshift
                intensities_all.append(exit_wave_proj.propFF().intensities)

        self.intensities = np.random.poisson(intensities_all) if self.poisson_noise else np.array(intensities_all)

    def filterScanPositionsAndIntensities(self, threshold: float = 1.0):
        from copy import deepcopy
        scan_grid_new = deepcopy(self.scan_grid)

        intensity_indices_new = np.arange(self.intensities.shape[0])[self.intensities.max(axis=(1, 2)) > threshold]
        full_rc_positions_indices_new = self.scan_grid.full_rc_positions_indices[intensity_indices_new]
        angle_indices, position_indices = full_rc_positions_indices_new.T
        position_indices_unique, inverse = np.unique(position_indices, return_inverse=True)

        scan_grid_new.positions_pix = self.scan_grid.positions_pix[position_indices_unique]
        scan_grid_new.positions_dist = self.scan_grid.positions_dist[position_indices_unique]
        scan_grid_new.positions_subpix = self.scan_grid.positions_subpix[position_indices_unique]

        #arr = np.empty(position_indices.max() + 1, dtype=np.int)
        #arr[position_indices_unique] = np.arange(position_indices_unique.shape[0])
        #position_indices_reindexed = arr[position_indices]
        position_indices_new = np.arange(position_indices_unique.shape[0])
        position_indices_reindexed = position_indices_new[inverse]
        scan_grid_new.full_rc_positions_indices = np.stack([angle_indices, position_indices_reindexed], axis=1)

        intensities_new = self.intensities[intensity_indices_new]
        return scan_grid_new, intensities_new