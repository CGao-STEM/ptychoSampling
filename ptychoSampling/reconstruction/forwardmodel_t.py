import tensorflow as tf
import numpy as np
import abc
from typing import Tuple, Union
from ptychoSampling.obj import Obj
from ptychoSampling.probe import Probe, Probe3D
from ptychoSampling.reconstruction.wavefront_t import propFF_t, fftshift_t, propTF_t
from ptychoSampling.grid import ScanGrid, BraggPtychoGrid
from ptychoSampling.logger import logger

class ForwardModelT(abc.ABC):

    def __init__(self,
                 obj: Obj,
                 probe: Probe,
                 scan_grid: ScanGrid,
                 upsampling_factor: int = 1):

        self.model_vars = {}

        with tf.device("/gpu:0"):
            self.obj_cmplx_t = self._addComplexVariable(obj.array, name="obj")
            self.obj_w_border_t = tf.pad(self.obj_cmplx_t, obj.border_shape, constant_values=obj.border_const)

            self.probe_cmplx_t = self._addComplexVariable(probe.wavefront, "probe")

            self.upsampling_factor = upsampling_factor
            logger.info("Creating obj views for the scan positions.")
            self._obj_views_all_t = self._getPtychoObjViewStack(obj, probe, scan_grid)
        #self.scan_positions_pix = scan_positions_pix

    def _addRealVariable(self, init: Union[float, np.ndarray],
                           name: str,
                           dtype: str='float32'):
        var = tf.Variable(init, dtype=dtype, name=name)
        self.model_vars[name] = {"variable": var,
                                 "output": var}
        return var

    def _addComplexVariable(self, init: Union[complex, np.ndarray],
                            name: str,
                            dtype: str = "float32" ):
        init_reals = np.array([np.real(init), np.imag(init)])
        var = tf.Variable(init_reals, dtype=dtype, name=name)
        output = tf.complex(var[0], var[1])
        self.model_vars[name] = {"variable": var,
                                 "output": output}

        return output

    def _getPtychoObjViewStack(self, obj: Obj, probe: Probe, scan_grid: ScanGrid) -> tf.Tensor:
        """Precalculate the object positioning for each scan position.

        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan
        position, we translate the object, then pad the object array to the size of the probe beam. For the padding,
        we assume free-space (transparent) propagation and use 1.0.

        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be
        buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and
        0 respectively.

        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        if scan_grid.full_field_probe:
            obj_views_t = self._getFullFieldProbeObjViews(scan_grid.positions_pix)
        else:
            obj_view_indices = self._genViewIndices(scan_grid.positions_pix)
            obj_view_indices_t = tf.constant(obj_view_indices)
            obj_views_t = tf.gather(tf.reshape(self.obj_w_border_t, [-1]), obj_view_indices_t)
            obj_views_t = tf.reshape(obj_views_t,
                                     (obj_view_indices.shape[0], self.probe_cmplx_t.get_shape().as_list()[0], -1))
        return obj_views_t

    def _genViewIndices(self, scan_positions_pix: np.ndarray):
        """ Generate the indices...

        Parameters
        ----------
        obj
        probe
        scan_positions_pix

        Returns
        -------

        """
        ny, nx = self.probe_cmplx_t.get_shape().as_list()
        views_indices_all = []
        ony, onx = self.obj_w_border_t.get_shape().as_list()
        for py, px in scan_positions_pix:
            Y, X = np.ogrid[py:ny + py, px:nx + px]
            view_single = (Y % ony) * onx + (X % onx)
            views_indices_all.append(view_single)

        return np.array(views_indices_all)

    def _getFullFieldProbeObjViews(self, scan_positions_pix: np.ndarray) -> tf.Tensor:
        """Precalculate the object positioning for each scan position.

        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan
        position, we translate the object, then pad the object array to the size of the probe beam. For the padding,
        we assume free-space (transparent) propagation and use 1.0.

        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be
        buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and
        0 respectively.

        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        obj_real_pads = []
        obj_imag_pads = []

        ony, onx = self.obj_w_border_t.get_shape().as_list()
        pny, pnx = self.probe_cmplx_t.get_shape().as_list()

        padfn = lambda p, t, c: tf.pad(t, [[p[0],  pny - (ony + p[0])], [p[1], pnx - (onx + p[1])]],
                                 constant_values=c)

        for p in scan_positions_pix:
            padded_real = padfn(p, tf.real(self.obj_w_border_t), 1.0)
            padded_imag = padfn(p, tf.imag(self.obj_w_border_t), 0.)
            obj_real_pads.append(padded_real)
            obj_imag_pads.append(padded_imag)

        obj_real_pads_t = tf.stack(obj_real_pads)
        obj_imag_pads_t = tf.stack(obj_imag_pads)

        obj_views_t = tf.complex(obj_real_pads_t, obj_imag_pads_t)
        return obj_views_t

    def _downsample(self, diffs_t: tf.Tensor) -> tf.Tensor:

        px = self.probe_cmplx_t.get_shape().as_list()[-1]
        py = self.probe_cmplx_t.get_shape().as_list()[-2]
        u = self.upsampling_factor
        px_new = px // u
        py_new = py // u
        diffs_t = tf.reshape(diffs_t, [-1, px_new, u, py_new, u])
        diffs_t = tf.reduce_sum(diffs_t, axis=(-1, -3))
        return diffs_t


    @abc.abstractmethod
    def predict(self, position_indices_t: tf.Tensor):
        pass

class FarfieldForwardModelT(ForwardModelT):

    def predict(self, position_indices_t: tf.Tensor):

        ndiffs = position_indices_t.get_shape().as_list()[0]


        if ndiffs == 0:
            return tf.zeros(shape=[], dtype='float32')

        batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t)
        batch_obj_views_t = fftshift_t(batch_obj_views_t)
        exit_waves_t = batch_obj_views_t * self.probe_cmplx_t
        out_wavefronts_t = propFF_t(exit_waves_t)
        guess_diffs_t = tf.abs(out_wavefronts_t) ** 2
        guess_diffs_t = self._downsample(guess_diffs_t) if self.upsampling_factor > 1 else guess_diffs_t
        return guess_diffs_t

class NearfieldForwardModelT(ForwardModelT):
    def __init__(self, obj: Obj,
                 probe: Probe,
                 scan_grid: ScanGrid,
                 propagation_dist: float,
                 upsampling_factor: int = 1):
        super().__init__(obj, probe, scan_grid, upsampling_factor)
        with tf.device("/gpu:0"):
            _, self._transfer_function = propTF_t(self.probe_cmplx_t,
                                                  wavelength=probe.wavelength,
                                                  pixel_size=probe.pixel_size,
                                                  prop_dist=propagation_dist,
                                                  return_transfer_function=True)

    def predict(self, position_indices_t: tf.Tensor):

        ndiffs = position_indices_t.get_shape().as_list()[0]

        if ndiffs == 0:
            return tf.zeros(shape=[], dtype='float32')

        batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t)
        batch_obj_views_t = fftshift_t(batch_obj_views_t)
        exit_waves_t = batch_obj_views_t * self.probe_cmplx_t

        out_wavefronts_t = propTF_t(exit_waves_t,
                                    reuse_transfer_function=True,
                                    transfer_function=self._transfer_function)
        guess_diffs_t = tf.abs(out_wavefronts_t)**2
        guess_diffs_t = self._downsample(guess_diffs_t) if self.upsampling_factor > 1 else guess_diffs_t
        return guess_diffs_t


class BraggPtychoForwardModelT(ForwardModelT):
    def __init__(self, obj: Obj,
                 probe: Probe3D,
                 scan_grid: BraggPtychoGrid,
                 exit_wave_axis: str = "y",
                 upsampling_factor: int = 1):
        if scan_grid.full_field_probe:
            e = ValueError("Full field probe not supported for Bragg ptychography.")
            logger.error(e)
            raise e
        if scan_grid.grid2d_axes != ("y", "z"):
            e = ValueError("Only supports the case where the ptychographic scan is on the yz-plane.")
            logger.error(e)
            raise e
        if exit_wave_axis != 'y':
            e = ValueError("Only supports the case where the exit waves are output along the y-direction.")
            logger.error(e)
            raise e

        super().__init__(obj, probe, scan_grid, upsampling_factor)
        logger.info("Creating the phase modulations for the scan angles.")

        with tf.device("/gpu:0"):
            self._probe_phase_modulations_all_t = tf.constant(self._getProbePhaseModulationsStack(probe, scan_grid),
                                                              dtype='complex64')
            self._full_rc_positions_indices_t = tf.constant(scan_grid.full_rc_positions_indices, dtype='int64')


    def predict(self, position_indices_t: tf.Tensor):
        ndiffs = position_indices_t.get_shape().as_list()[0]

        if ndiffs == 0:
            return tf.zeros(shape=[], dtype='float32')

        with tf.device("/gpu:0"):
            batch_rc_positions_indices = tf.gather(self._full_rc_positions_indices_t, position_indices_t)
            batch_obj_views_t = tf.gather(self._obj_views_all_t, batch_rc_positions_indices[:, 1])
            batch_phase_modulations_t = tf.gather(self._probe_phase_modulations_all_t, batch_rc_positions_indices[:, 0])

            batch_obj_views_t = batch_obj_views_t
            exit_waves_t = batch_obj_views_t * self.probe_cmplx_t * batch_phase_modulations_t
            exit_waves_proj_t = fftshift_t(tf.reduce_sum(exit_waves_t, axis=-3))

            out_wavefronts_t = propFF_t(exit_waves_proj_t)
            guess_diffs_t = tf.abs(out_wavefronts_t) ** 2
            guess_diffs_t = self._downsample(guess_diffs_t) if self.upsampling_factor > 1 else guess_diffs_t
        return guess_diffs_t

    def _genViewIndices(self, scan_positions_pix: np.ndarray):
        """ Generate the indices...

        Parameters
        ----------
        obj
        probe
        scan_positions_pix

        Returns
        -------

        """
        ny, nx, nz = self.probe_cmplx_t.get_shape().as_list()
        ony, onx, onz = self.obj_w_border_t.get_shape().as_list()

        views_indices_all = []
        for py, pz in scan_positions_pix:
            Y, X, Z = np.ogrid[py: py + ny, 0: nx , pz: pz + nz]
            view_single = ((Y % ony) * onx + (X % onx)) * onz + (Z % onz)
            views_indices_all.append(view_single)
        return np.array(views_indices_all)

    def _getPtychoObjViewStack(self, obj: Obj, probe: Probe, scan_grid: BraggPtychoGrid) -> tf.Tensor:
        """Precalculate the object positioning for each scan position.

        Assumes a small object that is translated within the dimensions of a full-field probe. For each scan
        position, we translate the object, then pad the object array to the size of the probe beam. For the padding,
        we assume free-space (transparent) propagation and use 1.0.

        In Tensorflow, performing the pad-and-stack procedure in the GPU for complex -valued arrays seems to be
        buggy. As a workaround, we separately pad-and-stack the real and imaginary parts of the object with 1.0 and
        0 respectively.

        Returns
        ----------
        obj_views : tensor(complex)
            Stack of tensors that correspond to the padded object at each object translation.
        """
        obj_view_indices = self._genViewIndices(scan_grid.positions_pix)
        obj_view_indices_t = tf.constant(obj_view_indices, dtype='int64')
        obj_views_t = tf.gather(tf.reshape(self.obj_w_border_t, [-1]), obj_view_indices_t)
        obj_views_t = tf.reshape(obj_views_t,
                                 (obj_view_indices.shape[0],
                                  *(self.probe_cmplx_t.get_shape().as_list())))
        return obj_views_t

    def _getProbePhaseModulationsStack(self, probe_3d: Probe3D, scan_grid: BraggPtychoGrid):
        ttheta = scan_grid.two_theta
        domega = scan_grid.del_omega

        ki = 2 * np.pi / probe_3d.wavelength * np.array([np.cos(ttheta), np.sin(ttheta), 0])
        kf = 2 * np.pi / probe_3d.wavelength * np.array([1, 0, 0])
        q = (kf - ki)[:, None]

        ki_new = 2 * np.pi / probe_3d.wavelength * np.array([np.cos(ttheta + scan_grid.rc_angles),
                                                         np.sin(ttheta + scan_grid.rc_angles),
                                                         0 * scan_grid.rc_angles])
        kf_new = 2 * np.pi / probe_3d.wavelength * np.array([np.cos(scan_grid.rc_angles),
                                                         np.sin(scan_grid.rc_angles),
                                                         0 * scan_grid.rc_angles])
        q_new = kf_new - ki_new
        delta_q = q_new - q
        # Probe dimensions in real space (assumes even shape)
        position_grids = [np.arange(-s // 2, s // 2) * ds for (s, ds) in zip(probe_3d.shape,
                                                                             probe_3d.pixel_size)]
        Ry, Rx, Rz = np.meshgrid(*position_grids, indexing='ij')
        phase_modulations_all = np.exp(1j * np.array([delta_q[0, i] * Ry
                                                      + delta_q[1, i] * Rx
                                                      + delta_q[2, i] * Rz
                                                      for i in range(scan_grid.n_rc_angles)]))
        return phase_modulations_all