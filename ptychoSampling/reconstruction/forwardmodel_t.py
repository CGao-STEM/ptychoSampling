import tensorflow as tf
import numpy as np
import abc
from typing import Callable, List, Union, Any
from ptychoSampling.obj import Obj
from ptychoSampling.probe import Probe
from ptychoSampling.reconstruction.wavefront_t import propFF_t, fftshift_t
from ptychoSampling.grid import ScanGrid

class ForwardModelT(abc.ABC):

    def __init__(self,
                 obj: Obj,
                 probe: Probe,
                 scan_positions_pix: np.ndarray,
                 upsampling_factor: int = 1):

        self.model_vars = {}

        self.obj_cmplx_t = self._addComplexVariable(obj.array, name="obj")
        self.obj_w_border_t = tf.pad(self.obj_cmplx_t, obj.border_shape, constant_values=obj.border_const)

        self.probe_cmplx_t = self._addComplexVariable(probe.wavefront, "probe")

        self.upsampling_factor = upsampling_factor
        self.scan_positions_pix = scan_positions_pix

    def _addRealVariable(self, init: Union[float, np.ndarray],
                           name: str,
                           dtype: str='float32'):
        with tf.device("/gpu:0"):
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

    @abc.abstractmethod
    def predict(self, position_indices_t: tf.Tensor):
        pass


class FarfieldForwardModelT(ForwardModelT):
    def __init__(self, obj, probe, scan_positions_pix, upsampling_factor=1):

        super().__init__(obj, probe, scan_positions_pix, upsampling_factor)

        obj_view_indices = self.genViewIndices(obj, probe, scan_positions_pix)
        self._obj_views_all_t = self._getPtychoObjViewsStack(obj_view_indices)


    def predict(self, position_indices_t: tf.Tensor):

        ndiffs = position_indices_t.get_shape()[0]
        px = self.probe_cmplx_t.get_shape()[-1]
        py = self.probe_cmplx_t.get_shape()[-2]

        if ndiffs == 0:
            return tf.zeros(shape=[], dtype='float32')

        batch_obj_views_t = tf.gather(self._obj_views_all_t, position_indices_t)
        batch_obj_views_t = fftshift_t(batch_obj_views_t)
        exit_waves_t = batch_obj_views_t * self.probe_cmplx_t
        out_wavefronts_t = propFF_t(exit_waves_t)
        guess_diffs_t = tf.abs(out_wavefronts_t) ** 2
        if self.upsampling_factor > 1:
            u = self.upsampling_factor
            px_new = px // u
            py_new = py // u
            guess_diffs_t = tf.reshape(guess_diffs_t, [-1, px_new, u, py_new, u])
            guess_diffs_t = tf.reduce_sum(guess_diffs_t, axis=(-1, -3))
        return guess_diffs_t


    def _getPtychoObjViewsStack(self, obj_view_indices: np.ndarray) -> tf.Tensor:
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
        obj_view_indices_t = tf.constant(obj_view_indices)
        obj_views_t = tf.gather(tf.reshape(self.obj_w_border_t, [-1]), obj_view_indices_t)
        obj_views_t = tf.reshape(obj_views_t, (obj_view_indices.shape[0], self.probe_cmplx_t.get_shape()[0], -1))
        return obj_views_t

    @staticmethod
    def genViewIndices(obj: Obj, probe: Probe, scan_positions_pix: np.ndarray):
        """
        .. todo::

            Need to double check this for correctness.
        """
        ny, nx = probe.shape
        views_indices_all = []
        ony, onx = obj.bordered_array.shape
        for px, py in scan_positions_pix:
            R, C = np.ogrid[py:ny + py, px:nx + px]
            view_single = (R % ony) * onx + (C % onx)
            views_indices_all.append(view_single)

        return np.array(views_indices_all)
