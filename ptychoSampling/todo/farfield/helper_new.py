import numpy as np
from ptychoSampling.obj import Obj
from ptychoSampling.probe import Probe
from ptychoSampling.grid import ScanGrid
from copy import deepcopy as _deepcopy
from skimage.feature import register_translation as _register_translation
from ptychoSampling.reconstruction.recons import ReconstructionT
from ptychoSampling.logger import logger
import tensorflow as tf

class ePieReconstruction:
    def __init__(self, obj: Obj,
                 probe: Probe,
                 scan_grid: ScanGrid,
                 intensities: np.ndarray,
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None):
        self.obj = _deepcopy(obj)
        self.probe = _deepcopy(probe)
        self.scan_grid = scan_grid
        self.intensities = intensities

        self._obj_border_mask = np.pad(np.full(self.obj.shape, False),
                                       self.obj.border_shape,
                                       mode="constant",
                                       constant_values=True)

        self.obj_array_true = obj_array_true
        self.probe_wavefront_true = probe_wavefront_true

        self.shuffle_order = []
        self.losses = []
        self.obj_errors = []
        self.probe_errors = []

    def run(self, n_epochs: int,
            n_probe_fixed_epochs: int=0,
            alpha:float =1,
            beta: float=1):

        ny, nx = self.intensities.shape[0]
        npy, npx = self.probe.shape
        for epoch in range(n_epochs):
            indices = np.random.permutation(self.intensities.shape[0])
            self.shuffle_order = np.concatenate((self.shuffle_order, indices))
            sum_squared_errors = []

            for j in indices:
                py, px = self.scan_grid.positions_pix[j]
                intensity = self.intensities[j]

                obj_view = np.fft.fftshift(self.obj.bordered_array[py: py + npy, px: px + npx])
                exit_wave = self.probe.wavefront * obj_view

                f1 = exit_wave.fft2()
                f2 = f1 * np.sqrt(intensity) / (np.abs(f1) + 1e-8)
                exit_wave_new = f2.ifft2()

                t1 = np.conjugate(self.probe.wavefront) / np.max(np.abs(self.probe.wavefront) ** 2)
                t2 = exit_wave_new - exit_wave
                obj_view_new = obj_view + alpha * t1 * t2

                self.obj.bordered_array[py: npy + py, px: npx + px] = obj_view_new
                self.obj.bordered_array[self._obj_border_mask] = self.obj.border_const

                if epoch >= n_probe_fixed_epochs:
                    t3 = np.conjugate(obj_view) / np.max(np.abs(obj_view) ** 2)
                    self.probe.wavefront += beta * t3 * t2

                sum_squared_errors.append(np.sum((np.abs(f1) - np.sqrt(intensity)) ** 2))

            loss = np.mean(sum_squared_errors) / 2
            self.losses = np.append(self.losses, loss)

            if self.obj_array_true is not None:
                shift, err, phase = _register_translation(self.obj_array_true, self.obj.array, upsample_factor=10)
                shift, err, phase = _register_translation(self.obj_array_true * np.exp(1j * phase),
                                                          self.obj.array,
                                                          upsample_factor=10)
                self.obj_errors = np.append(self.obj_errors, err)

            if self.probe_wavefront_true is not None:
                shift, err, phase = _register_translation(self.probe_wavefront_true,
                                                          self.probe.wavefront,
                                                          upsample_factor=10)
                shift, err, phase = _register_translation(self.probe_wavefront_true * np.exp(1j * phase),
                                                          self.probe.wavefront,
                                                          upsample_factor=10)
                self.probe_errors = np.append(self.probe_errors, err)




class ADPieReconstructionT(ReconstructionT):
    def __init__(self, *args: int,
                 obj_array_true: np.ndarray = None,
                 probe_wavefront_true: np.ndarray = None,
                 shuffle_order: list = None,
                 **kwargs: int):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        logger.info('attaching fwd model...')
        self.attachForwardModel("farfield")
        logger.info('creating loss fn...')
        self.attachLossFunction("least_squared")

    def _getObjectLearningRate(self):
        batch_view_indices = tf.unstack(tf.reshape(self.batch_view_indices, [self.batch_size, -1, 1]))
        probe_sq = tf.reshape(tf.abs(self.tf_probe) ** 2, [-1])
        size = (self.obj.shape[0] + 2 * self.obj_border_npix) ** 2

        tf_mat = tf.zeros(size, dtype=tf.float32)
        for b in batch_view_indices:
            tf_mat += tf.scatter_nd(indices=b,
                                    shape=[size],
                                    updates=probe_sq)
        return 1 / tf.reduce_max(tf_mat)

    def _getProbeLearningRate(self):
        return 1 / tf.reduce_max(tf.reduce_sum(tf.abs(self.batch_obj_views) ** 2, axis=0))

    def _createDataBatches(self):
        """Use TensorFlow Datasets to create minibatches.

        When the diffraction data set is small enough to easily fit in the GPU memory, we can use the minibatch
        strategy detailed here to avoid I/O bottlenecks. For larger datasets, we have to adopt a slightly different
        minibatching strategy. More information about minibatches vs timing will be added later on in jupyter notebooks.

        In the scenario that the dataset fits into the GPU memory (which we shall assume as a given from now on),
        we can adopt the strategy:

            1) pre-calculate which (subset of) object pixels the probe interacts with at each scan position. We call
                these ``obj_views``. Ensure that the order of stacking of these ``obj_views`` match with the order of
                stacking of the diffraction patterns.

            2) create a list :math:`[0,1,...,N-1]` where :math:`N` is the number of diffraction patterns. Randomly
                select minibatches from this list (without replacement), then use the corresponding ``obj_view`` and
                diffraction intensity for further calculation.

            3) Use the iterator framework from TensorFlow to generate these minibatches. Inconveniently, when we use
                iterators, the minbatches of ``obj_views`` and diffraction patterns thus generated are not stored in the
                memory---every time we access the iterator, we get a new minibatch. In other words, there is no
                temporary storage to store this intermediate information at every step. If we want to do finer analysis
                on  the minibatches, we might want this information. For this temporary storage, we can use a TensorFlow
                Variable object, and store the minibatch information in the variable using an assign operation. The
                values of TensorFlow variables change only when we use these assign operations. In effect,
                we only access the iterator when we assign the value to the variable. Otherwise, the value of the
                variable remains in memory, unconnected to the iterator. Thus the minibatch information is preserved
                until we use the assign operation again.

        After generating a minibatch of ``obj_views``, we use the forward model to generate the predicted
        diffraction patterns for the current object and probe guesses.

        Parameters
        ----------
        batch_size : int
            Number of diffraction patterns in each minibatch.
        """
        all_indices_shuffled_t = tf.constant(np.random.permutation(self.n_all), dtype='int64')
        validation_indices_t = all_indices_shuffled_t[:self.n_validation]
        train_indices_t = all_indices_shuffled_t[self.n_validation:]

        train_batch_size = self.batch_size if self.batch_size > 0 else self.n_train
        validation_batch_size = min(train_batch_size, self.n_validation)

        train_iterate = self._getBatchedDataIterate(train_batch_size, train_indices_t)
        validation_iterate = self._getBatchedDataIterate(validation_batch_size, validation_indices_t)

        with tf.device("/gpu:0"):
            self._batch_train_input_v = tf.Variable(tf.zeros(train_batch_size, dtype=tf.int64))
            self._batch_validation_input_v = tf.Variable(tf.zeros(validation_batch_size, dtype=tf.int64))

            self._new_train_batch_op = self._batch_train_input_v.assign(train_iterate)
            self._new_validation_batch_op = self._batch_validation_input_v.assign(validation_iterate)

        self._iterations_per_epoch = self.n_train // train_batch_size

        dataset = tf.data.Dataset.from_tensor_slices(self.shuffle_order)

        dataset_batch = dataset.batch(batch_size, drop_remainder=True)
        dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', 5))

        iterator = dataset_batch.make_one_shot_iterator()


        dataset_indices = tf.data.Dataset.from_tensor_slices(self.shuffle_order)

        dataset_batch = dataset_indices.batch(self.batch_size, drop_remainder=True)
        self.dataset_batch = dataset_batch.prefetch(10)

        self.iterator = self.dataset_batch.make_one_shot_iterator()

        batchi = self.iterator.get_next()
        self.batch_indices = tf.Variable(tf.zeros(self.batch_size, dtype=tf.int64),
                                         dtype=tf.int64, trainable=False)

    @staticmethod
    def _getBatchedDataIterate(shuffle_order):
        dataset = tf.data.Dataset.from_tensor_slices(shuffle_order)

        dataset_batch = dataset.batch(1, drop_remainder=True)
        dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', 10))

        iterator = dataset_batch.make_one_shot_iterator()
        return iterator.get_next()





