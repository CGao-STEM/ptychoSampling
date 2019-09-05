import numpy as np
import abc
import tensorflow as tf
import ptychoSampling.probe
import ptychoSampling.obj
#import ptychoSampling.ptycho
import ptychoSampling.grid
from ptychoSampling.logger import logger
from ptychoSampling.reconstruction.forwardmodel_t import  FarfieldForwardModelT
import ptychoSampling.reconstruction.options
from ptychoSampling.reconstruction.datalogs_t import DataLogs
import copy

class ReconstructionT(abc.ABC):
    r"""
    """

    def __init__(self, obj: ptychoSampling.obj.Obj,
                 probe: ptychoSampling.probe.Probe,
                 grid: ptychoSampling.grid.ScanGrid,
                 intensities: np.ndarray,
                 n_validation: int = 0,
                 batch_size: int = 0):
        self.obj = copy.deepcopy(obj)
        self.probe = copy.deepcopy(probe)
        self.grid = copy.deepcopy(grid)
        self.intensities = copy.deepcopy(intensities)

        self.n_all = self.intensities.shape[0]
        self.n_validation = n_validation
        self.n_train = self.n_all - self.n_validation
        self.batch_size = batch_size

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device("/gpu:0"):
                self._intensities_t = tf.constant(intensities, dtype=tf.float32)
            logger.info('creating batches...')
            self._createDataBatches()

        logger.info('creating log...')
        self.iteration = 0
        self.datalog = DataLogs()
        self._default_log_items = {"epoch": None,
                                   "train_loss": None}

        for key in self._default_log_items:
            self.datalog.addSimpleMetric(key)

        if self.n_validation > 0:
            self._validation_log_items = {"validation_loss": None,
                                          "validation_min": None,
                                          "patience": None}
            for key in self._validation_log_items:
                self.datalog.addSimpleMetric(key)


    def attachForwardModel(self, model_type: str, **kwargs: dict):
        models_all =  ptychoSampling.reconstruction.options.OPTIONS["forward models"]
        self._checkConfigProperty(models_all, model_type)
        self._model_type = model_type

        with self.graph.as_default():
            self.fwd_model = models_all[model_type](self.obj, self.probe, self.grid.positions_pix, **kwargs)

    def attachLossFunction(self, loss_type: str, amplitude_loss: bool = True):
        losses_all = ptychoSampling.reconstruction.options.OPTIONS["loss functions"]
        self._checkAttr("fwd_model", "loss functions")
        self._checkConfigProperty(losses_all, loss_type)

        self._loss_type = loss_type
        self._amplitude_loss = amplitude_loss

        power = 0.5 if amplitude_loss else 1.0

        with self.graph.as_default():
            self._batch_train_predictions_t = self.fwd_model.predict(self._batch_train_input_v) ** power
            self._batch_validation_predictions_t = self.fwd_model.predict(self._batch_validation_input_v) ** power

            self._batch_train_data_t = tf.gather(self._intensities_t, self._batch_train_input_v) ** power
            self._batch_validation_data_t = tf.gather(self._intensities_t, self._batch_validation_input_v) ** power

            #if amplitude_loss:
            #    for t in [self._batch_train_predictions_t,
            #              self._batch_validation_predictions_t,
            #              self._batch_train_data_t,
            #              self._batch_validation_data_t]:
            #       t = tf.sqrt(t)

            self._train_loss_t = losses_all[loss_type](self._batch_train_predictions_t, self._batch_train_data_t)
            self._validation_loss_t = losses_all[loss_type](self._batch_validation_predictions_t,
                                                            self._batch_validation_data_t)


    def attachOptimizerPerVariable(self, variable_name: str,
                                   optimizer_type: str,
                                   optimizer_args: dict,
                                   initial_update_delay: int = 0,
                                   update_frequency: int = 1,
                                   checkpoint_frequency: int = 100):
        """
               Set up the training and validation loss functions and the probe and object optimization procedure.

               For the training data, we use the loss value for the current minibatch of scan positions. We then try to
               minimize this loss value by using the Adam optimizer for the object (and probe) variables.

               For the validation data, we use the entire validation data set to calculate the loss value. We do not use
               this for the gradient calculations.

               For now, we use the amplitude loss function.

               Parameters
               ----------
               obj_learning_rate : float
                   Learning rate (or initial step size) for the Adam optimizer for the object variable. Defaults to 0.01.
               probe_learning_rate : float
                   Learning rate (or initial step size) for the Adam optimizer for the probe variable. Only applies we
                   enable probe reconstruction. Defaults to 10.
        """
        self._checkAttr("fwd_model", "optimizers")
        if variable_name not in self.fwd_model.model_vars:
            e = ValueError(f"{variable_name} is not a supported variable in {self.fwd_model}")
            logger.error(e)
            raise e

        optimization_all = ptychoSampling.reconstruction.options.OPTIONS["optimization_methods"]
        self._checkAttr("_train_loss_t", "optimizer")
        self._checkConfigProperty(optimization_all, optimizer_type)

        if not hasattr(self, "optimizers"):
            self.optimizers = []
        with self.graph.as_default():
            optimizer = optimization_all[optimizer_type](**optimizer_args)
            minimize_op = optimizer.minimize(self._train_loss_t,
                                             var_list=[self.fwd_model.model_vars[variable_name]["variable"]])
        self.optimizers.append({"var": variable_name,
                                "optimizer": optimizer,
                                "minimize_op": minimize_op,
                                "initial_update_delay": initial_update_delay,
                                "update_frequency": update_frequency})



    def _checkAttr(self, attr_to_check, attr_this):
        if not hasattr(self, attr_to_check):
            e = AttributeError(f"First attach a {attr_to_check} before attaching {attr_this}.")
            logger.error(e)
            raise e

    def _checkConfigProperty(self, options: dict, key_to_check: str):
        if key_to_check not in options:
            e = ValueError(f"{key_to_check} is not currently supported. "
                           + f"Check if {key_to_check} exists as an option among {options} in options.py")
            logger.error(e)
            raise e

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

        self._batch_train_input_v = tf.Variable(tf.zeros(train_batch_size, dtype=tf.int64))
        self._batch_validation_input_v = tf.Variable(tf.zeros(validation_batch_size, dtype=tf.int64))

        self._new_train_batch_op = self._batch_train_input_v.assign(train_iterate)
        self._new_validation_batch_op = self._batch_validation_input_v.assign(validation_iterate)

        self._iterations_per_epoch = self.n_train // train_batch_size

    @property
    def epoch(self):
        return self.iteration // self._iterations_per_epoch


    @staticmethod
    def _getBatchedDataIterate(batch_size: int, data_tensor: tf.Tensor):
        dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset = dataset.shuffle(data_tensor.get_shape()[0])
        dataset = dataset.repeat()

        dataset_batch = dataset.batch(batch_size, drop_remainder=True)
        dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', 5))

        iterator = dataset_batch.make_one_shot_iterator()
        return iterator.get_next()

    def addCustomMetricToDataLog(self, title: str, tensor: tf.Tensor, registration_ground_truth=None):
        if registration_ground_truth is None:
            self.datalog.addCustomTensorMetric(title=title, tensor=tensor)
        else:
            self.datalog.addCustomTensorMetric(title=title, tensor=tensor, registration=True,
                                               true=registration_ground_truth)

    def finalizeSetup(self):
        self._checkAttr("optimizers", "finalize")
        self.datalog.finalize()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            self.session.run(tf.global_variables_initializer())

    def _printDebugOutput(self, debug_output_epoch_frequency, epoch, print_debug_header):
        if not epoch % debug_output_epoch_frequency == 0:
            return print_debug_header
        self.datalog.printDebugOutput(print_debug_header)
        return False

    def run(self, max_iterations: int = 5000,
            validation_epoch_frequency: int = 1,
            improvement_threshold: float = 5e-4,
            patience_epoch: int = 50,
            patience_increase_factor: float = 1.5,
            debug_output: bool = True,
            debug_output_epoch_frequency: int = 10):
        """Perform the optimization a specified number of times (with early stopping).

                This command provides fine control over the number of times we run the minization procedure and over the
                early stopping criterion. To understand how this works, we first introduce some terminology:
                    - ``iteration``: Every minibatch update counts as one iteration.
                    - ``epoch``: A single pass through the entire data set counts as one epoch. In the minibatch setting,
                            each epoch usually consists of multiple iterations.
                    - ``patience``: When there is no improvement in the minimum loss value obtained after an epoch of
                            optimization, we can either pull the trigger immediately, or wait for a fixed number of epochs
                            (without improvement) before pulling the trigger. This fixed number of epochs where we wait,
                            even when we see no improvement, is the patience.
                    - ``patience_increase_factor``: Typically, during the optimization procedure, we expect a fast
                            improvement in the loss value at the beginning of the optimization procedure, with the rate of
                            improvement slowing down as we proceed with the optimization. To account for this, we want an
                            early stopping procedure with low patience at the beginning, and with increasing patience as we
                            move towards the minimum. The `patience_increase_factor` controls the rate of increase of the
                            patience (which depends on the `validation_frequency` parameter).

                Parameters
                ----------
                validation_frequency : int
                    Number of epochs between each calculation of the validation loss. This is also the number of epochs
                    between each check (and update) of the `patience` parameter.
                improvement_threshold : float
                    Relative tolerance for ``improvement`` of the minimum loss value, where ``improvement`` is defined as
                    ``improvement = abs(validation_best_loss - validation_loss) / validation_best_loss``.
                patience : int
                    Minimum allowable number of epochs betweeen improvement in the minimum loss value, where the
                    ``improvement`` is as defined by `improvement_threshold`. The `patience` is increased dynamically (
                    depending on the `patience_increase_factor` and the `validation_frequency`)  during the optimization procedure.
                patience_increase_factor : float
                    Factor by which the patience is increased whenever the ``improvement`` is better than the
                    `improvement_threshold`.
                max_iters : int
                    Maximum number of ``iterations`` to perform. Each ``epoch`` is usually composed of multiple iterations.
                debug_output : bool
                    Whether to print the validation log output to the screen.
                debug_output_frequency : int
                    Number of validation updates after which we print the validation log output to the screen.
                probe_fixed_epochs : int
                    Number of epochs (at the beginning) where we only adjust the object variable.
        """

        if not hasattr(self, "session"):
            self.finalizeSetup()

        print_debug_header = True

        epochs_start = self.epoch
        for i in range(max_iterations):
            self.iteration += 1
            self.session.run(self._new_train_batch_op)

            min_ops = [self._train_loss_t]
            for o in self.optimizers:
                if (o["initial_update_delay"] <= self.iteration) and (self.iteration % o["update_frequency"] == 0):
                    min_ops.append(o["minimize_op"])
            outs = self.session.run(min_ops)

            self._default_log_items["epoch"] = self.epoch
            self._default_log_items["train_loss"] = outs[0]
            self.datalog.logStep(self.iteration, self._default_log_items)

            #epoch_complete = (i % self._iterations_per_epoch == 0)
            #if not epoch_complete:
            #    continue
            if i % self._iterations_per_epoch != 0:
                continue

            #epochs_this_run += 1 if i > 0 else 0
            epochs_this_run = self.epoch - epochs_start
            custom_metrics = self.datalog.getCustomTensorMetrics(epochs_this_run)
            custom_metrics_tensors = list(custom_metrics.values())
            if len(custom_metrics_tensors) > 0:
                custom_metrics_values = self.session.run(custom_metrics_tensors)
                self.datalog.logStep(self.iteration, dict(zip(custom_metrics.keys(), custom_metrics_values)))

            if self.n_validation == 0 or self.epoch % validation_epoch_frequency != 0:
                if debug_output:
                    print_debug_header = self._printDebugOutput(debug_output_epoch_frequency,
                                                                epochs_this_run,
                                                                print_debug_header)
                continue

            self.session.run(self._new_validation_batch_op)
            v = self.session.run(self._validation_loss_t)

            v_min = np.inf if self.iteration == 1 else self._validation_log_items["validation_min"]
            if v < v_min:
                if np.abs(v - v_min) > v_min * improvement_threshold:
                    patience_epoch = max(patience_epoch, epochs_this_run * patience_increase_factor)
                v_min = v

            self._validation_log_items["validation_loss"] = v
            self._validation_log_items["validation_min"] = v_min
            self._validation_log_items["patience"] = patience_epoch
            self.datalog.logStep(self.iteration, self._validation_log_items)

            if debug_output:
                print_debug_header = self._printDebugOutput(debug_output_epoch_frequency,
                                                            epochs_this_run,
                                                            print_debug_header)
            if epochs_this_run >= patience_epoch:
                break
        self._updateOutputs()


    @abc.abstractmethod
    def _updateOutputs(self):
        pass





class FarFieldGaussianReconstructionT(ReconstructionT):
    def __init__(self, *args, **kwargs):
        logger.info('initializing...')
        super().__init__(*args, **kwargs)

        logger.info('attaching fwd model...')
        self.attachForwardModel("farfield")
        logger.info('creating loss fn...')
        self.attachLossFunction("least_squared")
        logger.info('creating optimizers...')
        self.attachOptimizerPerVariable("obj", optimizer_type="adam", optimizer_args = {"learning_rate":1e-2})
        self.attachOptimizerPerVariable("probe", optimizer_type="adam", optimizer_args={"learning_rate":1e-1},
                                        initial_update_delay=100)


    def _updateOutputs(self):
        obj_array, probe_wavefront = self.session.run([self.fwd_model.model_vars["obj"]["output"],
                                                       self.fwd_model.model_vars["probe"]["output"]])
        self.obj.array = obj_array
        self.probe.wavefront = probe_wavefront


    def genPlotsRecons(self) -> None:
        import matplotlib.pyplot as plt
        """Plot the reconstructed probe and object amplitudes and phases."""
        self._updateOutputs()

        plt.figure(figsize=[14, 3])
        plt.subplot(1, 4, 1)
        plt.pcolormesh(np.abs(self.obj.array), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.pcolormesh(np.angle(self.obj.array), cmap='gray')
        plt.subplot(1, 4, 3)
        plt.pcolormesh(np.abs(self.probe.wavefront), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.pcolormesh(np.angle(self.probe.wavefront), cmap='gray')
        plt.colorbar()
        plt.show()

    def genPlotMetrics(self) -> None:
        """Plot the metrics recorded in the log."""
        import matplotlib.pyplot as plt
        log = self.datalog.dataframe
        fig, axs = plt.subplots(1, 4, figsize=[14, 3])
        axs[0].plot(np.log(log['train_loss'].dropna()))
        axs[0].set_title('train_loss')

        #axs[1].plot(log['obj_error'].dropna())
        #axs[1].set_title('obj_error')

        #axs[2].plot(log['probe_error'].dropna())
        #axs[2].set_title('probe_error')

        #axs[3].plot(np.log(log['validation_loss'].dropna()))
        #axs[3].set_title('validation_loss')
        plt.show()

    def _getClipOp(self, max_abs: float = 1.0) -> None:
        """Not used for now"""
        with self.graph.as_default():
            obj_reshaped = tf.reshape(self.tf_obj, [2, -1])
            obj_clipped = tf.clip_by_norm(obj_reshaped, max_abs, axes=[0])
            obj_clipped_reshaped = tf.reshape(obj_clipped, [-1])
            clipped = tf.assign(self.tf_obj, obj_clipped_reshaped, name='clip_op')
        return clipped