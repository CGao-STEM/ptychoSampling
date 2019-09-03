import numpy as np
import tensorflow as tf
import ptychoSampling.probe
import ptychoSampling.obj
import ptychoSampling.ptycho
import ptychoSampling.grid
from ptychoSampling.logger import logger
from ptychoSampling.reconstruction.forwardmodel_t import  FarfieldForwardModelT
import ptychoSampling.reconstruction.options
from ptychoSampling.reconstruction.datalogs_t import DataLogs



class ReconstructionT:
    r"""
    """

    def __init__(self, ptycho: ptychoSampling.ptycho.Ptychography):
        pass



class FarFieldReconstructionT:#(ReconstructionT):
    def __init__(self, obj: ptychoSampling.obj.Obj,
                 probe: ptychoSampling.probe.Probe,
                 grid: ptychoSampling.grid.ScanGrid,
                 intensities: np.ndarray,
                 n_validation: int = 0,
                 batch_size: int = 0):
        self.obj = obj
        self.probe = probe
        self.grid = grid
        self.intensities = intensities

        self.n_all = self.intensities.shape[0]
        self.n_validation = n_validation
        self.n_train = self.n_all - self.n_validation
        self.batch_size = batch_size

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._createFwdModel()
            self._createDataBatches()

    def attachForwardModel(self, model_type: str):
        models_all =  ptychoSampling.reconstruction.options.OPTIONS["forward models"]
        self._checkConfigProperty(model_type, models_all)
        self._model_type = model_type

        with self.graph.as_default():
            self.fwd_model = models_all[model_type](self.obj, self.probe, self.grid.positions_pix)


    def attachLossFunction(self, loss_type: str, amplitude_loss: bool = True):
        losses_all = ptychoSampling.reconstruction.options.OPTIONS["loss functions"]
        self._checkAttr("fwd_model", "loss functions")
        self._checkConfigProperty("loss functions", loss_type)

        self._loss_type = loss_type
        self._amplitude_loss = amplitude_loss

        with self.graph.as_default():
            self._batch_train_predictions_t = self.fwd_model.predict(self._batch_train_input_v)
            self._batch_validation_predictions_t = self.fwd_model.predict(self._batch_validation_input_v)

            self._batch_train_data_t = tf.gather(self._intensities_t, self._batch_train_input_v)
            self._batch_validation_data_t = tf.gather(self._intensities_t, self._batch_validation_input_v)

            if amplitude_loss:
                for t in [self._batch_train_predictions_t,
                          self._batch_validation_predictions_t,
                          self._batch_train_data_t,
                          self._batch_validation_data_t]:
                    t = tf.sqrt(t)

            self.train_loss_t = losses_all[loss_type](self._batch_train_predictions_t, self._batch_train_data_t)
            self.validation_loss_t = losses_all[loss_type](self._batch_validation_predictions_t,
                                                            self._batch_validation_data_t)
            self.addMetric(self.train_loss_t, "train_loss")
            self.addMetric(self.validation_loss_t, "validation_loss")

    def addMetric(self, t: tf.Tensor, name: str):
        if not hasattr(self, "_metrics"):
            self._metrics = {}
        self._metrics[name] = t

    def attachOptimizerPerVariable(self, variable_name: str, optimizer_type: str, optimizer_args={}):
        self._checkAttr("fwd_model", "optimizers")
        if variable_name not in self.fwd_model.model_vars:
            e = ValueError(f"{variable_name} is not a supported variable in {self.fwd_model}")
            logger.error(e)
            raise e

        optimizers_all = ptychoSampling.reconstruction.options.OPTIONS["optimizers"]
        self._checkAttr("train_loss_t", "optimizers")
        self._checkConfigProperty("optimizers", optimizer_type)

        if not hasattr(self, "optimizers"):
            self.optimizers = []
        with self.graph.as_default():
            optimizer = optimizers_all[optimizer_type](optimizer_args=optimizer_args)
            minimize_op = optimizer.minimize(self.train_loss_t,
                                             var_list=[self.fwd_model.model_vars[variable_name]["variable"]])
        self.optimizers.append({"var": variable_name,
                                "optimizer": optimizer,
                                "minimize_op": minimize_op})

    def initDataLog(self, global_print_frequency: int):
        if not hasattr(self, "_datalog"):
            self._datalog = DataLogs(global_print_frequency)
        else:
            e = AttributeError("Data log already created. Cannot recreate.")
            logger.error(e)
            raise e

    def attachDataLog(self, variable_or_metric: str,
                      log_freq: int,
                      registration_ground_truth: np.ndarray = None):

        if variable_or_metric in self._metrics:
            item = self._metrics[variable_or_metric]
        elif variable_or_metric in self.fwd_model.model_vars:
            item = self.fwd_model.model_vars[variable_or_metric]["output"]
        else:
            e = AttributeError(f"Cannot add log for {variable_or_metric}. Requested item should be present in either "
                               + "self._metrics or self.fwd_model.model_vars.")
            logger.error(e)
            raise e

        if registration_ground_truth is not None:
            self._datalog.addRegistration(variable_or_metric, item, registration_ground_truth, log_freq)
        else:
            self._datalog.addTensorLog(variable_or_metric, item, log_freq)



    def _checkAttr(self, attr_to_check, attr_this):
        if not hasattr(self, attr_to_check):
            e = AttributeError(f"First attach a {attr_to_check} before attaching {attr_this}.")
            logger.error(e)
            raise e

    def _checkConfigProperty(self, option_to_check: str, key_to_check: str):
        if key_to_check not in ptychoSampling.reconstruction.options.OPTIONS[option_to_check]:
            e = ValueError(f"{key_to_check} is not currently supported. "
                           + f"Check if {key_to_check} exists as an option  for {option_to_check} in options.py")
            logger.error(e)
            raise e

    def _createFwdModel(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.model = FarfieldForwardModelT(self.obj, self.probe, self.grid.positions_pix)
            self._intensities_t = tf.constant(self.intensities)

    def _createDataBatches(self):
        all_indices_shuffled_t = tf.constant(np.random.permutation(self.n_all), dtype='int64')
        validation_indices_t = all_indices_shuffled_t[:self.n_validation_diffs]
        train_indices_t = all_indices_shuffled_t[self.n_validation_diffs:]

        train_batch_size = self.batch_size if self.batch_size > 0 else self.n_train
        validation_batch_size = min(train_batch_size, self.n_validation)

        train_iterate = self._getBatchedDataIterate(train_batch_size, train_indices_t)
        validation_iterate = self._getBatchedDataIterate(validation_batch_size, validation_indices_t)

        self._batch_train_input_v = tf.Variable(tf.zeros(train_batch_size, dtype=tf.int64))
        self._batch_validation_input_v = tf.Variable(tf.zeros(validation_batch_size, dtype=tf.int64))

        self._new_train_batch_op = self._batch_train_input_v.assign(train_iterate)
        self._new_validation_batch_op = self._batch_validation_input_v.assign(validation_iterate)


    @staticmethod
    def _getBatchedDataIterate(batch_size: int, data_tensor: tf.Tensor):
        dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
        dataset = dataset.shuffle(data_tensor.get_shape()[0])
        dataset = dataset.repeat()

        dataset_batch = dataset.batch(batch_size, drop_remainder=True)
        dataset_batch = dataset_batch.apply(tf.data.experimental.prefetch_to_device('/gpu:0', 5))

        iterator = dataset_batch.make_one_shot_iterator()
        return iterator.get_next()

    def initSession(self):
        self._checkAttr("optimizers", "session")
        with self._graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run([self._new_train_batch_op, self._new_validation_batch_op])
        pass

    def run(self):
        pass


    def __init__(self,
                 positions: np.ndarray,
                 diffraction_mods: np.ndarray,
                 det_args: dict,
                 obj_args: dict,
                 probe_args: dict,
                 obj_guess: Optional[np.ndarray] = None,
                 probe_guess: Optional[np.ndarray] = None,
                 probe_recons: bool = True,
                 batch_size: int = 0,
                 n_validation_diffs: int = 0,
                 obj_true: Optional[np.ndarray] = None,
                 probe_true: Optional[np.ndarray] = None) -> None:

        det_params = self._setParams(DetectorParams, det_args)
        obj_args = self._checkGuessNpix(obj_args, obj_guess)
        probe_args = self._checkGuessNpix(probe_args, probe_guess)
        obj_params = self._setParams(ObjParams, obj_args)
        probe_params = self._setParams(ProbeParams, probe_args)

        self.inits = ReconsInits(positions=positions,
                                 mods=diffraction_mods,
                                 obj_params=obj_params,
                                 probe_params=probe_params,
                                 det_params=det_params,
                                 obj=obj_guess,
                                 probe=probe_guess)

        self.probe_recons = probe_recons

        # Tensorflow setup
        self._createTFModels(n_validation_diffs)
        self._createBatches(batch_size)

        self.obj_true = obj_true
        self.probe_true = probe_true

        log = DataFrame(columns=['loss', 'epoch', 'obj_error', 'probe_error', 'validation_loss', 'patience'],
                        dtype='float32')
        self.outs = ReconsOutputs(np.zeros_like(obj_guess),
                                  np.zeros_like(probe_guess),
                                  log)


    @staticmethod
    def _setParams(params_dataclass: object,
                   args: dict) -> object:
        """Ignore any extra parameters supplied as an argument to initialize the pramams_dataclass"""
        fields = [f.name for f in dt.fields(params_dataclass) if f.init == True]
        args_filtered = {k: v for k, v in args.items() if k in fields}
        return params_dataclass(**args_filtered)

    def _createModel(self):
        pass

    def _createTFModels(self, n_validation_diffs: int) -> None:
        """Create the TensorFlow Graph, the object and probe variables, and the training and validation forward models.

        Creates:
            * the Tensorflow graph object for the tensorflow calculations.
            * real-valued object and probe variables for the optimization procedure (see documentation for
                ReconsVarsAndConsts for more detail)
            * creates the complex-valued object and probe tensors, and adds borders if necessary.
            * creates the propagation kernel.
            * Divides the data into training and validation sets depending on the `n_validation_diffs` parameter.

        Parameters
        ----------
        n_validation_diffs : int
            Number of randomly selected diffraction patterns to allocate to the validation data set.
        """
        oguess = self.inits.obj
        pguess = self.inits.probe
        ndiffs = self.inits.positions.shape[0]
        n_train_diffs = ndiffs - n_validation_diffs
        self.n_train_diffs = n_train_diffs
        self.n_validation_diffs = n_validation_diffs

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._obj_v = tf.Variable(np.array([np.real(oguess), np.imag(oguess)]), dtype='float32')
            self._obj_cmplx_t = tf.complex(self._obj_v[0], self._obj_v[1])
            pad = self.inits.obj_params.border_npix
            self._obj_w_border_t = tf.pad(self._obj_cmplx_t, [[pad, pad], [pad, pad]],
                                          constant_values=self.inits.obj_params.border_const)
            self._prop_kernel_t = tf.constant(self.inits.prop_kernel, dtype='complex64')

            self._probe_v = tf.Variable(np.array([np.real(pguess), np.imag(pguess)]), dtype='float32')
            self._probe_cmplx_t = tf.complex(self._probe_v[0], self._probe_v[1])

            self._mods_t = tf.constant(self.inits.mods_shifted, dtype='float32')
            self._obj_views_t = self._getObjViewsStack()

            all_indices_shuffled_t = tf.constant(np.random.permutation(ndiffs), dtype='int64')
            validation_indices_t = all_indices_shuffled_t[:n_validation_diffs]
            train_indices_t = all_indices_shuffled_t[n_validation_diffs:]

            train_mods_t = tf.gather(self._mods_t, train_indices_t)
            train_obj_views_t = tf.gather(self._obj_views_t, train_indices_t)

            validation_mods_t = tf.gather(self._mods_t, validation_indices_t)
            validation_obj_views_t = tf.gather(self._obj_views_t, validation_indices_t)
            validation_predictions_t = self._getBatchPredictedData(validation_obj_views_t)

        self._train_full = ForwardModel(ndiffs=n_train_diffs,
                                        indices_t=train_indices_t,
                                        mods_t=train_mods_t,
                                        obj_views_t=train_obj_views_t)
        self._validation = ForwardModel(ndiffs=n_validation_diffs,
                                        indices_t=validation_indices_t,
                                        mods_t=validation_mods_t,
                                        obj_views_t=validation_obj_views_t,
                                        predictions_t=validation_predictions_t)

    def _createBatches(self, batch_size: int) -> None:
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



        self._batch_model = ForwardModel(ndiffs=batch_size,
                                         indices_t=batch_indices_v,
                                         mods_t=batch_mods_t,
                                         obj_views_t=batch_obj_views_t,
                                         predictions_t=batch_predictions_t)

        self._dataset_indices = dataset_indices
        self._dataset_batch = dataset_batch
        self._new_batch_op = new_batch_op


    def _getBatchAmplitudeLoss(self,
                               predicted_data_t: tf.Tensor,
                               measured_data_t: tf.Tensor) -> tf.Tensor:
        """Get the amplitude (gaussian) loss function for a minibatch.

        The is the amplitude loss, or the loss function for the gaussian noise model. It is a least squares function
        defined as ``1/2 * sum((predicted_data - measured_data)**2)``.

        Parameters
        ----------
        predicted_data_t : tensor(float)
            Diffraction data calculated using the forward model for the current minibatch of scan positions. Uses
            the current values of the object and probe variables.
        measured_data_t : tensor(float)
            Diffraction data measured experimentally for the current minibatch of scan positions.

        Returns
        -------
        loss_t : tensor(float)
            Scalar value for the current loss function.
        """
        loss_t = 0.5 * tf.reduce_sum((predicted_data_t - measured_data_t) ** 2)
        return loss_t

    def initLossAndOptimizers(self,
                              obj_learning_rate: float = 1e-2,
                              probe_learning_rate: float = 1e1) -> None:
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
        with self._graph.as_default():
            batch_loss_t = self._getBatchAmplitudeLoss(self._batch_model.predictions_t,
                                                       self._batch_model.mods_t)

            validation_loss_t = self._getBatchAmplitudeLoss(self._validation.predictions_t,
                                                            self._validation.mods_t)

            obj_optimizer = tf.train.AdamOptimizer(obj_learning_rate)
            obj_minimize_op = obj_optimizer.minimize(batch_loss_t,
                                                     var_list=[self._obj_v])

            self._recons_ops = ReconsLossAndOptimizers(batch_loss_t=batch_loss_t,
                                                       validation_loss_t=validation_loss_t,
                                                       obj_learning_rate=obj_learning_rate,
                                                       obj_opt=obj_optimizer,
                                                       obj_min_op=obj_minimize_op)

            if self.probe_recons:
                self._recons_ops.probe_learning_rate = probe_learning_rate
                self._recons_ops.probe_opt = tf.train.AdamOptimizer(probe_learning_rate)
                self._recons_ops.probe_min_op = self._recons_ops.probe_opt.minimize(batch_loss_t,
                                                                                    var_list=[self._probe_v])

    def initSession(self):
        """Initialize the graph and set up the gradient calculation.

        Run after creating optimizers."""
        assert hasattr(self, '_recons_ops'), "Create optimizers before initializing the session."
        with self._graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(self._new_batch_op)

    def updateOutputs(self) -> None:
        """Populate numpy arrays from the current values of the probe and object tensorflow variables."""
        obj_out, probe_out = self.session.run([self._obj_cmplx_t, self._probe_cmplx_t])
        self.outs.obj = obj_out
        self.outs.probe = probe_out

    def saveOutputs(self, out_name: str) -> Tuple[str]:
        """Save the current probe and object values and the current log to disk.

        The files saved are named as::

            * log file : <out_name>.pkl
            * object transmission function : <out_name>_obj.npy
            * probe transmission function : <out_name>_probe.npy

        Parameters
        ----------
        out_name : str
            Common prefix for the saved output files.

        Returns
        -------
        log_pkl, obj_npy, probe_npy : str
            Names of the saved files for the log, the object function, and the probe function respectively.
        """
        log_pkl = f'{out_name}.pkl'
        obj_npy = f'{out_name}_obj.npy'
        probe_npy = f'{out_name}_probe.npy'
        self.outs.log.to_pickle(log_pkl)
        np.save(obj_npy, self.outs.obj)
        np.save(probe_npy, self.outs.probe)
        return log_pkl, obj_npy, probe_npy

    def run(self,
            validation_frequency: int = 1,
            improvement_threshold: float = 5e-4,
            patience: int = 50,
            patience_increase_factor: float = 1.5,
            max_iters: int = 5000,
            debug_output: bool = True,
            debug_output_frequency: int = 10,
            probe_fixed_epochs: int = 0) -> None:
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

        if debug_output:
            print(f"{'iter':<8}{'epoch':<7}{'train_loss':<12}{'obj_err':<10}{'probe_err':<10}" +
                  f"{'patience':<10}{'val_loss':<10}{'val_best_loss':<13}")

        epochs_this = 0
        log = self.outs.log
        index = len(log)
        for i in range(max_iters):
            ix = index + i

            if self.probe_recons and epochs_this >= probe_fixed_epochs:
                _ = self.session.run(self._recons_ops.probe_min_op)

            lossval, _ = self.session.run([self._recons_ops.batch_loss_t,
                                           self._recons_ops.obj_min_op])
            _ = self.session.run(self._new_batch_op)
            log.loc[ix, 'loss'] = lossval

            if ix == 0:
                log.loc[0, 'epoch'] = 0
                continue
            elif ix % (self._train_full.ndiffs // self.batch_size) != 0:
                log.loc[ix, 'epoch'] = log['epoch'][ix - 1]
                continue

            log.loc[ix, 'epoch'] = log['epoch'][ix - 1] + 1
            epochs_this += 1

            if epochs_this % validation_frequency != 0:
                continue
            validation_lossval = self.session.run(self._recons_ops.validation_loss_t)
            log.loc[ix, 'validation_loss'] = validation_lossval

            obj_registration_error = self.getObjRegistrationError()
            log.loc[ix, 'obj_error'] = obj_registration_error

            probe_registration_error = self.getProbeRegistrationError()
            log.loc[ix, 'probe_error'] = probe_registration_error

            validation_best_loss = np.inf if ix == 0 else log['validation_loss'][:-1].min()

            if validation_lossval <= validation_best_loss:
                if np.abs(validation_lossval - validation_best_loss) > validation_best_loss * improvement_threshold:
                    patience = max(patience, epochs_this * patience_increase_factor)

            log.loc[ix, 'patience'] = patience

            if debug_output and epochs_this % (debug_output_frequency * validation_frequency) == 0:
                print(f'{i:<8} '
                      + f'{epochs_this:<7}'
                      + f'{lossval:<12.7g} '
                      + f'{obj_registration_error:<10.7g} '
                      + f'{probe_registration_error:<10.7g} '
                      + f'{patience:<10.7g} '
                      + f'{validation_lossval:<10.7g} '
                      + f'{validation_best_loss:<13.7g}')

            if epochs_this >= patience:
                break
        self.updateOutputs()

    def genPlotsRecons(self) -> None:
        """Plot the reconstructed probe and object amplitudes and phases."""
        self.updateOutputs()

        plt.figure(figsize=[14, 3])
        plt.subplot(1, 4, 1)
        plt.pcolormesh(np.abs(self.outs.obj), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.pcolormesh(np.angle(self.outs.obj), cmap='gray')
        plt.subplot(1, 4, 3)
        plt.pcolormesh(np.abs(self.outs.probe), cmap='gray')
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.pcolormesh(np.angle(self.outs.probe), cmap='gray')
        plt.colorbar()
        plt.show()

    def genPlotMetrics(self) -> None:
        """Plot the metrics recorded in the log."""
        log = self.outs.log
        fig, axs = plt.subplots(1, 4, figsize=[14, 3])
        axs[0].plot(np.log(log['loss'].dropna()))
        axs[0].set_title('loss')

        axs[1].plot(log['obj_error'].dropna())
        axs[1].set_title('obj_error')

        axs[2].plot(log['probe_error'].dropna())
        axs[2].set_title('probe_error')

        axs[3].plot(np.log(log['validation_loss'].dropna()))
        axs[3].set_title('validation_loss')
        plt.show()

    def _getClipOp(self, max_abs: float = 1.0) -> None:
        """Not used for now"""
        with self.graph.as_default():
            obj_reshaped = tf.reshape(self.tf_obj, [2, -1])
            obj_clipped = tf.clip_by_norm(obj_reshaped, max_abs, axes=[0])
            obj_clipped_reshaped = tf.reshape(obj_clipped, [-1])
            clipped = tf.assign(self.tf_obj, obj_clipped_reshaped, name='clip_op')
        return clipped