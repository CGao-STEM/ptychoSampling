from pandas import DataFrame
import numpy as np
from ptychoSampling.logger import logger
import tensorflow as tf
import dataclasses as dt
from skimage.feature import register_translation

@dt.dataclass
class DataItem:
    title: str
    test_t: tf.Tensor
    log_freq: int
    registration: bool = False
    true: np.ndarray = None
    columns: list = dt.field(default_factory=list, init=False) # not using this for now

    def __post_init__(self):
        if self.registration and self.true is None:
            e = ValueError("Require true data for registration.")
            logger.error(e)
            raise e
        if not self.registration:
            self.columns = [self.title]
        else:
            appends = ['_err']#, '_shift', '_phase']
            self.columns = [self.title + string for string in appends]



class DataLogs:
    def __init__(self, global_print_freq: int = 100,
                 save_name: str = "",
                 save_update_freq: int=100):
        self._global_print_freq = global_print_freq
        self._save_name = save_name
        self._save_update_freq = save_update_freq
        self._datalog_items = []


    def addRegistration(self, title: str,
                         test_t: tf.Tensor,
                         true: np.ndarray,
                         log_freq: int):
        self._datalog_items.append(DataItem(title, test_t=test_t, log_freq=log_freq,
                                            registration=True, true=true))

    def addTensorLog(self, title, tensor, log_freq):
        self._datalog_items.append(DataItem(title, test_t=tensor, log_freq=log_freq))
        pass

    @staticmethod
    def _register(test: np.ndarray,
                  true: np.ndarray) -> float:
        shift, err, phase = register_translation(test, true, upsample_factor=10)
        shift, err, phase = register_translation(test * np.exp(-1j * phase), true, upsample_factor=10)
        return err


    def getStepTensors(self, global_step):
        self._this_step_out_items = []
        for item in self._datalog_items:
            if global_step % item.log_freq:
                self._this_step_out_items.append(item)
        out_tensors = [item.test_t for item in self._this_step_out_items]
        return out_tensors

    def logStep(self, log_values, global_step):
        #out_items = []
        #for item in self._datalog_items:
        #    if global_step % item.log_freq:
        #        out_items.append(item)
        #out_ops = [item.test_t for item in out_items]
        #outs = self._session_t.run(out_ops)
        for i, outval in enumerate(log_values):
            if self._this_step_out_items[i].registration:
                out_float = self._register(outval, self._this_step_out_items[i].true)
            else:
                out_float = outval
            self.dataframe[global_step, self._this_step_out_items[i].title] = out_float

        self._addToDataframe(log_values, global_step)
        if self._global_print_freq > 0 and (global_step % self._global_print_freq == 0):
            self._addToPrint()


    def _addToDataframe(self, log_values, global_step):
        for i, outval in enumerate(log_values):
            if self._this_step_out_items[i].registration:
                out_float = self._register(outval, self._this_step_out_items[i].true)
            else:
                out_float = outval
            self.dataframe[global_step, self._this_step_out_items[i].title] = out_float

    def _addToPrint(self):
        if not hasattr(self, "_print_str"):
            self._print_str = self.dataframe.iloc[-1].to_string(float_format="%10.3g", header=True)
        else:
            self._print_str = self.dataframe.iloc[-1].to_string(float_format="%10.3g", header=False)
        print(self._print_str)

    def finalize(self):
        columns = []
        for item in self._datalog_items:
            columns.append(item.title)
        logger.info("Initializing the log outputs...")
        self.dataframe = DataFrame(columns=columns, dtype='float32')
        self.dataframe.loc[0] = np.nan

    def _checkFinalized(self):
        if not hasattr(self, "dataframe"):
            e = AttributeError("Cannot add item to the log file after starting the optimization. "
                               + "The log file remains unchanged. Only the print output is affected.")
            logger.warning(e)

    #def _saveCheckpoint(self, global_step):
    #    if not hasattr(self, "_name"):
    #        self._name = self._checkpoint_name + ".csv"
    #        self.dataframe.to_csv(self._name, sep="\t", header=True, float_format=)



