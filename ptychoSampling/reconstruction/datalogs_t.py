from pandas import DataFrame
import numpy as np
from ptychoSampling.logger import logger
import tensorflow as tf
import dataclasses as dt
from skimage.feature import register_translation


@dt.dataclass
class SimpleMetric:
    title: str


@dt.dataclass
class CustomTensorMetric(SimpleMetric):
    tensor: tf.Tensor
    log_epoch_frequency: int
    registration: bool = False
    true: np.ndarray = None
    columns: list = dt.field(default_factory=list, init=False)  # not in use right now

    def __post_init__(self):
        if self.registration and self.true is None:
            e = ValueError("Require true data for registration.")
            logger.error(e)
            raise e
        if not self.registration:
            self.columns = [self.title]
        else:
            appends = ['_err']  # , '_shift', '_phase']
            self.columns = [self.title + string for string in appends]


class DataLogs:
    def __init__(self, save_name: str = "", save_update_freq: int=100):
        self._save_name = save_name
        self._save_update_freq = save_update_freq
        self._datalog_items = []


    def addSimpleMetric(self, title: str):
        self._datalog_items.append(SimpleMetric(title))


    def addCustomTensorMetric(self, title: str,
                              tensor: tf.Tensor,
                              log_epoch_frequency,
                              **kwargs):

        self._datalog_items.append(CustomTensorMetric(title, tensor, log_epoch_frequency, **kwargs))


    @staticmethod
    def _register(test: np.ndarray,
                  true: np.ndarray) -> float:
        shift, err, phase = register_translation(test, true, upsample_factor=10)
        shift, err, phase = register_translation(test * np.exp(-1j * phase), true, upsample_factor=10)
        return err

    def _getItemFromTitle(self, key: str):
        for item in self._datalog_items:
            if item.title == key:
                return item

    def getCustomTensorMetrics(self, epoch):
        tensors = {}
        for item in self._datalog_items:
            if isinstance(item, CustomTensorMetric):
                if epoch % item.log_epoch_frequency == 0:
                    tensors[item.title] = item.tensor
        return tensors

    def logStep(self, step, log_values_this_step: dict):
        for key in log_values_this_step:
            item = self._getItemFromTitle(key)
            if item.registration:
                value = self._register(log_values_this_step[key], item.true)
            else:
                value = log_values_this_step[key]
            self.dataframe.loc[step, key] = value

    def printDebugOutput(self, epoch):
        header = True if epoch==1 else False
        print(self.dataframe.iloc[-1].to_string(float_format="10.3g", header=header))


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

    #def _saveCheckpoint(self, iteration):
    #    if not hasattr(self, "_name"):
    #        self._name = self._checkpoint_name + ".csv"
    #        self.dataframe.to_csv(self._name, sep="\t", header=True, float_format=)



