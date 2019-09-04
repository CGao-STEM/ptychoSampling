from pandas import DataFrame
import numpy as np
from ptychoSampling.logger import logger
import tensorflow as tf
import dataclasses as dt
from skimage.feature import register_translation


@dt.dataclass
class DataItem:
    title: str
    current_value: float = None
    is_tensor: bool = False
    tensor: tf.Tensor = None
    registration: bool = False
    true: np.ndarray = None

    def __post_init__(self):
        if self.registration and self.true is None:
            e = ValueError("Require true data for registration.")
            logger.error(e)
            raise e


class DataLogs:
    def __init__(self):
        self.datalog_items = []


    def addRegistration(self, title:str,
                        is_tensor: bool,
                        tensor: tf.Tensor,
                        true: np.ndarray):
        self.datalog_items.append(DataItem(title, is_tensor, tensor, registration=True, true=true))

    def addLog(self, title:str,
               is_tensor: bool,
               tensor: tf.Tensor):
        self.datalog_items.append((DataItem(title, is_tensor, tensor)))


    @staticmethod
    def _register(test: np.ndarray,
                  true: np.ndarray) -> float:
        shift, err, phase = register_translation(test, true, upsample_factor=10)
        shift, err, phase = register_translation(test * np.exp(-1j * phase), true, upsample_factor=10)
        return err

    def getItemFromTitle(self, key: str):
        for item in self.datalog_items:
            if item.title == key:
                return item


    def getStepItems(self, global_step) -> list:
        out_items = []
        for item in self.datalog_items:
                #out_items.append(item)
                out_items.append(item.title)
        #out_tensors = {item.title: item.tensor for item in out_items}
        #return out_tensors
        return out_items

    def logStep(self, step: int, log_values: dict):
        for key in log_values:
            if item.registration:
                current_value = self._register(log_values[item.title], item.true)
            else:
                current_value = log_values[item.title]
            self.dataframe.loc[step, item.title] = out_float


    def _addToDataframe(self, log_values, global_step):

        #for i, outval in enumerate(log_values):
        #    if self._this_step_out_items[i].registration:
        #        out_float = self._register(outval, self._this_step_out_items[i].true)
        #    else:
        #        out_float = outval
        #    self.dataframe[iteration, self._this_step_out_items[i].title] = out_float

    #def addToPrint(self):
    #    if not hasattr(self, "_print_str"):
    #        self._print_str = self.dataframe.iloc[-1].to_string(float_format="%10.3g", header=True)
    #    else:
    #        self._print_str = self.dataframe.iloc[-1].to_string(float_format="%10.3g", header=False)
    #    print(self._print_str)

    def finalize(self):
        columns = []
        for item in self.datalog_items:
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



