import tensorflow as tf
import numpy as np

def least_squared_loss_t(predicted_data_t, measured_data_t):
        return 0.5 * tf.reduce_sum((predicted_data_t - measured_data_t) ** 2)
    

