import tensorflow as tf
import numpy as np

def least_squared_loss_t(predicted_data_t, measured_data_t):
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
        return 0.5 * tf.reduce_sum((predicted_data_t - measured_data_t) ** 2)
    

