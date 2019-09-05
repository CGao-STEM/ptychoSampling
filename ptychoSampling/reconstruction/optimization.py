import tensorflow as tf
import dataclasses as dt


def getOptimizer(optimizer_class: tf.train.Optimizer,
                 **optimizer_kwargs):
    """Does not currently support optimizers that are not subclassed from tf.train.Optimizer"""
    return optimizer_class(**optimizer_kwargs)

def getAdamOptimizer(learning_rate=1e-2, **optimizer_kwargs):
    return getOptimizer(tf.train.AdamOptimizer, learning_rate=learning_rate, **optimizer_kwargs)



