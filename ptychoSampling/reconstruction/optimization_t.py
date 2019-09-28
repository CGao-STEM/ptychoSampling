import tensorflow as tf
import abc

class Optimizer(abc.ABC):
    def __init__(self, init_args: dict = {},
                 minimize_args: dict = {},
                 initial_update_delay: int = 0,
                 update_frequency: int = 1):
        self._init_args = init_args
        self._minimize_args = minimize_args
        self.initial_update_delay = initial_update_delay
        self.update_frequency = update_frequency

    @property
    @abc.abstractmethod
    def minimize_op(self):
        pass

class AdamOptimizer(Optimizer):
    def __init__(self, *args: int,
                 **kwargs: int):
        super().__init__(*args, **kwargs)
        if "learning_rate" not in self._init_args:
            self._init_args["learning_rate"] = 1e-2
        self._optimizer = tf.train.AdamOptimizer(**self._init_args)
        self._minimize_op = self._optimizer.minimize(**self._minimize_args)

    @property
    def minimize_op(self):
        return self._minimize_op

#class LMAOptimizer:
#    def __init__(self, ):
#        pass

#def getOptimizer(optimizer_class: tf.train.Optimizer,
#                 **optimizer_kwargs):
#    """Does not currently support optimizers that are not subclassed from tf.train.Optimizer"""
#    return optimizer_class(**optimizer_kwargs)
#
#def getAdamOptimizer(learning_rate=1e-2, **optimizer_kwargs):
#    return getOptimizer(tf.train.AdamOptimizer, learning_rate=learning_rate, **optimizer_kwargs)




