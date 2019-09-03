import tensorflow as tf
import dataclasses as dt

@dt.dataclass
class Optimizer:
    """Does not currently support optimizers that are not subclassed from tf.train.Optimizer"""
    optimizer: tf.train.Optimizer
    optimizer_args: dict = dt.field(default_factory=dict)
    initial_update_delay: int = 0
    update_frequency: int = 1
    checkpoint_frequency: int = 100
    _default_optimizer_args: dict = dt.field(default={}, init=False)

    def __post_init__(self):
        for key in self._default_optimizer_args:
            if key not in self.optimizer_args:
                self.optimizer_args[key] = self._default_optimizer_args[key]

class AdamOptimizer(Optimizer):
    optimizer: tf.train.Optimizer = tf.train.AdamOptimizer
    optimizer_args: dict = dt.field(default_factory=dict)
    _default_optimizer_args: dict = dt.field(default={'learning_rate': 1e-2}, init=False)




