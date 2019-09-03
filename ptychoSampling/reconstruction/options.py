from ptychoSampling.reconstruction.forwardmodel_t import FarfieldForwardModelT
from ptychoSampling.reconstruction.lossfn_t import least_squared_loss_t
from ptychoSampling.reconstruction.optimization import Optimizer, AdamOptimizer

OPTIONS = {"forward models":
               {"farfield": FarfieldForwardModelT},

           "loss functions":
               {"least_squared": least_squared_loss_t},

           "optimizers": {"adam": AdamOptimizer,
                          "custom": Optimizer}}