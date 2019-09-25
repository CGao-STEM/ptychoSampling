import ptychoSampling.reconstruction.forwardmodel_t
import ptychoSampling.reconstruction.lossfn_t
import ptychoSampling.reconstruction.optimization

OPTIONS = {"forward models":
               {"farfield": ptychoSampling.reconstruction.forwardmodel_t.FarfieldForwardModelT,
                "nearfield": ptychoSampling.reconstruction.forwardmodel_t.NearfieldForwardModelT,
                "bragg": ptychoSampling.reconstruction.forwardmodel_t.BraggPtychoForwardModelT},

           "loss functions":
               {"least_squared": ptychoSampling.reconstruction.lossfn_t.least_squared_loss_t},

           "optimization_methods": {"adam": ptychoSampling.reconstruction.optimization.getAdamOptimizer,
                                    "custom": ptychoSampling.reconstruction.optimization.getOptimizer}}