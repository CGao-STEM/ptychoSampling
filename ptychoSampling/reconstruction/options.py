import ptychoSampling.reconstruction.forwardmodel_t
import ptychoSampling.reconstruction.lossfn_t
import ptychoSampling.reconstruction.optimization_t

OPTIONS = {"forward models":
               {"farfield": ptychoSampling.reconstruction.forwardmodel_t.FarfieldForwardModelT,
                "nearfield": ptychoSampling.reconstruction.forwardmodel_t.NearfieldForwardModelT,
                "bragg": ptychoSampling.reconstruction.forwardmodel_t.BraggPtychoForwardModelT},

           "loss functions":
               {"least_squared": ptychoSampling.reconstruction.lossfn_t.least_squared_loss_t},

           "optimization_methods": {"adam": ptychoSampling.reconstruction.optimization_t.AdamOptimizer}}

           #"optimization_methods": {"adam": ptychoSampling.reconstruction.optimization_t.getAdamOptimizer,
           #                         "custom": ptychoSampling.reconstruction.optimization_t.getOptimizer}}