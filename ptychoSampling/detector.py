import numpy as np
import dataclasses as dt

@dt.dataclass(frozen=True)
class Detector:
    npix : int
    pixel_size : float
    obj_dist : float