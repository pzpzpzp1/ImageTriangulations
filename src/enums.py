"""
Strategy Enums
"""
from enum import Enum

class SaliencyStrategy(Enum):
    """Strategy for computing saliency maps."""
    MANUAL = "manual"
    # automatic classical saliency methods did not look good :(

class OptStrategy(Enum):
    """Optimization strategy for gradient descent."""
    NONE = "none"
    ADA_DELTA = "adaDelta" 
    RMS_PROP = "RMSProp" # preferred

class DtStrategy(Enum):
    """Time step strategy for optimization."""
    NONE = "none" # constant dt. NOT RECOMMENDED
    ONE_PIX = "onepix" # max vertex displacement per step is 1 pixel
    CONSTRAINED = "constrained" # same as one_pix but also perform area line-search
    
class SubdivisionStrategy(Enum):
    """Strategy for mesh subdivision."""
    EDGE = "edge" # split edges of mesh
    LOOP = "loop" # NOT IMPLEMENTED: split triangles into 4, patch surroundings for conformity

class InitialMesh(Enum):
    """Initial triangle mesh generation strategy."""
    HEXAGONAL = "hexagonal" # highly recommend hexagonal over grid
    GRID = "grid" # grid but cut into triangles with (optionally random) diagonals
