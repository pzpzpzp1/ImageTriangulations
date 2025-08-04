"""
Optimization algorithms for gradient based optimization
"""
import numpy as np
from typing import Tuple, Optional


class AdaptiveOptimizer:
    """
    Adaptive gradient descent optimizer implementing AdaDelta and RMSProp.
    Custom implementation with per-vertex learning rates.
    """
    
    def __init__(self):
        """Initialize optimizer state."""
        self.exp_g2: Optional[np.ndarray] = None  # Exponential moving average of gradient squares
        self.exp_th2: Optional[np.ndarray] = None  # Exponential moving average of update squares
        self.gamma = 0.7  # Decay rate
        self.eps = 1e-8   # Small constant for numerical stability
    
    def get_descent_direction(self, grad: np.ndarray, rate_flag: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get descent direction using AdaDelta or RMSProp.
        
        Args:
            grad: Gradient array (nX, 2)
            rate_flag: 1 for AdaDelta, 2 for RMSProp
            
        Returns:
            desc_dir: Descent direction (nX, 2) 
            rates: Learning rates per vertex (nX,)
        """
        nX = grad.shape[0]
        
        # Initialize on first call or if vertex count changed
        if self.exp_g2 is None or self.exp_g2.shape[0] != grad.shape[0]:
            desc_dir = -grad
            self.exp_g2 = np.sum(grad**2, axis=1) 
            self.exp_th2 = np.sum(desc_dir**2, axis=1)
            rates = np.ones(nX)
            return desc_dir, rates
        
        # Update exponential moving average of gradient squares
        self.exp_g2 = self.gamma * self.exp_g2 + (1 - self.gamma) * np.sum(grad**2, axis=1)
        
        # Compute learning rates
        if rate_flag == 1:
            # AdaDelta
            rates = np.sqrt((self.exp_th2 + self.eps) / (self.exp_g2 + self.eps))
        elif rate_flag == 2:
            # RMSProp
            rates = np.sqrt(1.0 / (self.exp_g2 + self.eps))
        else:
            raise ValueError(f"Unknown rate_flag: {rate_flag}")
        
        # Compute descent direction with per-vertex rates
        desc_dir = -rates[:, np.newaxis] * grad
        
        # update decaying average of previous descent dir square norms
        self.exp_th2 = self.gamma * self.exp_th2 + (1 - self.gamma) * np.sum(desc_dir**2, axis=1)
        
        return desc_dir, rates

def get_adadelta_desc_dir(grad: np.ndarray, rate_flag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Static wrapper for adaptive optimizer.
    Maintains function-cached state for cleaner usage. No class init needed.
    Only single threaded support
    
    Args:
        grad: Gradient array (nX, 2)
        rate_flag: 1 for AdaDelta, 2 for RMSProp
        
    Returns:
        desc_dir: Descent direction (nX, 2)
        rates: Learning rates per vertex (nX,)
    """
    if not hasattr(get_adadelta_desc_dir, '_optimizer'):
        get_adadelta_desc_dir._optimizer = AdaptiveOptimizer()
    
    return get_adadelta_desc_dir._optimizer.get_descent_direction(grad, rate_flag)


