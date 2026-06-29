import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Convert input to a numpy array to handle lists/scalars seamlessly
    x = np.asarray(x, dtype=float)
    
    # Compute the sigmoid function element-wise
    return 1 / (1 + np.exp(-x))