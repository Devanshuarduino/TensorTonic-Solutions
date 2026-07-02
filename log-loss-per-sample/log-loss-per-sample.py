import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss for Python lists using the math module.
    """
    loss = []
    for yt, yp in zip(y_true, y_pred):
        # Clip yp manually
        if yp < eps:
            yp = eps
        elif yp > 1 - eps:
            yp = 1 - eps
            
        # Compute loss for this specific sample
        sample_loss = -yt * math.log(yp) - (1 - yt) * math.log(1 - yp)
        loss.append(sample_loss)
        
    return loss