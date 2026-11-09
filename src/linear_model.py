"""NumPy-only predictions, half-MSE cost and analytic gradient (tasks 3–4)."""
import numpy as np

def validate_inputs(X, y, theta):
    X=np.asarray(X,dtype=float)
    y=np.asarray(y,dtype=float)
    theta=np.asarray(theta,dtype=float)
    if X.ndim!=2 or len(X)==0 or X.shape[1]==0:
        raise ValueError('X must be a nonempty 2D matrix.')
    if y.shape!=(len(X),) or theta.shape!=(X.shape[1],):
        raise ValueError('Expected y shape (n,) and theta shape (d,).')
    if not all(np.isfinite(v).all() for v in (X,y,theta)):
        raise ValueError('Inputs must be finite.')
    return X,y,theta

def predict(X, theta):
    X=np.asarray(X,dtype=float); theta=np.asarray(theta,dtype=float)
    if X.ndim!=2 or theta.shape!=(X.shape[1],) or not np.isfinite(X).all() or not np.isfinite(theta).all():
        raise ValueError('Invalid shape or nonfinite prediction input.')
    with np.errstate(over='raise',invalid='raise'):
        result=X@theta
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite prediction.')
    return result

def compute_cost(X, y, theta):
    X,y,theta=validate_inputs(X,y,theta)
    with np.errstate(over='raise',invalid='raise'):
        error=X@theta-y
        value=float(error@error/(2*len(y)))
    if not np.isfinite(value): raise FloatingPointError('Nonfinite cost.')
    return value

def compute_gradient(X, y, theta):
    X,y,theta=validate_inputs(X,y,theta)
    with np.errstate(over='raise',invalid='raise'):
        result=X.T@(X@theta-y)/len(y)
    if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite gradient.')
    return result
