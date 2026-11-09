"""Batch GD with simultaneous updates and paired, finite histories."""
import numpy as np
from .linear_model import validate_inputs, compute_cost, compute_gradient

def gradient_descent(X,y,theta_initial=None,learning_rate=.1,max_iter=10000,
                     tolerance=1e-6,divergence_factor=1e12):
    if theta_initial is None:
        candidate=np.asarray(X)
        if candidate.ndim!=2: raise ValueError('X must be 2D.')
        theta_initial=np.zeros(candidate.shape[1])
    X,y,theta=validate_inputs(X,y,theta_initial)
    theta=theta.copy()
    if not np.isfinite(learning_rate) or learning_rate<=0: raise ValueError('learning_rate must be positive finite.')
    if isinstance(max_iter,bool) or not isinstance(max_iter,(int,np.integer)) or max_iter<0: raise ValueError('max_iter must be a nonnegative integer.')
    if not np.isfinite(tolerance) or tolerance<0: raise ValueError('tolerance must be nonnegative finite.')
    if not np.isfinite(divergence_factor) or divergence_factor<=1: raise ValueError('divergence_factor must exceed one.')
    costs=[compute_cost(X,y,theta)]; thetas=[theta.copy()]
    threshold=divergence_factor*max(costs[0],1.)
    status='max_iter'
    for k in range(max_iter+1):
        try:
            g=compute_gradient(X,y,theta)
            # Stable norm avoids overflow while checking a finite gradient.
            m=float(np.max(np.abs(g)))
            norm=0. if m==0 else float(m*np.linalg.norm(g/m))
        except FloatingPointError:
            status='nonfinite'; break
        if norm<=tolerance:
            status='converged'; break
        if k==max_iter: break
        try:
            with np.errstate(over='raise',invalid='raise'):
                updated=theta-learning_rate*g
            cost=compute_cost(X,y,updated)
        except (FloatingPointError,ValueError):
            status='nonfinite'; break
        theta=updated
        thetas.append(theta.copy()); costs.append(cost)
        if cost>threshold:
            status='diverged'; break
    # Diagnostics must describe the last stored state, including divergence stops.
    try:
        final_g=compute_gradient(X,y,theta)
        final_m=float(np.max(np.abs(final_g)))
        with np.errstate(over='raise',invalid='raise'):
            norm=0. if final_m==0 else float(final_m*np.linalg.norm(final_g/final_m))
    except FloatingPointError:
        norm=None
    return {'theta_final':theta,'theta_history':np.asarray(thetas),
            'cost_history':np.asarray(costs),'n_iter':len(costs)-1,
            'status':status,'converged':status=='converged',
            'gradient_norm':norm,'learning_rate':float(learning_rate),
            'tolerance':float(tolerance),'max_iter':int(max_iter)}
