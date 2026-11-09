"""Independent analytic oracle for TV3 verification; leader owns task 5."""
import numpy as np
from .linear_model import validate_inputs

def analytical_reference(X,y):
    X,y,_=validate_inputs(X,y,np.zeros(np.asarray(X).shape[1]))
    if X.shape[1]!=2 or not np.allclose(X[:,0],1.): raise ValueError('Expected intercept and one feature.')
    x=X[:,1]; xc=x-x.mean(); yc=y-y.mean()
    variance=float(xc@xc)
    if variance<=0: raise ValueError('Unique solution requires nonconstant input.')
    slope=float(xc@yc/variance)
    covariance_solution=np.array([y.mean()-slope*x.mean(),slope])
    normal_solution=np.linalg.solve(X.T@X,X.T@y)
    np.testing.assert_allclose(normal_solution,covariance_solution,atol=1e-9,rtol=0)
    return normal_solution

def metrics(y,pred):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    if y.ndim!=1 or y.shape!=pred.shape or len(y)==0 or not np.isfinite([y,pred]).all(): raise ValueError('Invalid evaluation inputs.')
    mse=float(np.mean((y-pred)**2)); sst=float(np.sum((y-y.mean())**2))
    return {'MSE':mse,'RMSE':float(np.sqrt(mse)),'R2':None if sst==0 else float(1-np.sum((y-pred)**2)/sst)}
