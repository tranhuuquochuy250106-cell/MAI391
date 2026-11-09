"""Analytical least-squares solution for Q5."""
import numpy as np

from .reference import analytical_reference


def solve_least_squares(X, y, at_raw, scaling):
    """Solve the normal equations and cross-check using raw-temperature moments."""
    theta = analytical_reference(X, y)

    at = np.asarray(at_raw, dtype=float)
    centered_at = at - at.mean()
    raw_slope = (centered_at @ (y - y.mean())) / (centered_at @ centered_at)
    raw_covariance = np.array([y.mean() - raw_slope * at.mean(), raw_slope])

    theta_raw = to_raw_scale(theta, scaling)
    np.testing.assert_allclose(theta_raw, raw_covariance, atol=1e-9, rtol=0)
    return theta, theta_raw, raw_covariance


def to_raw_scale(theta, scaling):
    """Return coefficients for PE = intercept + slope * AT in original units."""
    slope = theta[1] / scaling['std_AT']
    return np.array([theta[0] - slope * scaling['mean_AT'], slope])
