import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the Gaussian Process kernel function
def kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Squared Exponential Kernel."""
    sqdist = np.sum((x1[:, None] - x2[None, :]) ** 2, axis=2)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)

# Generate random points for training
np.random.seed(42)
X_train = np.random.uniform(-5, 5, 5).reshape(-1, 1)  # 5 random points in 1D
y_train = np.sin(X_train).ravel() + 0.1 * np.random.randn(len(X_train))  # Noisy observations

# Generate points for prediction
X_pred = np.linspace(-6, 6, 100).reshape(-1, 1)

# Compute the covariance matrices
K_train = kernel(X_train, X_train[:, None])
K_pred_train = kernel(X_pred, X_train[:, None])
K_pred_pred = kernel(X_pred, X_pred[:, None])

# Add noise to the diagonal of the training kernel matrix
noise_variance = 1e-4
K_train += noise_variance * np.eye(len(X_train))

# Compute the posterior mean and covariance
K_inv = np.linalg.inv(K_train)
posterior_mean = K_pred_train @ K_inv @ y_train
posterior_cov = K_pred_pred - K_pred_train @ K_inv @ K_pred_train.T

# Sample from the posterior distribution
samples = multivariate_normal.rvs(mean=posterior_mean, cov=posterior_cov, size=3)

# Plot the Gaussian process
plt.figure(figsize=(10, 6))
plt.plot(X_pred, posterior_mean, 'r-', label='Mean')
plt.fill_between(
    X_pred.ravel(),
    posterior_mean - 2 * np.sqrt(np.diag(posterior_cov)),
    posterior_mean + 2 * np.sqrt(np.diag(posterior_cov)),
    color='pink',
    alpha=0.5,
    label='2 SD Confidence Interval'
)
plt.plot(X_train, y_train, 'ko', label='Training Points')
for i, sample in enumerate(samples):
    plt.plot(X_pred, sample, label=f'Sample {i+1}')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('1D Gaussian Process')
plt.legend()
plt.grid()
plt.show()
