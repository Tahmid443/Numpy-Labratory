"""Common random distributions."""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Normal (Gaussian)
normal = rng.normal(loc=0, scale=1, size=1000)
print(f"Normal: mean={normal.mean():.3f}, std={normal.std():.3f}")

# Uniform
uniform = rng.uniform(low=0, high=1, size=1000)
print(f"Uniform: min={uniform.min():.3f}, max={uniform.max():.3f}")

# Binomial
binomial = rng.binomial(n=10, p=0.5, size=1000)
print(f"Binomial: mean={binomial.mean():.2f}, expected=5")

# Poisson
poisson = rng.poisson(lam=3, size=1000)
print(f"Poisson: mean={poisson.mean():.2f}, expected=3")

# Exponential
exponential = rng.exponential(scale=2, size=1000)
print(f"Exponential: mean={exponential.mean():.2f}, expected=2")

# Beta
beta = rng.beta(a=2, b=5, size=1000)
print(f"Beta: mean={beta.mean():.3f}, expected={2/7:.3f}")

# Gamma
gamma = rng.gamma(shape=2, scale=2, size=1000)
print(f"Gamma: mean={gamma.mean():.2f}, expected=4")

# Chi-square
chisq = rng.chisquare(df=5, size=1000)
print(f"Chi-square: mean={chisq.mean():.2f}, expected=5")

# Multivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
mv = rng.multivariate_normal(mean, cov, size=1000)
print(f"\nMultivariate normal shape: {mv.shape}")
print(f"Means: {mv.mean(axis=0)}")

# Simple visualization (optional)
try:
    samples = {
        'Normal': rng.normal(0, 1, 1000),
        'Uniform': rng.uniform(0, 1, 1000),
        'Exponential': rng.exponential(1, 1000),
        'Poisson': rng.poisson(5, 1000),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (name, data) in zip(axes.flat, samples.items()):
        ax.hist(data, bins=30, edgecolor='black')
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig('distributions.png')
    print("\nPlot saved to distributions.png")
except ImportError:
    print("\nmatplotlib not installed, skipping plots")