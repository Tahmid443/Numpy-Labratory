"""Monte Carlo simulation."""

import numpy as np

def estimate_pi(n=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = rng.random(n)
    inside = (x**2 + y**2) <= 1
    return 4 * inside.mean()

def option_pricing(S0=100, K=100, T=1, r=0.05, sigma=0.2, n=100_000, seed=42):
    """Black-Scholes via Monte Carlo."""
    rng = np.random.default_rng(seed)
    dt = T
    z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    call_payoff = np.maximum(ST - K, 0)
    put_payoff = np.maximum(K - ST, 0)
    
    call_price = np.exp(-r * T) * call_payoff.mean()
    put_price = np.exp(-r * T) * put_payoff.mean()
    
    return call_price, put_price, call_payoff.std() / np.sqrt(n), put_payoff.std() / np.sqrt(n)

def random_walk_simulation(n_paths=1000, n_steps=252, seed=42):
    rng = np.random.default_rng(seed)
    steps = rng.choice([-1, 1], size=(n_paths, n_steps))
    paths = np.cumsum(steps, axis=1)
    final = paths[:, -1]
    return {
        'mean': final.mean(),
        'std': final.std(),
        'min': final.min(),
        'max': final.max(),
        'pct_positive': (final > 0).mean(),
    }

def risk_simulation(seed=42):
    """Simulate portfolio VaR."""
    rng = np.random.default_rng(seed)
    n = 100_000
    # 3 assets with correlations
    corr = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, 3))
    returns = z @ L.T
    returns *= np.array([0.02, 0.015, 0.01])  # scale to volatility
    returns += np.array([0.001, 0.0008, 0.0005])  # drift
    
    weights = np.array([0.5, 0.3, 0.2])
    portfolio_returns = returns @ weights
    
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    return {
        'VaR_95%': var_95,
        'VaR_99%': var_99,
        'CVaR_95%': cvar_95,
        'mean_return': portfolio_returns.mean(),
        'std_return': portfolio_returns.std(),
    }

if __name__ == "__main__":
    print("=" * 50)
    print("MONTE CARLO ESTIMATION OF PI")
    print("=" * 50)
    for n in [1000, 10_000, 100_000, 1_000_000]:
        est = estimate_pi(n)
        err = abs(est - np.pi)
        print(f"n={n:>8,}: π ≈ {est:.6f}, error={err:.6f}")
    
    print("\n" + "=" * 50)
    print("OPTION PRICING")
    print("=" * 50)
    call, put, call_se, put_se = option_pricing()
    print(f"Call price: {call:.4f} (±{call_se:.4f})")
    print(f"Put price:  {put:.4f} (±{put_se:.4f})")
    
    print("\n" + "=" * 50)
    print("RANDOM WALK")
    print("=" * 50)
    results = random_walk_simulation()
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 50)
    print("PORTFOLIO RISK")
    print("=" * 50)
    risk = risk_simulation()
    for k, v in risk.items():
        print(f"  {k}: {v:.6f}")