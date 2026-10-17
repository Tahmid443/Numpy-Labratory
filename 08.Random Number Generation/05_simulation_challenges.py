"""Challenges: simulations."""

import numpy as np

def challenge_1():
    """Estimate pi using Monte Carlo."""
    rng = np.random.default_rng(42)
    n = 1_000_000
    x = rng.random(n)
    y = rng.random(n)
    inside = (x**2 + y**2) <= 1
    return 4 * inside.mean()

def challenge_2():
    """Simulate 1000 coin flips."""
    rng = np.random.default_rng(42)
    flips = rng.integers(0, 2, 1000)
    return flips.sum(), 1000 - flips.sum()

def challenge_3():
    """Roll two dice 10000 times, get distribution."""
    rng = np.random.default_rng(42)
    d1 = rng.integers(1, 7, 10000)
    d2 = rng.integers(1, 7, 10000)
    sums = d1 + d2
    return np.bincount(sums)[2:13]

def challenge_4():
    """Simulate random walk."""
    rng = np.random.default_rng(42)
    steps = rng.choice([-1, 1], 1000)
    return np.cumsum(steps)[-1]

def challenge_5():
    """Birthday paradox: probability of shared birthday in 23 people."""
    rng = np.random.default_rng(42)
    n_trials = 10_000
    shared = 0
    for _ in range(n_trials):
        birthdays = rng.integers(0, 365, 23)
        if len(np.unique(birthdays)) < 23:
            shared += 1
    return shared / n_trials

def challenge_6():
    """Simulate dice game: roll until 6, count rolls."""
    rng = np.random.default_rng(42)
    results = []
    for _ in range(1000):
        count = 0
        while True:
            count += 1
            if rng.integers(1, 7) == 6:
                break
        results.append(count)
    return np.mean(results)

def challenge_7():
    """Buffon's needle simulation."""
    rng = np.random.default_rng(42)
    n = 100_000
    L, d = 1, 1  # needle length, line spacing
    y = rng.random(n) * d / 2
    theta = rng.random(n) * np.pi / 2
    crosses = y <= (L / 2) * np.sin(theta)
    return 2 * L * n / (crosses.sum() * d)

def challenge_8():
    """Estimate e using random sums."""
    rng = np.random.default_rng(42)
    n_trials = 10_000
    total = 0
    for _ in range(n_trials):
        s = 0
        count = 0
        while s < 1:
            s += rng.random()
            count += 1
        total += count
    return total / n_trials

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}: {globals()[f'challenge_{i}']()}")