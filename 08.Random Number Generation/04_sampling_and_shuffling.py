"""Sampling and shuffling."""

import numpy as np

rng = np.random.default_rng(42)
data = np.arange(20)

print(f"Data: {data}")

# Shuffle in-place
rng.shuffle(data)
print(f"After shuffle: {data}")

# permutation (returns copy)
original = np.arange(10)
shuffled = rng.permutation(original)
print(f"\nOriginal: {original}")
print(f"Permutation: {shuffled}")

# Random sampling with replacement
samples = rng.choice(10, size=5, replace=True)
print(f"\nSample with replacement: {samples}")

# Without replacement
samples = rng.choice(10, size=5, replace=False)
print(f"Sample without replacement: {samples}")

# Weighted sampling
items = ['A', 'B', 'C', 'D']
weights = [0.1, 0.2, 0.3, 0.4]
samples = rng.choice(items, size=20, p=weights)
print(f"\nWeighted samples: {samples}")

# Stratified sampling
data = np.arange(100)
rng.shuffle(data)
strata = data.reshape(5, 20)
samples = [rng.choice(s, 3, replace=False) for s in strata]
print(f"\nStratified sample: {np.concatenate(samples)}")

# Bootstrap sampling
def bootstrap(data, n_bootstrap=3, sample_size=None):
    if sample_size is None:
        sample_size = len(data)
    samples = []
    for _ in range(n_bootstrap):
        samples.append(rng.choice(data, size=sample_size, replace=True))
    return np.array(samples)

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
boots = bootstrap(data, n_bootstrap=3)
print(f"\nBootstrap samples:\n{boots}")
print(f"Bootstrap means: {boots.mean(axis=1)}")

# Train-test split
all_data = np.arange(100)
rng.shuffle(all_data)
split = int(0.8 * len(all_data))
train, test = all_data[:split], all_data[split:]
print(f"\nTrain size: {len(train)}, Test size: {len(test)}")