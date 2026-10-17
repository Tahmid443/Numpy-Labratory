"""Mini project: data analysis pipeline."""

import numpy as np

class DataAnalyzer:
    """A simple data analysis toolkit."""
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.data = None
    
    def generate_dataset(self, n_samples=1000):
        """Generate synthetic dataset."""
        self.data = {
            'age': self.rng.integers(18, 80, n_samples),
            'income': self.rng.normal(50000, 20000, n_samples).round(2),
            'score': self.rng.normal(70, 15, n_samples).round(1),
            'category': self.rng.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
            'purchased': self.rng.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
        return self
    
    def summary(self):
        """Print summary statistics."""
        print("Dataset Summary")
        print("-" * 40)
        for col, values in self.data.items():
            if isinstance(values, np.ndarray) and values.dtype.kind in 'if':
                print(f"\n{col}:")
                print(f"  count: {len(values)}")
                print(f"  mean:  {values.mean():.2f}")
                print(f"  std:   {values.std():.2f}")
                print(f"  min:   {values.min():.2f}")
                print(f"  Q1:    {np.percentile(values, 25):.2f}")
                print(f"  Q2:    {np.percentile(values, 50):.2f}")
                print(f"  Q3:    {np.percentile(values, 75):.2f}")
                print(f"  max:   {values.max():.2f}")
    
    def filter(self, mask):
        """Filter data by boolean mask."""
        return DataAnalyzer._from_dict(
            {k: v[mask] if isinstance(v, np.ndarray) else v
             for k, v in self.data.items()}
        )
    
    @classmethod
    def _from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.data = d
        return obj
    
    def groupby(self, group_col, value_col):
        """Group by one column, aggregate another."""
        groups = self.data[group_col]
        values = self.data[value_col]
        unique = np.unique(groups)
        result = {}
        for g in unique:
            mask = groups == g
            result[g] = {
                'count': mask.sum(),
                'mean': values[mask].mean(),
                'std': values[mask].std(),
            }
        return result
    
    def correlation_matrix(self):
        """Compute correlation matrix of numeric columns."""
        numeric_cols = {k: v for k, v in self.data.items()
                       if isinstance(v, np.ndarray) and v.dtype.kind in 'if'}
        matrix = np.column_stack(list(numeric_cols.values()))
        return np.corrcoef(matrix.T), list(numeric_cols.keys())
    
    def find_outliers(self, col, method='iqr'):
        """Find outliers in a column."""
        values = self.data[col]
        if method == 'iqr':
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (values < lower) | (values > upper)
        elif method == 'zscore':
            mask = np.abs((values - values.mean()) / values.std()) > 3
        return np.where(mask)[0]

if __name__ == "__main__":
    analyzer = DataAnalyzer(seed=42)
    analyzer.generate_dataset(1000)
    
    print("=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    analyzer.summary()
    
    print("\n" + "=" * 50)
    print("GROUP BY CATEGORY")
    print("=" * 50)
    groups = analyzer.groupby('category', 'income')
    for cat, stats in groups.items():
        print(f"\nCategory {cat}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}")
    
    print("\n" + "=" * 50)
    print("CORRELATIONS")
    print("=" * 50)
    corr, cols = analyzer.correlation_matrix()
    print(f"{'':<10}" + "".join(f"{c:>12}" for c in cols))
    for i, row_name in enumerate(cols):
        print(f"{row_name:<10}" + "".join(f"{corr[i,j]:>12.3f}" for j in range(len(cols))))
    
    print("\n" + "=" * 50)
    print("OUTLIERS (income, IQR method)")
    print("=" * 50)
    outliers = analyzer.find_outliers('income', 'iqr')
    print(f"Found {len(outliers)} outliers at indices {outliers[:10]}")
    
    print("\n" + "=" * 50)
    print("FILTERED DATA (age > 50, purchased)")
    print("=" * 50)
    mask = (analyzer.data['age'] > 50) & (analyzer.data['purchased'] == 1)
    filtered = analyzer.filter(mask)
    print(f"Filtered count: {len(filtered.data['age'])}")
    print(f"Mean income: {filtered.data['income'].mean():.2f}")