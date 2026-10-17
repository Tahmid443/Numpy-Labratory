"""Analyze sales data."""

import numpy as np

def generate_sales_data(seed=42):
    rng = np.random.default_rng(seed)
    # 4 quarters x 12 months x 5 products
    sales = rng.integers(100, 1000, size=(12, 5))
    return sales

def analyze(sales):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    products = ['A', 'B', 'C', 'D', 'E']
    
    print("Sales Data (months x products):")
    print(f"{'':<6}" + "".join(f"{p:>8}" for p in products))
    for i, m in enumerate(months):
        print(f"{m:<6}" + "".join(f"{v:>8}" for v in sales[i]))
    
    print(f"\nTotal sales per product:")
    for i, p in enumerate(products):
        total = sales[:, i].sum()
        print(f"  Product {p}: {total}")
    
    print(f"\nTotal sales per month:")
    for m, total in zip(months, sales.sum(axis=1)):
        print(f"  {m}: {total}")
    
    print(f"\nBest-selling product: {products[sales.sum(axis=0).argmax()]}")
    print(f"Best month: {months[sales.sum(axis=1).argmax()]}")
    print(f"Worst month: {months[sales.sum(axis=1).argmin()]}")
    
    print(f"\nQuarterly totals:")
    quarters = sales.reshape(4, 3, 5).sum(axis=1)
    for i, q in enumerate(quarters):
        print(f"  Q{i+1}: {q.sum()}")
    
    # Growth analysis
    monthly_totals = sales.sum(axis=1)
    growth = np.diff(monthly_totals) / monthly_totals[:-1] * 100
    print(f"\nMonth-over-month growth (%):")
    for m, g in zip(months[1:], growth):
        print(f"  {m}: {g:+.2f}%")
    
    # Product correlation
    print(f"\nProduct correlations:")
    corr = np.corrcoef(sales.T)
    for i in range(len(products)):
        print(f"  {products[i]}: " + " ".join(f"{corr[i,j]:>6.3f}" for j in range(len(products))))

if __name__ == "__main__":
    sales = generate_sales_data()
    analyze(sales)