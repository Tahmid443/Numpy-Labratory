"""Predict the output of these expressions."""

import numpy as np

def predict(title, expression, expected):
    """Run expression and reveal answer."""
    try:
        actual = eval(expression, {'np': np, 'a': a, 'b': b, 'm': m})
        match = "✓" if np.array_equal(np.asarray(actual), np.asarray(expected)) else "✗"
        print(f"{match} {title}")
        print(f"   Expression: {expression}")
        print(f"   Expected:   {expected}")
        print(f"   Actual:     {actual}")
    except Exception as e:
        print(f"✗ {title}")
        print(f"   Error: {e}")

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2], [3, 4]])
m = np.arange(6).reshape(2, 3)

if __name__ == "__main__":
    print("=" * 60)
    print("PREDICT THE OUTPUT")
    print("=" * 60)
    
    predict("a[2]", "a[2]", 3)
    predict("a[-1]", "a[-1]", 5)
    predict("a[1:3]", "a[1:3]", np.array([2, 3]))
    predict("a[::2]", "a[::2]", np.array([1, 3, 5]))
    predict("a[::-1]", "a[::-1]", np.array([5, 4, 3, 2, 1]))
    predict("a[a > 2]", "a[a > 2]", np.array([3, 4, 5]))
    predict("a * 2", "a * 2", np.array([2, 4, 6, 8, 10]))
    predict("a + 1", "a + 1", np.array([2, 3, 4, 5, 6]))
    predict("a.sum()", "a.sum()", 15)
    predict("a.mean()", "a.mean()", 3.0)
    predict("a.max()", "a.max()", 5)
    predict("a.argmax()", "a.argmax()", 4)
    predict("b[0, 1]", "b[0, 1]", 2)
    predict("b[:, 0]", "b[:, 0]", np.array([1, 3]))
    predict("b.sum(axis=0)", "b.sum(axis=0)", np.array([4, 6]))
    predict("b.sum(axis=1)", "b.sum(axis=1)", np.array([3, 7]))
    predict("b.T", "b.T", np.array([[1, 3], [2, 4]]))
    predict("m.shape", "m.shape", (2, 3))
    predict("m.sum()", "m.sum()", 15)
    predict("m.sum(axis=1)", "m.sum(axis=1)", np.array([3, 12]))
    predict("m[0]", "m[0]", np.array([0, 1, 2]))
    predict("m[:, 1]", "m[:, 1]", np.array([1, 4]))