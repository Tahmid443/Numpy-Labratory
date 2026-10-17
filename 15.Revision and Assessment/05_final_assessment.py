"""Final assessment - comprehensive test."""

import numpy as np

def run_assessment():
    print("=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    print("\nThis assessment covers all topics.")
    print("Solve each problem and verify with the provided check.\n")

    score = 0
    total = 10

    # --- Section 1: Array creation and metadata ---
    print("\n--- Section 1: Array Creation (2 problems) ---")

    # Problem 1.1
    print("\n[1.1] Create a 4x4 matrix with:
    - 1 on the main diagonal
    - 0 everywhere else
    Then verify its shape is (4, 4) and its determinant is 1.")
    print("Solution:")

    def check_1_1(answer):
        global score
        try:
            assert answer.shape == (4, 4), f"Shape should be (4,4), got {answer.shape}"
            assert np.array_equal(answer, np.eye(4)), "Should be identity matrix"
            assert np.isclose(np.linalg.det(answer), 1.0), "Determinant should be 1"
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")

    # Solution
    a = np.eye(4)
    check_1_1(a)

    # Problem 1.2
    print("\n[1.2] Create a 1D array [5, 10, 15, 20, 25] using arange.")
    print("Solution:")

    def check_1_2(answer):
        global score
        try:
            assert np.array_equal(answer, np.array([5, 10, 15, 20, 25]))
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")

    a = np.arange(5, 30, 5)
    check_1_2(a)

    # --- Section 2: Indexing and slicing ---
    print("\n--- Section 2: Indexing (2 problems) ---")

    # Problem 2.1
    print("\n[2.1] Given array = [[1,2,3],[4,5,6],[7,8,9]], extract
    the diagonal [1, 5, 9].")
    print("Solution:")
    
    def check_2_1(answer):
        global score
        try:
            assert np.array_equal(answer, np.array([1, 5, 9]))
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    m = np.arange(1, 10).reshape(3, 3)
    diag = m.diagonal()
    check_2_1(diag)
    
    # Problem 2.2
    print("\n[2.2] From array [1,2,3,4,5,6,7,8,9,10], extract all even numbers.")
    print("Solution:")
    
    def check_2_2(answer):
        global score
        try:
            assert np.array_equal(answer, np.array([2, 4, 6, 8, 10]))
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    a = np.arange(1, 11)
    even = a[a % 2 == 0]
    check_2_2(even)
    
    # --- Section 3: Broadcasting ---
    print("\n--- Section 3: Broadcasting (1 problem) ---")
    
    # Problem 3.1
    print("\n[3.1] Create a multiplication table (9x9) where M[i,j] = (i+1)*(j+1).")
    print("Solution:")
    
    def check_3_1(answer):
        global score
        try:
            expected = np.outer(np.arange(1, 10), np.arange(1, 10))
            assert np.array_equal(answer, expected)
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    table = np.arange(1, 10)[:, np.newaxis] * np.arange(1, 10)
    check_3_1(table)
    
    # --- Section 4: Reductions ---
    print("\n--- Section 4: Reductions (1 problem) ---")
    
    # Problem 4.1
    print("\n[4.1] For matrix m = arange(12).reshape(3,4), compute:
    - row sums
    - column means
    - overall max")
    print("Solution:")
    
    def check_4_1(rows, cols, mx):
        global score
        try:
            assert np.array_equal(rows, np.array([6, 22, 38]))
            assert np.allclose(cols, np.array([4, 5, 6, 7]))
            assert mx == 11
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    m = np.arange(12).reshape(3, 4)
    check_4_1(m.sum(axis=1), m.mean(axis=0), m.max())
    
    # --- Section 5: Shape manipulation ---
    print("\n--- Section 5: Shape (1 problem) ---")
    
    # Problem 5.1
    print("\n[5.1] Reshape array [0..11] into (2, 6), then transpose
    to (6, 2). Verify the (0,1) element of transpose is 2.")
    print("Solution:")
    
    def check_5_1(answer):
        global score
        try:
            assert answer.shape == (6, 2)
            assert answer[0, 1] == 2
            assert answer[1, 0] == 2
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    a = np.arange(12).reshape(2, 6).T
    check_5_1(a)
    
    # --- Section 6: Linalg ---
    print("\n--- Section 6: Linear Algebra (1 problem) ---")
    
    # Problem 6.1
    print("\n[6.1] Solve: 2x + 3y = 8, x + y = 3. Check x=1, y=2.")
    print("Solution:")
    
    def check_6_1(answer):
        global score
        try:
            assert np.allclose(answer, [1, 2])
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    A = np.array([[2, 3], [1, 1]])
    b = np.array([8, 3])
    x = np.linalg.solve(A, b)
    check_6_1(x)
    
    # --- Section 7: Random ---
    print("\n--- Section 7: Random (1 problem) ---")
    
    # Problem 7.1
    print("\n[7.1] Generate 1000 standard normal samples with seed 42.")
    print("Verify the mean is close to 0 (within 0.1).")
    print("Solution:")
    
    def check_7_1(answer):
        global score
        try:
            assert len(answer) == 1000
            assert abs(answer.mean()) < 0.1
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(1000)
    check_7_1(samples)
    
    # --- Section 8: Numerical ---
    print("\n--- Section 8: Numerical (1 problem) ---")
    
    # Problem 8.1
    print("\n[8.1] Check if 0.1 + 0.2 is close to 0.3 using np.isclose.")
    print("Solution:")
    
    def check_8_1(answer):
        global score
        try:
            assert answer == True
            print("  ✓ Correct!")
            score += 1
        except AssertionError as e:
            print(f"  ✗ {e}")
    
    result = np.isclose(0.1 + 0.2, 0.3)
    check_8_1(result)
    
    # --- Results ---
    print("\n" + "=" * 60)
    print(f"FINAL SCORE: {score}/{total}")
    print("=" * 60)
    if score == total:
        print("🎉 Perfect score! You've mastered NumPy!")
    elif score >= total * 0.8:
        print("👍 Excellent! Review the missed topics.")
    elif score >= total * 0.6:
        print("📚 Good effort. Review the missed topics.")
    else:
        print("📖 Keep practicing. Review the topics above.")
    
    return score

if __name__ == "__main__":
    run_assessment()