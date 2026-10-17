"""Analyze student marks."""

import numpy as np

def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    n_students = 30
    n_subjects = 5
    marks = rng.integers(40, 100, size=(n_students, n_subjects))
    return marks

def analyze(marks):
    n_students, n_subjects = marks.shape
    print(f"Number of students: {n_students}")
    print(f"Number of subjects: {n_subjects}")
    print(f"\nMarks matrix:\n{marks}")
    
    print(f"\nSubject-wise statistics:")
    print(f"  Means:    {marks.mean(axis=0)}")
    print(f"  Std:      {marks.std(axis=0)}")
    print(f"  Min:      {marks.min(axis=0)}")
    print(f"  Max:      {marks.max(axis=0)}")
    print(f"  Median:   {np.median(marks, axis=0)}")
    
    print(f"\nStudent-wise statistics:")
    totals = marks.sum(axis=1)
    averages = marks.mean(axis=1)
    print(f"  Totals: {totals}")
    print(f"  Averages: {averages}")
    
    print(f"\nTop 3 students (by average):")
    top_indices = np.argsort(-averages)[:3]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  Rank {rank}: Student {idx} - Average: {averages[idx]:.2f}")
    
    print(f"\nGrade distribution (based on average):")
    grades = np.where(averages >= 90, 'A',
             np.where(averages >= 80, 'B',
             np.where(averages >= 70, 'C',
             np.where(averages >= 60, 'D', 'F'))))
    for g in ['A', 'B', 'C', 'D', 'F']:
        count = (grades == g).sum()
        print(f"  {g}: {count} students")
    
    print(f"\nSubject correlations:")
    print(np.corrcoef(marks.T))
    
    # Pass/fail
    print(f"\nPass (>=40) count: {(marks >= 40).sum()} / {marks.size}")
    
    # Best/worst subjects
    subject_means = marks.mean(axis=0)
    best_subj = subject_means.argmax()
    worst_subj = subject_means.argmin()
    print(f"\nBest subject: {best_subj} (mean {subject_means[best_subj]:.2f})")
    print(f"Worst subject: {worst_subj} (mean {subject_means[worst_subj]:.2f})")

if __name__ == "__main__":
    marks = generate_data()
    analyze(marks)